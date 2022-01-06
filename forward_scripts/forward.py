import torch
import logging
from omegaconf import OmegaConf, DictConfig
import os
import numpy as np
import wandb
import concurrent.futures
from tqdm import tqdm
import time
import argparse

import copclib as copc

# Import BaseModel / BaseDataset for type checking
from torch_points3d.models.base_model import BaseModel

# Import from metrics
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

# Utils import
from utils.dataset_tiles import get_keys
from utils.upsampling.upsample import upsample_labels_grid3d

from torch_points3d.datasets.segmentation.copc_dataset import CopcDatasetFactoryInference
from torch_points3d.core.data_transform import SaveOriginalPosId

log = logging.getLogger(__name__)


class CopcInference:
    def __init__(
        self,
        checkpoint_dir,
        model_name="ResUNet32",
        metric="miou",
        num_workers=12,
        batch_size=1,
        debug=False,
        upsample=False,
        confidence_threshold=0.0,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.metric = metric
        self.num_workers = num_workers
        self.batch_size = 1  # batch size of 1 is only supported
        self.debug = debug
        self.upsample = upsample
        self.confidence_threshold = confidence_threshold

    def _update_node_predictions(self, compressed_points, node, changed_idx, prediction, file_header, override_all):
        """Decompress a node, update its classifications based on the model predictions, and recompress the node

        Args:
            compressed_points (str): Raw byte string of compressed byte data for that node
            node (copclib.Node): The current node
            changed_idx (np array): List of changed point indices
            prediction (np array): List of predictions for each changed_idx
            file_header (copclib.LasHeader): las header info
            override_all (bool): If true, overrides all of the existing classifications in this node
                by setting all to 0.

        Raises:
            RuntimeError: [description]

        Returns:
            (str, copclib.Node): The output compressed byte string and the node for reference
        """
        uncompressed_copc_points = copc.DecompressBytes(compressed_points, file_header, node.point_count)
        copc_points = copc.Points.Unpack(uncompressed_copc_points, file_header)

        # if we're not overriding all the classificaitons, this node should have at least one changed prediction
        # or else it should have been written out by CopcInference._write_unchanged_nodes
        if len(changed_idx) == 0 and not override_all:
            raise RuntimeError("No points were classified for this node!", node, changed_idx, prediction)

        if override_all:
            # start from a fresh slate
            classifications_new = np.zeros(node.point_count)
        else:
            # update the existing classifications
            classifications_new = np.copy(copc_points.classification)

        # update the node, if needed (if override_all is true, then changed_idx may be empty)
        if len(changed_idx) > 0:
            classifications_new[changed_idx] = prediction

        # update classifications
        copc_points.classification = classifications_new.astype(int)

        # recompress points
        compressed_points = copc.CompressBytes(copc_points.Pack(), file_header)
        return compressed_points, node

    def _get_predictions(self, model, dataset, device, reverse_class_map):
        """Runs a dataset through inference and generate the true class predictions

        Args:
            model (BaseModel): Model to run through inference
            dataset (BaseDataset): Dataset to run through inference
            device (str): What device to run inference on (cuda/cpu)
            reverse_class_map (list): A list of lists of two elements each which maps between
                raw prediction classes (first element) and the output class we want to put in the LAS file

        Returns:
            dict: A mapping between nodes (represented as a key string) and a tuple of lists,
            where the first list is a point index relative to all of the points in the node,
            and the second list is that point index's predicted classification
        """
        key_prediction_map_list = []
        for dataloader in dataset.test_dataloaders:
            iter_data_time = time.time()
            with Ctq(dataloader) as tq_test_loader:
                # run forward on each batch
                for data in tq_test_loader:
                    # timing
                    t_data = time.time() - iter_data_time
                    iter_start_time = time.time()

                    with torch.no_grad():
                        # run data through model
                        output = self._do_model_inference(model, device, data)

                        # timing
                        t_iter = time.time() - iter_start_time
                        t_other_start = time.time()

                        # compute node/point predictions from model results
                        batch_key_prediction_map = self._extract_model_predictions(reverse_class_map, data, output)
                        # add the predictions for this batch to the list
                        key_prediction_map_list.append(batch_key_prediction_map)

                    # update progress bar
                    t_other = time.time() - t_other_start
                    tq_test_loader.set_postfix(
                        data_loading=float(t_data), iteration=float(t_iter), other=float(t_other)
                    )
                    iter_data_time = time.time()

        return self._merge_batch_predictions(key_prediction_map_list)

    def _merge_batch_predictions(self, key_prediction_map_list):
        """Creates a large dictionary of key:(changed_idx, preds) out of a list of smaller dictionaries

        Args:
            key_prediction_map_list (list[dict]): A list of dictionaries with key:(changed_idx,preds)

        Returns:
            dict: one large key:(changed_idx,preds) dict
        """

        key_prediction_map = {}
        # iterate through each batch dictionary and merge them into a global dictionary
        for map in key_prediction_map_list:
            for key_str, (idx, label) in map.items():
                if key_str not in key_prediction_map:
                    key_prediction_map[key_str] = ([], [])

                key_prediction_map[key_str][0].extend(idx)
                key_prediction_map[key_str][1].extend(label)

        return key_prediction_map

    def _extract_model_predictions(self, reverse_class_map, data, output):
        """From a raw model output (pre-softmax), use the reverse_class_map to generate the true class predictions

        Args:
            reverse_class_map (list): A list of lists of two elements each which maps between
                raw prediction classes (first element) and the output class we want to put in the LAS file
            data (pyg.Batch): The current batch data
            output (tensor): The pre-softmax model output

        Returns:
            dict: key:(changed_idx,preds) dict
        """
        # run softmax on the outputs and check the confidence level
        probs = torch.nn.functional.softmax(output, dim=1)
        probs_max, preds = torch.max(probs, dim=1)
        preds = preds + 1  # make room for the ignore class at class 0
        preds[probs_max <= self.confidence_threshold] = 0  # set unconfident predictions to 0

        # convert the dense classes into ASPRS classes
        new_labels = preds.clone()
        for source, target in reverse_class_map:
            mask = preds == source
            new_labels[mask] = target
        new_labels = new_labels.cpu().numpy()

        return self._upsample_preds_and_convert_to_dict(data, new_labels)

    def _upsample_preds_and_convert_to_dict(self, data, new_labels):
        """From a tensor of label predictions, find each label's original Node and the index of that label within
        all of the points of that node

        Args:
            data (pyg.Batch): Batch data
            new_labels (tensor): vector of predictions

        Returns:
            dict: key:(changed_idx,preds) dict
        """
        origin_idx = data[SaveOriginalPosId.KEY].cpu().numpy()
        points_key = data.points_key.cpu().numpy()
        points_idx_node = data.points_idx.cpu().numpy()

        # upsample if needed
        # else limit the points that are changed to just those that didn't get downsampled in inference
        if self.upsample:
            new_labels = upsample_labels_grid3d(
                data.pos_orig, data.grid_size.item(), data.grid_start.cpu(), new_labels, origin_idx
            )
        else:
            points_key = points_key[origin_idx]
            points_idx_node = points_idx_node[origin_idx]

        batch_key_prediction_map = {}
        # map each changed point to a key in batch_key_prediction_map
        for key, idx, label in zip(points_key, points_idx_node, new_labels):
            key_str = tuple(key)
            if key_str not in batch_key_prediction_map:
                batch_key_prediction_map[key_str] = ([], [])

            # idx 0 is for the point index, idx 1 is for the point label
            batch_key_prediction_map[key_str][0].append(idx)
            batch_key_prediction_map[key_str][1].append(label)

        # return the batch predictions
        return batch_key_prediction_map

    def _do_model_inference(self, model, device, data):
        model.set_input(data, device)
        model.forward()
        return model.get_output()

    def _run_inference(
        self,
        model: BaseModel,
        dataset,
        device,
        in_path,
        out_path,
        reverse_class_map,
        override_all,
    ):
        """From a dataset and model, run inference on a file and write those predictions out to a new file."""
        # Load input file
        reader = copc.FileReader(in_path)

        print("RUNNING INFERENCE ON FILE: %s" % in_path)
        # 'key_prediction_map' is a dictionary that maps voxel_key: (changed_point_indices, changed_point_class)
        key_prediction_map = self._get_predictions(model, dataset, device, reverse_class_map)

        print("WRITING OUTPUT FILE: %s" % out_path)

        writer = copc.FileWriter(out_path, reader.copc_config)

        # for all the nodes that haven't been classified, we can write them out directly
        # (this should only be nodes whose depth is greater than our min_resolution)
        # when debug flag is set, only write the classified points to the file
        if not self.debug:
            self._write_unchanged_nodes(override_all, reader, key_prediction_map, writer)

        # for nodes that have been classified, we need to decompress and get its points,
        # update the classifications, recompress, and write to the file
        # we use the method from
        # https://github.com/RockRobotic/rockconvert/blob/f03b1a73964bd2049a26d2e83dce7c5a4e78b5b6/process/copc/copcReproject.py#L107

        with tqdm(total=len(key_prediction_map)) as progress:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = []
                depth = reader.GetDepthAtResolution(dataset.dataset_opt.resolution)
                # for each node in the key_prediction_map, launch a future that updates its classification
                for key_str in list(key_prediction_map.keys()):
                    changed_idx, prediction = key_prediction_map[key_str]

                    key = copc.VoxelKey(*list(key_str))
                    node = reader.FindNode(key)
                    compressed_points = reader.GetPointDataCompressed(key)
                    # with the debug flag, we only want to write out changed nodes, and no nodes greater than
                    # the target depth are changed.
                    if self.debug and key.d > depth:
                        continue

                    future = executor.submit(
                        self._update_node_predictions,
                        compressed_points,
                        node,
                        np.array(changed_idx),
                        np.array(prediction),
                        writer.copc_config.las_header,
                        override_all,
                    )
                    future.add_done_callback(lambda _: progress.update())
                    futures.append(future)

                    # empty memory (microoptimziation)
                    del key_prediction_map[key_str]

                # as each future finishes, write the updated node points out to the file
                for fut in concurrent.futures.as_completed(futures):
                    compressed_points, node = fut.result()
                    writer.AddNodeCompressed(node.key, compressed_points, node.point_count, node.page_key)

        writer.Close()
        print("DONE RUNNING INFERENCE!")

    def _write_unchanged_nodes(self, override_all, reader, key_prediction_map, writer):
        """If we want to keep the existing predictions for the file, we can write out all the nodes
        that don't have predictions associated them without decompressing them

        Args:
            override_all (bool): override all classifications or not
            reader (copc.Reader): copc reader
            key_prediction_map (dict): key:(changed_idx,preds) dict
            writer (copc.Writer): copc writer
        """
        # if we want to override all the point's classifications and start fresh,
        # we need to make nodes_not_changed empty so that all nodes get decompressed and rewritten
        all_nodes = reader.GetAllNodes()
        if override_all:
            for node in all_nodes:
                key_str = (node.key.d, node.key.x, node.key.y, node.key.z)
                if key_str not in key_prediction_map:
                    key_prediction_map[key_str] = ([], [])

        # get a list of all the nodes that don't have a prediction mapped to them
        nodes_not_changed = [
            node for node in all_nodes if (node.key.d, node.key.x, node.key.y, node.key.z) not in key_prediction_map
        ]
        # for all nodes that don't have a prediction, we can write them unchanged to the output file.
        for node in nodes_not_changed:
            compressed_points = reader.GetPointDataCompressed(node.key)
            writer.AddNodeCompressed(node.key, compressed_points, node.point_count, node.page_key)

    def run_inference(
        self,
        in_file_path,
        out_file_path,
        wandb_run,
        cuda,
        hUnits,
        vUnits,
        override_all=False,
    ):
        """Public interface for running inference on a copc file given a wandb model

        Args:
            in_file_path (str): path to the input copc file
            out_file_path (str): full path to output copc file
            wandb_run (str): the wandb run to use for inference
            cuda (bool): Whether to run the inference on the gpu or not
            hUnits (float): Scaling factor applied to convert the copc file units to meters
            vUnits (float): Scaling factor applied to convert the copc file (vertical) units to meters
            override_all (bool, optional): If true, all classifications in the copc file will be set to 0
                and overridden. If False, the existing classifications will remain and only the points ran through the
                ML model will be overridden. Defaults to False.

        Raises:
            ValueError: Invalid arguments are provided
        """
        if not os.path.exists(os.path.dirname(out_file_path)):
            os.makedirs(os.path.dirname(out_file_path))

        if not os.path.exists(in_file_path):
            raise ValueError("The input file does not exist! %s" % in_file_path)

        if not torch.cuda.is_available() and cuda:
            print("WARNING: Cuda was selected for inference, but cuda is not available.")
            cuda = False

        device = torch.device("cuda" if cuda else "cpu")
        print("DEVICE : {}".format(device))

        torch.backends.cudnn.enabled = True

        self._init_model_from_wandb(in_file_path, out_file_path, wandb_run, hUnits, vUnits, override_all, device)

    def _init_model_from_wandb(self, in_file_path, out_file_path, wandb_run, hUnits, vUnits, override_all, device):
        if wandb_run:
            print("Downloading wandb run weights...")
            checkpoint_dir = os.path.join(self.checkpoint_dir, wandb_run)
            wandb.restore(self.model_name + ".pt", run_path=wandb_run, root=checkpoint_dir)

        print("Loading checkpoint...")
        checkpoint = ModelCheckpoint(checkpoint_dir, self.model_name, self.metric, strict=True)

        data_config = OmegaConf.to_container(checkpoint.data_config, resolve=True)
        data_config["is_inference"] = True

        model = checkpoint.create_model(DictConfig(checkpoint.dataset_properties), weight_name=self.metric)
        model.eval()
        model = model.to(device)
        print("Model size = %i" % sum(param.numel() for param in model.parameters() if param.requires_grad))

        dataset = self._init_dset(in_file_path, hUnits, vUnits, data_config, model)
        print(dataset)

        self._run_inference(
            model,
            dataset,
            device,
            in_file_path,
            out_file_path,
            checkpoint.data_config.reverse_class_map,
            override_all,
        )

    def _init_dset(self, in_file_path, hUnits, vUnits, data_config, model):
        """Creates a CopcDataset from a COPC file for inference

        Args:
            in_file_path (str): full path to input copc file
            hUnits (float): horizontal scaling factor
            vUnits (float): vertical scaling factor
            data_config (dict): Dictionary of config values generated from the omegaconf config file that got saved
                for the model we're running inference on
            model (BaseModel): the pytorch model to run inference on

        Returns:
            CopcForwardDataset: dataset to run inference on
        """
        data_config["inference_file"] = in_file_path

        # Load the tiles from copc file
        keys = get_keys(in_file_path, data_config["max_resolution"] / hUnits, data_config["target_tile_size"] / hUnits)

        # Instantiate CopcInternalDataset
        dataset = CopcDatasetFactoryInference(DictConfig(data_config), keys, hUnits, vUnits, data_config["resolution"])
        dataset.create_dataloaders(
            model,
            self.batch_size,
            False,
            self.num_workers,
            False,
        )

        print("Scaling horizontal by %f and vertical by %f" % (hUnits, vUnits))
        dataset.test_dataset[0].hunits = hUnits
        dataset.test_dataset[0].vunits = vUnits

        return dataset


def get_args():
    """helper function that reads args from cmd line

    Returns:
        args: args obj
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--in_file_path",
        type=str,
        default="/media/ml1_nvme/pcdata/autzen/split_0/octree.copc.laz",
        help="Absolute path to file to process",
    )
    parser.add_argument(
        "-o",
        "--out_file_path",
        type=str,
        default="/media/ml1_nvme/pcdata/autzen/split_0/inference.copc.laz",
        help="Absolute path to file to save",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/media/ml2_processing/torch-points3d/wandb",
        help="Absolute path to checkpoint directory",
    )
    parser.add_argument("--model_name", type=str, default="ResUNet32", help="Model Name")
    parser.add_argument(
        "--wandb_run",
        type=str,
        default="rock-robotic/building-v1/15e45z0x",
        # default="rock-robotic/COPC-ground-v1/31kc98wj",
        help="Wandb run",
    )
    parser.add_argument("--metric", type=str, default="miou", help="miou")
    parser.add_argument("--cuda", dest="cuda", action="store_true")
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    parser.set_defaults(cuda=True)
    parser.add_argument("--num_workers", type=int, default=12, help="Number of CPU workers")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size")
    parser.add_argument("--hUnits", type=float, default=0.30, help="Horizontal Units")
    parser.add_argument("--vUnits", type=float, default=0.30, help="Vertical Units")
    parser.add_argument("--confidence_threshold", type=float, default=0.8, help="Confidence Threshold")
    parser.add_argument("--no_override_all", dest="override_all", action="store_false", help="Override All")
    parser.add_argument("--upsample", action="store_true", help="upsample")
    parser.add_argument("--debug", action="store_true", help="debug flag")
    parser.set_defaults(override_all=True)
    parser.set_defaults(upsample=False)
    parser.set_defaults(debug=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # the defaults of all these arguments are probably fine for prod (change upsample on/off as desired)
    inference = CopcInference(
        args.checkpoint_dir,
        model_name=args.model_name,
        metric=args.metric,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        debug=args.debug,
        upsample=args.upsample,
        confidence_threshold=args.confidence_threshold,
    )

    # all these args are required except override_all
    inference.run_inference(
        in_file_path=args.in_file_path,
        out_file_path=args.out_file_path,
        cuda=args.cuda,
        wandb_run=args.wandb_run,
        hUnits=args.hUnits,
        vUnits=args.vUnits,
        override_all=args.override_all,
    )
