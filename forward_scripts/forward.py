import torch
import logging
from omegaconf import OmegaConf, DictConfig
import os
import numpy as np
import wandb
import concurrent.futures
from tqdm import tqdm
import ast
import argparse
from itertools import islice
import time

import copclib as copc

# Import BaseModel / BaseDataset for type checking
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset

# Import from metrics
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint

# Utils import
from utils.dataset_tiles import get_keys

from torch_points3d.datasets.segmentation.copc_dataset import CopcDatasetFactoryInference

log = logging.getLogger(__name__)


def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}


def update_node_predictions(compressed_points, node, changed_idx, prediction, file_header, override_all):
    uncompressed_copc_points = copc.DecompressBytes(compressed_points, file_header, node.point_count)
    copc_points = copc.Points.Unpack(uncompressed_copc_points, file_header)

    classifications_new = np.zeros(node.point_count)
    classifications_new[changed_idx] = prediction

    classifications = np.array(copc_points.Classification)
    mask = classifications <= 2 if not override_all else [True] * len(classifications)
    classifications[mask] = classifications_new[mask]
    copc_points.Classification = classifications.astype(int)

    # for point in copc_points:
    #     point.Red = node.key.x
    #     point.Green = node.key.y
    #     point.Blue = node.key.z
    #     point.Intensity = node.key.d
    #     point.PointSourceID = node.key.x + node.key.y

    compressed_points = copc.CompressBytes(copc_points.Pack(), file_header)

    return compressed_points, node


def get_sample_attribute(data, key, sample_idx, conv_type):
    return BaseDataset.get_sample(data, key, sample_idx, conv_type).cpu().numpy()


def do_inference(model, dataset, device, confidence_threshold, reverse_class_map, key_prediction_map):
    loaders = dataset.test_dataloaders
    for loader in loaders:
        iter_data_time = time.time()
        with Ctq(loader) as tq_test_loader:
            # run forward on each batch
            for data in tq_test_loader:
                t_data = time.time() - iter_data_time
                iter_start_time = time.time()
                with torch.no_grad():
                    model.set_input(data, device)
                    model.forward()

                    t_iter = time.time() - iter_start_time
                    t_other_start = time.time()
                    # add the predictions as a sample attribute
                    output = model.get_output()
                    num_batches = BaseDataset.get_num_samples(data, model.conv_type).item()
                    setattr(data, "_pred", output)
                    # iterate through each sample and update key_prediction_map
                    # TODO: this part is slow?
                    for sample_idx in range(num_batches):
                        prediction = get_sample_attribute(data, "_pred", sample_idx, model.conv_type)
                        points_key = get_sample_attribute(data, "points_key", sample_idx, model.conv_type)
                        points_idx_node = get_sample_attribute(data, "points_idx", sample_idx, model.conv_type)

                        # run softmax on the outputs and check the confidence level
                        probs = torch.nn.functional.softmax(torch.from_numpy(np.copy(prediction)), dim=1)
                        probs_max, preds = torch.max(probs, dim=1)
                        preds = preds + 1  # make room for the ignore class at class 0
                        preds[probs_max <= confidence_threshold] = 0  # set unconfident predictions to 0

                        # convert the dense classes into ASPRS classes
                        new_labels = preds.clone()
                        for source, target in reverse_class_map:
                            mask = preds == source
                            new_labels[mask] = target
                        new_labels = new_labels.numpy()

                        # update key_prediction_map
                        for key, idx, label in zip(points_key, points_idx_node, new_labels):
                            key_str = str(tuple(key))
                            if key_str not in key_prediction_map:
                                key_prediction_map[key_str] = ([], [])

                            key_prediction_map[key_str][0].append(idx)
                            key_prediction_map[key_str][1].append(label)
                t_other = time.time() - t_other_start
                tq_test_loader.set_postfix(data_loading=float(t_data), iteration=float(t_iter), other=float(t_other))
                iter_data_time = time.time()


def run(
    model: BaseModel,
    dataset,
    device,
    in_path,
    out_path,
    reverse_class_map,
    confidence_threshold,
    override_all=True,
    debug=False,
):
    # Load input file
    reader = copc.FileReader(in_path)
    all_nodes = reader.GetAllChildren()

    key_prediction_map = {}

    print("RUNNING INFERENCE ON FILE: %s" % in_path)
    do_inference(model, dataset, device, confidence_threshold, reverse_class_map, key_prediction_map)

    print("WRITING OUTPUT FILE: %s" % out_path)

    cfg = copc.LasConfig(
        reader.GetLasHeader(),
        reader.GetExtraByteVlr(),
    )
    writer = copc.FileWriter(out_path, cfg, reader.GetCopcHeader().span, reader.GetWkt())

    # if we want to override all the point's classifications and start fresh,
    # make sure the nodes_not_changed list is empty
    if override_all:
        for node in all_nodes:
            key_str = str((node.key.d, node.key.x, node.key.y, node.key.z))
            if key_str not in key_prediction_map:
                key_prediction_map[key_str] = ([], [])

    # for all the nodes that haven't been classified, we can write them out directly
    # (this should only be nodes whose depth is greater than our min_resolution)
    if not debug:
        nodes_not_changed = [
            node
            for node in all_nodes
            if str((node.key.d, node.key.x, node.key.y, node.key.z)) not in key_prediction_map
        ]
        for node in nodes_not_changed:
            compressed_points = reader.GetPointDataCompressed(node.key)
            writer.AddNodeCompressed(writer.GetRootPage(), node.key, compressed_points, node.point_count)

    # for nodes that have been classified, we need to decompress and get its points,
    # update the classifications, recompress, and write to the file
    # we use the method from
    # https://github.com/RockRobotic/rockconvert/blob/f03b1a73964bd2049a26d2e83dce7c5a4e78b5b6/process/copc/copcReproject.py#L107

    with tqdm(total=len(key_prediction_map)) as progress:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            depth = reader.GetDepthAtResolution(dataset.dataset_opt.resolution)
            # for each node in the key_prediction_map, launch a future that updates its classification
            for key_str, (changed_idx, prediction) in key_prediction_map.items():
                # tuple string to tuple
                key = ast.literal_eval(key_str)
                key = copc.VoxelKey(*list(key))
                if debug and key.d > depth:
                    continue
                node = reader.FindNode(key)
                compressed_points = reader.GetPointDataCompressed(key)

                future = executor.submit(
                    update_node_predictions,
                    compressed_points,
                    node,
                    changed_idx,
                    prediction,
                    writer.GetLasHeader(),
                    override_all,
                )
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            # as each future finishes, write the updated node points out to the file
            for fut in concurrent.futures.as_completed(futures):
                compressed_points, node = fut.result()
                writer.AddNodeCompressed(writer.GetRootPage(), node.key, compressed_points, node.point_count)

    writer.Close()
    print("DONE RUNNING INFERENCE!")


def predict_file(
    in_file_path,
    out_file_path,
    checkpoint_dir,
    model_name="ResUNet32",
    wandb_run="",
    metric="miou",
    cuda=True,
    num_workers=2,
    batch_size=1,
    hUnits=1.0,
    vUnits=1.0,
    confidence_threshold=0.0,
    override_all=True,
    debug=False,
):
    if not os.path.exists(os.path.dirname(out_file_path)):
        os.makedirs(os.path.dirname(out_file_path))

    device = torch.device("cuda" if (torch.cuda.is_available() and cuda) else "cpu")
    print("DEVICE : {}".format(device))

    torch.backends.cudnn.enabled = True

    if wandb_run:
        print("Downloading wandb run weights...")
        checkpoint_dir = os.path.join(checkpoint_dir, wandb_run)
        wandb.restore(model_name + ".pt", run_path=wandb_run, root=checkpoint_dir)

    print("Loading checkpoint...")
    checkpoint = ModelCheckpoint(checkpoint_dir, model_name, metric, strict=True)

    data_config = OmegaConf.to_container(checkpoint.data_config, resolve=True)
    data_config["is_inference"] = True
    model = checkpoint.create_model(DictConfig(checkpoint.dataset_properties), weight_name=metric)

    print("Model size = %i", sum(param.numel() for param in model.parameters() if param.requires_grad))

    data_config["inference_file"] = in_file_path

    # Load the tiles from copc file
    keys = get_keys(in_file_path, data_config["max_resolution"] / hUnits, data_config["target_tile_size"] / hUnits)

    # Instanciate CopcInternalDataset
    dataset = CopcDatasetFactoryInference(DictConfig(data_config), keys)
    dataset.create_dataloaders(
        model,
        batch_size,
        False,
        num_workers,
        False,
    )
    print(dataset)

    print("Scaling horizontal by %f and vertical by %f" % (hUnits, vUnits))
    dataset.test_dataset[0].hunits = hUnits
    dataset.test_dataset[0].vunits = vUnits

    model.eval()
    model = model.to(device)

    # Predict
    run(
        model,
        dataset,
        device,
        in_file_path,
        out_file_path,
        checkpoint.data_config.reverse_class_map,
        confidence_threshold,
        override_all,
        debug,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_file_path",
        type=str,
        default="/media/nvme/pcdata/autzen/copc/split_0/octree.copc.laz",
        help="Absolute path to file to process",
    )
    parser.add_argument(
        "--out_file_path",
        type=str,
        default="/media/nvme/pcdata/autzen/copc/split_0/inference.copc.laz",
        help="Absolute path to file to save",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/processing/torch-points3d/outputs/2021-10-11/15-20-55",
        help="Absolute path to checkpoint directory",
    )
    parser.add_argument("--model_name", type=str, default="ResUNet32", help="Model Name")
    parser.add_argument(
        "--wandb_run",
        type=str,
        default="",
        help="Wandb run",
    )
    parser.add_argument("--metric", type=str, default="miou", help="miou")
    parser.add_argument("--cuda", type=bool, default=True, help="cuda use flag")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of CPU workers")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size")
    parser.add_argument("--h_units", type=float, default=1.0, help="Horizontal Units")
    parser.add_argument("--v_units", type=float, default=1.0, help="Vertical Units")
    parser.add_argument("--confidence_threshold", type=float, default=0.8, help="Confidence Threshold")
    parser.add_argument("--override_all", action="store_true", help="Override All")
    parser.add_argument("--debug", action="store_true", help="debug flag")

    args = parser.parse_args()

    predict_file(
        args.in_file_path,
        args.out_file_path,
        args.checkpoint_dir,
        args.model_name,
        args.wandb_run,
        args.metric,
        args.cuda,
        args.num_workers,
        args.batch_size,
        args.h_units,
        args.v_units,
        args.confidence_threshold,
        args.override_all,
        args.debug,
    )
