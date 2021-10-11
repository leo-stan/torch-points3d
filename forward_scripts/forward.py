import torch
import logging
from omegaconf import OmegaConf
import os
import numpy as np
import wandb
import concurrent.futures
from tqdm import tqdm

# Import BaseModel / BaseDataset for type checking
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset

# Import from metrics
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint
from torch_points3d.core.data_transform import SaveOriginalPosId

log = logging.getLogger(__name__)
import laspy
import copclib as copc


def update_node_predictions(compressed_points, node, changed_idx, prediction, file_header):
    uncompressed_copc_points = copc.DecompressBytes(compressed_points, file_header, node.point_count)
    copc_points = copc.Points.Unpack(uncompressed_copc_points, file_header)

    classifications = np.array(copc_points.Classification)
    classifications[changed_idx] = prediction

    compressed_points = copc.CompressBytes(copc_points.Pack(), file_header)

    return compressed_points, node


def get_sample_attribute(data, key, sample_idx, conv_type):
    return BaseDataset.get_sample(data, key, sample_idx, conv_type).cpu().numpy()


def run(model: BaseModel, dataset, device, in_path, out_path, reverse_class_map, confidence_threshold):
    # Load input file
    reader = copc.FileReader(in_path)
    all_nodes = reader.GetAllChildren()

    key_prediction_map = {}

    print("RUNNING INFERENCE ON FILE: %s" % in_path)
    loaders = dataset.test_dataloaders
    for loader in loaders:
        loader.dataset.name
        with Ctq(loader) as tq_test_loader:
            # run forward on each batch
            for data in tq_test_loader:
                with torch.no_grad():
                    model.set_input(data, device)
                    model.forward()

                    # add the predictions as a sample attribute
                    output = model.get_output()
                    num_batches = BaseDataset.get_num_samples(data, model.conv_type).item()
                    setattr(data, "_pred", output)

                    # iterate through each sample and update key_prediction_map
                    for sample_idx in range(num_batches):
                        prediction = get_sample_attribute(data, "_pred", sample_idx, model.conv_type)
                        points_idx_downsample = get_sample_attribute(
                            data, SaveOriginalPosId.KEY, sample_idx, model.conv_type
                        )
                        points_key = get_sample_attribute(data, "points_key", sample_idx, model.conv_type)
                        points_idx_node = get_sample_attribute(data, "points_idx", sample_idx, model.conv_type)

                        # run softmax on the outputs and check the confidence level
                        probs = torch.nn.functional.softmax(torch.from_numpy(np.copy(prediction)), dim=1)
                        probs_max, preds = torch.max(probs, dim=1)
                        preds = preds + 1  # make room for the ignore class at class 0
                        preds[probs_max <= confidence_threshold] = 0  # set unconfident predictions to 0

                        # convert the dense classes into ASPRS classes
                        new_labels = preds.clone()
                        for source, target in reverse_class_map.items():
                            mask = preds == source
                            new_labels[mask] = target

                        # update key_prediction_map
                        changed_idx = points_idx_node[points_idx_downsample]
                        key_prediction_map[points_key] = (changed_idx, new_labels)

    cfg = copc.LasConfig(
        reader.GetLasHeader(),
        reader.GetExtraByteVlr(),
    )
    writer = copc.FileWriter(out_path, cfg, reader.GetCopcHeader().span, reader.GetWkt())

    print("WRITING OUTPUT FILE: %s" % out_path)

    # for all the nodes that haven't been classified, we can write them out directly
    # (this should only be nodes whose depth is greater than our min_resolution)
    nodes_not_changed = [node for node in all_nodes if node not in key_prediction_map]
    for node in nodes_not_changed:
        compressed_points = reader.GetPointDataCompressed(node.key)
        writer.AddNodeCompressed(writer.GetRootPage(), node.key, compressed_points, node.point_count)

    # for nodes that have been classified, we need to decompress and get its points,
    # update the classifications, recompress, and write to the file
    # we use the method from
    # https://github.com/RockRobotic/rockconvert/blob/f03b1a73964bd2049a26d2e83dce7c5a4e78b5b6/process/copc/copcReproject.py#L107

    with concurrent.futures.ProcessPoolExecutor() as executor:
        with tqdm(total=len(key_prediction_map)) as progress:
            futures = []
            # for each node in the key_prediction_map, launch a future that updates its classification
            for key, (changed_idx, prediction) in key_prediction_map:
                node = reader.FindNode(key)
                compressed_points = reader.GetPointDataCompressed(key)
                future = executor.submit(update_node_predictions, compressed_points, node, changed_idx, prediction)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            # as each future finishes, write the updated node points out to the file
            for fut in concurrent.futures.as_completed(futures):
                compressed_points, node = fut.result()
                writer.AddNodeCompressed(writer.GetRootPage(), node.key, compressed_points, node.point_count)

    print("DONE RUNNING INFERENCE!")


def predict_folder(
    in_folder,
    out_folder,
    wandb_run,
    wandb_dir,
    model_name="ResUNet32",
    metric="miou",
    cuda=False,
    num_workers=2,
    batch_size=1,
    hUnits=1.0,
    vUnits=1.0,
    debug=False,
    confidence_threshold=0.0,
):
    device = torch.device("cuda" if (torch.cuda.is_available() and cuda) else "cpu")
    print("DEVICE : {}".format(device))

    torch.backends.cudnn.enabled = True

    print("Download run weights...")
    wandb_dir = os.path.join(wandb_dir, wandb_run)
    wandb.restore(model_name + ".pt", run_path=wandb_run, root=wandb_dir)

    print("Loading checkpoint...")
    checkpoint = ModelCheckpoint(wandb_dir, model_name, metric, strict=True)

    print("Setting dataset directory to: %s" % in_folder)
    setattr(checkpoint.data_config, "dataroot", in_folder)
    setattr(checkpoint.data_config, "is_test", True)

    model = checkpoint.create_model(checkpoint.dataset_properties, weight_name=metric)
    # log.info(model)
    print("Model size = %i", sum(param.numel() for param in model.parameters() if param.requires_grad))

    dataset = instantiate_dataset(checkpoint.data_config)
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

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    run(model, dataset, device, out_folder, debug, confidence_threshold)


def predict_folder_local(
    in_folder,
    out_folder,
    checkpoint_dir,
    model_name="ResUNet32",
    metric="miou",
    cuda=False,
    num_workers=2,
    batch_size=1,
    debug=False,
    confidence_threshold=0.0,
):
    device = torch.device("cuda" if (torch.cuda.is_available() and cuda) else "cpu")
    print("DEVICE : {}".format(device))

    torch.backends.cudnn.enabled = True

    print("Loading checkpoint...")
    checkpoint = ModelCheckpoint(checkpoint_dir, model_name, metric, strict=True)

    print("Setting dataset directory to: %s" % in_folder)
    setattr(checkpoint.data_config, "dataroot", in_folder)
    setattr(checkpoint.data_config, "is_test", True)

    model = checkpoint.create_model(checkpoint.dataset_properties, weight_name=metric)
    # log.info(model)
    print("Model size = %i", sum(param.numel() for param in model.parameters() if param.requires_grad))

    dataset = instantiate_dataset(checkpoint.data_config)
    dataset.create_dataloaders(
        model,
        batch_size,
        False,
        num_workers,
        False,
    )
    print(dataset)

    model.eval()
    model = model.to(device)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    run(model, dataset, device, out_folder, debug, confidence_threshold)


if __name__ == "__main__":
    predict_folder(
        "/media/machinelearning/machine-learning/test-data/split/canyon",
        "/media/machinelearning/machine-learning/test-data/test-out",
        "rock-robotic/ground-v1/gdq4kqj0",
        wandb_dir="/media/machinelearning/machine-learning/torch-points3d/wandb",
        model_name="ResUNet32",
        metric="miou",
        cuda=False,
        num_workers=8,
        batch_size=8,
    )
