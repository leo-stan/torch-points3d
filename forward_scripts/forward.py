import torch
import hydra
import logging
from omegaconf import OmegaConf
import os
import numpy as np
import wandb
from joblib import Parallel, delayed
import traceback
import matplotlib.cm as cm
import matplotlib as mpl

# Import building function for model and dataset
from torch_points3d.datasets.dataset_factory import instantiate_dataset, get_dataset_class
from torch_points3d.models.model_factory import instantiate_model

# Import BaseModel / BaseDataset for type checking
from torch_points3d.models.base_model import BaseModel
from torch_points3d.datasets.base_dataset import BaseDataset

# Import from metrics
from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq
from torch_points3d.metrics.model_checkpoint import ModelCheckpoint
from torch_points3d.core.data_transform import SaveOriginalPosId

# Utils import
from torch_points3d.utils.colors import COLORS

log = logging.getLogger(__name__)
import laspy
import copclib as copc


def save_file(filename, output_path, predicted, origindid, debug, confidence_threshold):
    reader = copc.FileReader(filename)
    file_dir = os.path.basename(os.path.dirname(filename))
    out_dir = os.path.join(output_path, file_dir)
    try:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    except FileExistsError:
        pass

    cfg = copc.LasConfig(
        reader.GetLasHeader(),
        reader.GetExtraByteVlr(),
    )
    # Create the COPC writer
    writer = copc.FileWriter(
        os.path.join(out_dir, os.path.basename(filename)), cfg, reader.GetCopcHeader().span, reader.GetWkt()
    )

    # TODO
    # For each copc split file
    # Load all modified nodes for file
    # For each Reader node
    # Check if node was modified
    # if modified
    # Update the point's classification
    # Get the predicted classification
    # Reverse map the classification to the dataset
    # Write the node

    # classifications = np.zeros(len(las.points), dtype=np.uint8)
    # probs = torch.nn.functional.softmax(torch.from_numpy(np.copy(predicted)), dim=1)
    # probs_max, preds = torch.max(probs, dim=1)
    #
    # preds[probs_max <= confidence_threshold] = -1
    #
    # classifications[origindid] = preds + 1
    # classifications[las_out.classification == 7] = 7
    # las_out.classification = classifications
    writer.Close()


def save_file_debug(filename, output_path, predicted, origindid, debug, confidence_threshold):
    las = laspy.read(filename)
    file_dir = os.path.basename(os.path.dirname(filename))
    out_dir = os.path.join(output_path, file_dir)
    try:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    except FileExistsError:
        pass
    las_out = laspy.LasData(las.header)
    las_out.points = np.copy(las.points[origindid])

    classifications = np.zeros(len(las_out.points), dtype=np.uint8)
    probs = torch.nn.functional.softmax(torch.from_numpy(np.copy(predicted)), dim=1)
    probs_max, preds = torch.max(probs, dim=1)

    preds[probs_max <= confidence_threshold] = -1

    colors = np.zeros((len(las_out.points), 4), dtype=np.uint16)
    colors = cm.plasma(probs_max.numpy()) * 65535

    las_out.red = colors[:, 0]
    las_out.green = colors[:, 1]
    las_out.blue = colors[:, 2]

    classifications = preds + 1
    classifications[las_out.classification == 7] = 7
    las_out.classification = classifications.numpy()

    las = None
    las_out.write(os.path.join(out_dir, os.path.basename(filename)))


def run(model: BaseModel, dataset, device, output_path, debug, confidence_threshold):
    loaders = dataset.test_dataloaders
    for loader in loaders:
        loader.dataset.name
        with Ctq(loader) as tq_test_loader:
            for data in tq_test_loader:
                try:
                    with torch.no_grad():
                        model.set_input(data, device)
                        model.forward()

                        save_args = []

                        output = model.get_output()
                        num_batches = BaseDataset.get_num_samples(data, model.conv_type).item()
                        setattr(data, "_pred", output)
                        for sample in range(num_batches):
                            predicted = BaseDataset.get_sample(data, "_pred", sample, model.conv_type).cpu().numpy()
                            # TODO
                            origindid = (
                                BaseDataset.get_sample(data, SaveOriginalPosId.KEY, sample, model.conv_type)
                                .cpu()
                                .numpy()
                            )
                            filename = dataset.test_dataset[0].files[data.file[sample][0]]
                            save_args.append(
                                {
                                    "filename": filename,
                                    "output_path": output_path,
                                    "predicted": predicted,
                                    "origindid": origindid,
                                    "debug": debug,
                                    "confidence_threshold": confidence_threshold,
                                }
                            )

                        save_fn = save_file
                        if debug:
                            save_fn = save_file_debug
                        Parallel(n_jobs=num_batches)(delayed(save_fn)(**save_args[i]) for i in range(num_batches))

                except Exception:
                    print("Error processing file with error:")
                    print(traceback.format_exc())
            print("done!")


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
