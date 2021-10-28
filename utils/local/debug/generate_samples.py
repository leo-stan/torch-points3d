from omegaconf import OmegaConf, DictConfig
import os
import numpy as np
from tqdm import tqdm
import laspy
from joblib import Parallel, delayed

import copclib as copc
from torch_points3d.datasets.segmentation.copc_dataset import CopcDatasetFactoryInference
from utils.dataset_tiles import get_keys

in_file_path = "/media/nvme/other/z/cloud-5e7590d/octree.copc.laz"

data_config = OmegaConf.load(os.path.join(".", "conf/data/segmentation/ground/ground-v1.yaml"))
data_config = OmegaConf.to_container(data_config, resolve=True)["data"]
data_config["is_inference"] = True
# print(data_config)
data_config["inference_file"] = in_file_path

hUnits = 0.30
# Load the tiles from copc file
keys = get_keys(in_file_path, data_config["max_resolution"] / hUnits, data_config["target_tile_size"] / hUnits)

# Instantiate CopcInternalDataset
dataset_factory = CopcDatasetFactoryInference(DictConfig(data_config), keys, hUnits, hUnits)
dataset = dataset_factory.test_dataset[0]

reader = copc.FileReader(in_file_path)
header = reader.copc_config.las_header
# print(header)

# def write_sample(data, i):
for i, data in tqdm(enumerate(dataset)):
    head = laspy.header.Header()
    las = laspy.file.File(os.path.join("/media/nvme/other/samples", "tile%d.laz" % i), mode="w", header=head)
    las.header.scale = [header.scale.x, header.scale.y, header.scale.z]
    # las.header.offset = [header.offset.x, header.offset.y, header.offset.z]
    las.x = np.array(data.pos[:, 0])
    las.y = np.array(data.pos[:, 1])
    las.z = np.array(data.pos[:, 2])
    las.classification = np.array(data.y)
    print("here")
    las.close()


Parallel(n_jobs=-1)(delayed(write_sample)(data, i) for i, data in tqdm(enumerate(dataset)))
