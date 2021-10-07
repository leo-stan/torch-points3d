import json
import os
import os.path as osp
import random
import copclib as copc
from joblib import Parallel, delayed
import numpy as np
from torch.functional import _return_counts
from tqdm import tqdm
import math

def m_to_ft(m):
    return m * 3.281

resolution = 0.5
target_tile_size = 100
max_resolution = 5
datasets = ["lux", "sasche", "tyrol"]
path = "/media/nvme/pcdata"
splits = {'train': 0.9, 'val': 0.05, 'test': 0.05}
version = "v1"

def get_split():
    x = 1 - random.random()

    if x < splits['test']:
        return "test"
    elif x < splits['test'] + splits['val']:
        return "val"
    else:
        return "train"

def split_dataset(dataset):

    dataset_dir = osp.join(path, dataset, "copc")
    
    def split_file(datafile):        
        datafile_path = osp.join(dataset_dir, datafile, "octree.copc.laz")
        if not osp.exists(datafile_path):
            return None

        #try:     
        if True:
            reader = copc.FileReader(datafile_path)
            header = reader.GetLasHeader()
            depth = reader.GetDepthAtResolution(resolution)
            max_depth = reader.GetDepthAtResolution(max_resolution)
            max_depth = reader.GetDepthAtResolution(max_resolution)

            span = header.GetSpan()
            nearest_depth = math.floor(math.log2(span/target_tile_size))
            num_voxels = 2**nearest_depth
            tile_size = span / num_voxels

            datafile_splits = {"train": {}, "test": {}, "val": {}}

            bounds = header.GetBounds()

            min_bounds = np.array([bounds.x_min, bounds.y_min, bounds.z_min])

            min = np.array(min_bounds, copy=True)
            
            xy = {}

            for x in range(num_voxels):
                min[0] = min_bounds[0] + tile_size * x
                for y in range(num_voxels):
                    xy[(nearest_depth, x,y)] = []
                    min[1] = min_bounds[1] + tile_size * y
                    for z in range(num_voxels):
                        min[2] = min_bounds[2] + tile_size * z
                        max = min + tile_size

                        if min[0] > bounds.x_max or min[1] > bounds.y_max or min[2] > bounds.z_max or max[0] < bounds.x_min or max[1] < bounds.y_min or max[2] < bounds.z_min:
                            continue

                        key = copc.VoxelKey(nearest_depth, x, y, z)

                        while key.d >= max_depth:
                            if reader.FindNode(key).IsValid():
                                xy[(nearest_depth, x,y)].append(z)
                                #datafile_splits[get_split()].append([nearest_depth, x, y, z])
                                break
                            key = key.GetParent()
            
            for k,v in xy.items():
                if len(v) > 0:
                    datafile_splits[get_split()][str(k)] = v
            
            reader.Close()
            return (datafile, datafile_splits)
    
    x = Parallel(n_jobs=-1)(delayed(split_file)(datafile) for datafile in tqdm(os.listdir(dataset_dir)))
    dataset_splits = {"train": {}, "test": {}, "val": {}}
    x = [z for z in x if z is not None]
    for f, d in x:
        for split in dataset_splits.keys():
            dataset_splits[split][f] = d[split]

    with open(osp.join(dataset_dir, "splits-%s.json" % version), 'w', encoding='utf-8') as f:
        json.dump(dataset_splits, f)

for dataset in datasets:
    split_dataset(dataset)
#Parallel(n_jobs=-1)(delayed(split_dataset)(dataset) for dataset in datasets)

with open(osp.join(path, "dataset-%s.json" % version), 'w', encoding='utf-8') as f:
    json.dump(
    {
        "resolution": resolution,
        "datasets": datasets,
        "path": path,
        "splits": splits,
        "version": version,
    }, f, ensure_ascii=False, indent=4)
