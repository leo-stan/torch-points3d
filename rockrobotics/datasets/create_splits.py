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
target_tile_size = 150
max_resolution = 10
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

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def split_dataset(dataset):

    dataset_dir = osp.join(path, dataset, "copc")
    
    def split_file(datafile):        
        datafile_path = osp.join(dataset_dir, datafile, "octree.copc.laz")
        if not osp.exists(datafile_path):
            return None

        try:
        #if True:
            reader = copc.FileReader(datafile_path)
            header = reader.GetLasHeader()
            max_depth = reader.GetDepthAtResolution(max_resolution)

            span = header.GetSpan()
            nearest_depth = round(math.log2(span/target_tile_size))
            num_voxels = 2**nearest_depth
            tile_size = span / num_voxels
            print(tile_size)

            min_z = reader.GetLasHeader().min.z
            max_z = reader.GetLasHeader().max.z
            max_z_coord = math.ceil((max_z - min_z) / (span / 2**nearest_depth))

            datafile_splits = {"train": {}, "test": {}, "val": {}}

            keys = np.array([[node.key.d, node.key.x, node.key.y, node.key.z] for node in reader.GetAllChildren()])
            keys_in_range = keys[keys[:,0] == max_depth][:,1:]
            keys_in_range_min = keys_in_range * 2**(nearest_depth-max_depth)
            keys_in_range_max = keys_in_range * 2**(nearest_depth-max_depth)+np.sum(2 ** np.arange(nearest_depth-max_depth))
            keys_in_range_max[:,2] = np.minimum(keys_in_range_max[:,2], max_z_coord)

            x_range = [np.arange(a, b) for a,b in zip(keys_in_range_min[:,0], keys_in_range_max[:,0])]
            y_range = [np.arange(a, b) for a,b in zip(keys_in_range_min[:,1], keys_in_range_max[:,1])]
            z_range = [np.arange(a, b) for a,b in zip(keys_in_range_min[:,2], keys_in_range_max[:,2])]

            test_keys = {str((nearest_depth, x, y)):zs.tolist() for xs, ys, zs in zip(x_range, y_range, z_range) for x,y in cartesian_product(xs, ys)}
            
            for k,v in test_keys.items():
                if len(v) > 0:
                    datafile_splits[get_split()][str(k)] = v
            
            reader.Close()
            return (datafile, datafile_splits)
        except:
            print("Error on file: " + datafile_path)
    
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
