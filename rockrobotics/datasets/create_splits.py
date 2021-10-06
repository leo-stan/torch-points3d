import json
import os
import os.path as osp
import random
import copclib as copc
from joblib import Parallel, delayed

resolution = 0.5
datasets = ["lux", "sasche", "tyrol"]
path = "/media/nvme/pcdata"
splits = {'train': 0.8, 'val': 0.1, 'test': 0.1}
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
    dataset_splits = {"train": {}, "test": {}, "val": {}}

    dataset_dir = osp.join(path, dataset, "copc")
    for datafile in os.listdir(dataset_dir):
        datafile_path = osp.join(dataset_dir, datafile, "octree.copc.laz")
        if not osp.exists(datafile_path):
            continue

        try:
            reader = copc.FileReader(datafile_path)
            nodes = reader.GetNodesAtResolution(resolution)
            
            for split in dataset_splits.keys():
                # initialize file split
                dataset_splits[split][datafile] = []

            for node in nodes:
                dataset_splits[get_split()][datafile].append(str(node.key))

            reader.Close()
        except (Exception) as err:
            print(datafile_path + " " + err)

    with open(osp.join(dataset_dir, "splits-%s.json" % version), 'w', encoding='utf-8') as f:
        json.dump(dataset_splits, f, ensure_ascii=False, indent=4)

Parallel(n_jobs=-1)(delayed(split_dataset)(dataset) for dataset in datasets)

with open(osp.join(path, "dataset-%s.json" % version), 'w', encoding='utf-8') as f:
    json.dump(
    {
        "resolution": resolution,
        "datasets": datasets,
        "path": path,
        "splits": splits,
        "version": version,
    }, f, ensure_ascii=False, indent=4)
