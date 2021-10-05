import json
from rockrobotics.utils.get_res_depth import query_depth
import os.path as osp
import random
import argparse

def get_split():
    x = random.random()
    if x > 0.9:
        return "test"
    elif x > 0.8:
        return "val"
    else:
        return "train"

def create_splits(path, res):
    _, targetDepth, metadata = query_depth(osp.join(path, "ept.json"), res)

    heirarchyPath = osp.join(path, "ept-hierarchy", "0-0-0-0.json")    
    with open(heirarchyPath) as f:
        hierarchy = json.load(f)

    depthFiles = [x for x in hierarchy.keys() if int(x[0]) == targetDepth]
    splits = {"train": [], "test": [], "val": []}
    for file in depthFiles:
        splits[get_split()].append(file)

    with open(osp.join(path, "splits.json"), 'w', encoding='utf-8') as f:
        json.dump(splits, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('res', type=float, default=0.5)

    args = parser.parse_args()
    create_splits(args.path, args.res)