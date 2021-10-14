import json
import os
import os.path as osp
import random
import copclib as copc
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
import math


def m_to_ft(m):
    return m * 3.281


def get_split():
    x = 1 - random.random()

    if x < splits["test"]:
        return "test"
    elif x < splits["test"] + splits["val"]:
        return "val"
    else:
        return "train"


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def get_keys(datafile_path, max_resolution, target_tile_size):
    reader = copc.FileReader(datafile_path)
    header = reader.GetLasHeader()
    max_depth = reader.GetDepthAtResolution(max_resolution)

    span = header.GetSpan()
    nearest_depth = round(math.log2(span / target_tile_size))
    num_voxels = 2 ** nearest_depth
    tile_size = span / num_voxels

    min_z = reader.GetLasHeader().min.z
    max_z = reader.GetLasHeader().max.z
    max_z_coord = math.ceil((max_z - min_z) / (span / 2 ** nearest_depth))

    keys = np.array([[node.key.d, node.key.x, node.key.y, node.key.z] for node in reader.GetAllChildren()])
    keys_in_range = keys[keys[:, 0] == max_depth][:, 1:]
    keys_in_range_min = keys_in_range * 2 ** (nearest_depth - max_depth)
    keys_in_range_max = keys_in_range * 2 ** (nearest_depth - max_depth) + np.sum(
        2 ** np.arange(nearest_depth - max_depth)
    )
    keys_in_range_max[:, 2] = np.minimum(keys_in_range_max[:, 2], max_z_coord)

    x_range = [np.arange(a, b + 1) for a, b in zip(keys_in_range_min[:, 0], keys_in_range_max[:, 0])]
    y_range = [np.arange(a, b + 1) for a, b in zip(keys_in_range_min[:, 1], keys_in_range_max[:, 1])]
    z_range = [np.arange(a, b + 1) for a, b in zip(keys_in_range_min[:, 2], keys_in_range_max[:, 2])]

    test_keys = {
        str((nearest_depth, x, y)): zs.tolist()
        for xs, ys, zs in zip(x_range, y_range, z_range)
        for x, y in cartesian_product(xs, ys)
    }

    reader.Close()

    return test_keys


def split_file(datafile_path, datafile, max_resolution, target_tile_size):
    # if True:
    try:
        datafile_splits = {"train": {}, "test": {}, "val": {}}

        test_keys = get_keys(datafile_path, max_resolution, target_tile_size)
        for k, v in test_keys.items():
            if len(v) > 0:
                datafile_splits[get_split()][str(k)] = v

        return (datafile, datafile_splits)
    except:
        print("Error on file: " + datafile_path)
        return None


def split_dataset(dataset, max_resolution, target_tile_size):
    dataset_dir = osp.join(path, dataset, "copc")

    data_files = [
        (osp.join(dataset_dir, datafile, "octree.copc.laz"), datafile) for datafile in os.listdir(dataset_dir)
    ]
    data_files = [tuple for tuple in data_files if osp.exists(tuple[0])]
    splits_list = Parallel(n_jobs=-1)(
        delayed(split_file)(datafile_path, datafile, max_resolution, target_tile_size)
        for datafile_path, datafile in tqdm(data_files)
    )
    splits_list = [split for split in splits_list if split is not None]

    all_splits = {"train": {}, "test": {}, "val": {}}
    for datafile, datafile_splits in splits_list:
        for split, datafile_split in datafile_splits.items():
            all_splits[split][datafile] = datafile_split

    with open(osp.join(dataset_dir, "splits-%s.json" % version), "w", encoding="utf-8") as f:
        json.dump(all_splits, f)


if __name__ == "__main__":
    # params
    # TODO: load this from tp3d dataset config!
    resolution = 0.5
    target_tile_size = 150
    max_resolution = 10
    datasets = ["lux", "sasche", "tyrol"]
    path = "/media/nvme/pcdata"
    splits = {"train": 0.9, "val": 0.05, "test": 0.05}
    version = "v1"

    for dataset in datasets:
        split_dataset(dataset, max_resolution, target_tile_size)

    with open(osp.join(path, "dataset-%s.json" % version), "w", encoding="utf-8") as f:
        json.dump(
            {
                "resolution": resolution,
                "datasets": datasets,
                "path": path,
                "splits": splits,
                "version": version,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )
