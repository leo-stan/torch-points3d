import concurrent.futures
import json
import os
import os.path as osp
import random
import copclib as copc
import numpy as np
from tqdm import tqdm
import math
import sys
import ast
from datetime import datetime
import time
from utils.dataset_helpers import get_valid_nodes
import logging as log
log.basicConfig(level=log.INFO,format='[%(asctime)s][%(levelname)s] %(message)s',datefmt='%H:%M:%S')

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


def mp_init_readers(function, in_path, hierarchy, max_depth):
    """
    Initializes each process with a reader object
    """
    function.reader = copc.FileReader(in_path)
    function.hierarchy = hierarchy
    function.max_depth = max_depth


def compute_tile_hist_worker(dxy, z):
    """
    Compute class histogram for a particular tile located at d,x,y,z
    Args:
        dxy (str): String containing "(d,x,y)" values of the octree key
        z (int): z value of the octree key

    Returns (dict): [dxy,z,label histogram] or None if no points are found in that tile

    """
    reader = compute_tile_hist_worker.reader
    header = reader.copc_config.las_header

    # Extract nearest depth, x, and y from sample
    nearest_depth, x, y = ast.literal_eval(dxy)
    # make a 2d box out of the voxelkey x,y
    sample_bounds = copc.Box(copc.VoxelKey(nearest_depth, x, y, 0), header)
    sample_bounds.z_min = -sys.float_info.max
    sample_bounds.z_max = sys.float_info.max

    valid_child_nodes, valid_parent_nodes = get_valid_nodes(z, compute_tile_hist_worker.hierarchy, compute_tile_hist_worker.max_depth,
                                                            nearest_depth, x, y)
    ### Check each valid node ###

    # Process keys that exist
    tile_points = copc.Points(header)

    # Load the node and all its child points
    # print("%s - Checking points in %d children nodes..." % (datetime.now().strftime("%H:%M:%S"),len(valid_child_nodes)))
    for node in valid_child_nodes.values():
        node_points = reader.GetPoints(node)
        tile_points.AddPoints(node_points)

    # For parents node we need to check which points fit within bounds
    # print("%s - Checking points in %d parent nodes..." % (datetime.now().strftime("%H:%M:%S"),len(valid_parent_nodes)))
    for key in list(valid_parent_nodes.keys()):
        node_points = reader.GetPoints(valid_parent_nodes[key]).GetWithin(sample_bounds)
        # If there are points within the tile bounds then add them to the overall count
        if len(node_points) > 0:
            tile_points.AddPoints(node_points)
        # If not, remove the node from list of valid candidates for this tile
        else:
            # print("Empty parent Node Removed")
            del valid_parent_nodes[key]

    # Check that there are points in tile
    # print("%s - Checking that tile has points and return..." % datetime.now().strftime("%H:%M:%S"))
    if len(tile_points) > 0:
        # Compute label ratio into a dictionary
        labels, histogram = np.unique(tile_points.classification, return_counts=True)
        # Return labels and histogram
        return dxy, z, {label: label_count for label, label_count in
                                            zip(labels.tolist(), histogram.tolist())}
    else:
        # If there are no points in the tile then signal to remove it from test_keys
        return None


def compute_dataset_histogram(datafile_path, test_keys, hierarchy, max_depth):
    """
    Computes the label histogram by going through the test keys and checking all points
    Args:
        datafile_path (str): Path to current dataset copc file being processed
        test_keys (dict): Dictionary of possible octree keys
        hierarchy (dict): Octree hierarchy
        max_depth (int): Maximum allowed depth based on maximum resolution

    Returns (dict): Dictionary of possible octree keys along with the label histogram for each key.

    """
    valid_tiles = {}
    # Parallel process each tile key
    log.info("Checking Tiles")
    with concurrent.futures.ProcessPoolExecutor(
            initializer=mp_init_readers,
            initargs=(
                    compute_tile_hist_worker,
                    datafile_path,
                    hierarchy,
                    max_depth,
            ),
    ) as executor:

        futures = [executor.submit(
            compute_tile_hist_worker,
            dxy,
            z["z"],
        ) for dxy, z in list(test_keys.items())]

        # write our upsampled child node out
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(test_keys)):
            result = fut.result()
            if result is not None:
                # Write histogram in the dictionary
                valid_tiles[result[0]] = {
                    "z": result[1],
                    "class_hist": result[2],
                }
    return valid_tiles


def get_keys(datafile_path, max_resolution, target_tile_size, include_label_hist=False):
    """
    Return keys corresponding to possible tiles at the requested resolution in the octree.
    Args:
        datafile_path (str): Path to current dataset copc file being processed
        max_resolution (float): Highest resolution at which keys are loaded
        target_tile_size (float): Targeted size for the tile
        include_label_hist (bool, optional): Flag to include histogram of labels in each proposed tile

    Returns (dict): A dictionary of possible octree keys at the targeted tile size

    """
    log.info("Loading COPC file")
    reader = copc.FileReader(datafile_path)
    header = reader.copc_config.las_header
    max_depth = reader.GetDepthAtResolution(max_resolution)

    span = header.Span()
    nearest_depth = round(math.log2(span / target_tile_size))
    # If the file span is smaller than the tile size then use the root node
    if span < target_tile_size:
        return {str((0, 0, 0)): [0]}, 0

    min_z = reader.copc_config.las_header.min.z
    max_z = reader.copc_config.las_header.max.z
    max_z_coord = math.ceil((max_z - min_z) / (span / 2 ** nearest_depth))
    log.info("Loading all keys in node")
    keys = np.array([[node.key.d, node.key.x, node.key.y, node.key.z] for node in reader.GetAllNodes()])
    log.info("Filtering keys")
    keys_in_range = keys[keys[:, 0] == max_depth][:, 1:]
    keys_in_range_min = keys_in_range * 2 ** (nearest_depth - max_depth)
    keys_in_range_max = keys_in_range * 2 ** (nearest_depth - max_depth) + np.sum(
        2 ** np.arange(nearest_depth - max_depth)
    )
    keys_in_range_max[:, 2] = np.minimum(keys_in_range_max[:, 2], max_z_coord)

    x_range = [np.arange(a, b + 1) for a, b in zip(keys_in_range_min[:, 0], keys_in_range_max[:, 0])]
    y_range = [np.arange(a, b + 1) for a, b in zip(keys_in_range_min[:, 1], keys_in_range_max[:, 1])]
    z_range = [np.arange(a, b + 1) for a, b in zip(keys_in_range_min[:, 2], keys_in_range_max[:, 2])]
    log.info("Putting keys in a dictionary")
    test_keys = {
        str((nearest_depth, x, y)): {"z": zs.tolist()}
        for xs, ys, zs in zip(x_range, y_range, z_range)
        for x, y in cartesian_product(xs, ys)
    }

    # Compute label histogram if requested for training sets and return it
    if include_label_hist:
        log.info("Computing dataset label histogram...")
        log.info("Getting File Hierarchy")
        hierarchy = {str(node.key): node for node in reader.GetAllNodes()}
        max_depth = reader.GetDepthAtResolution(0.5)
        reader.Close()
        return compute_dataset_histogram(datafile_path, test_keys, hierarchy, max_depth)

    return test_keys


def split_file(datafile_path, datafile, max_resolution, target_tile_size, include_label_hist):
    """
    Gets the possible keys for a particular dataset and splits them into train/val/test sets
    Args:
        datafile_path (str): Path to current dataset copc file being processed
        datafile (str): Dataset name
        max_resolution (float): Highest resolution at which keys are loaded
        target_tile_size (float): Targeted size for the tile
        include_label_hist (bool, optional): Flag to include histogram of labels in each proposed tile

    Returns (dict): Dictionary of possible keys for each split

    """
    datafile_splits = {"train": {}, "test": {}, "val": {}}

    test_keys = get_keys(datafile_path, max_resolution, target_tile_size, include_label_hist)
    for k, v in test_keys.items():
        if len(v) > 0:
            datafile_splits[get_split()][str(k)] = v

    return (datafile, datafile_splits)


def split_dataset(dataset, max_resolution, target_tile_size, include_label_hist=False):
    dataset_dir = osp.join(path, dataset)

    data_files = [
        (osp.join(dataset_dir, datafile, "octree.copc.laz"), datafile) for datafile in os.listdir(dataset_dir)
    ]
    data_files = [tuple for tuple in data_files if osp.exists(tuple[0])]

    splits_list = []
    for datafile_path, datafile in data_files:
        log.info("Processing %s" % datafile)
        split = split_file(datafile_path, datafile, max_resolution, target_tile_size, include_label_hist)
        if split is not None:
            splits_list.append(split)

    all_splits = {"train": {}, "test": {}, "val": {}}
    for datafile, datafile_splits in splits_list:
        for split, datafile_split in datafile_splits.items():
            all_splits[split][datafile] = datafile_split

    with open(osp.join(dataset_dir, "splits-%s.json" % version), "w", encoding="utf-8") as f:
        json.dump(all_splits, f)


if __name__ == "__main__":
    # params
    resolution = 0.5
    target_tile_size = 150
    max_resolution = 10
    datasets = [
        "tyrol",
        "lux",
        "basel",
        "montreal",
        # "sasche",
        # "zurich",
        # "autzen",
    ]
    path = "/media/ml2_nvme/pcdata"
    splits = {"train": 0.9, "val": 0.05, "test": 0.05}
    version = "v2"
    start_time = time.perf_counter()
    for dataset in datasets:
        log.info("Processing %s" % dataset)
        split_dataset(dataset, max_resolution, target_tile_size, include_label_hist=True)
    print("Finished in %f seconds." % (time.perf_counter() - start_time))
    with open(osp.join(path, "dataset-%s.json" % version), "w", encoding="utf-8") as f:
        json.dump(
            {
                "resolution": resolution,
                "target_tile_size": target_tile_size,
                "max_resolution": max_resolution,
                "datasets": datasets,
                "path": path,
                "splits": splits,
                "version": version,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )
