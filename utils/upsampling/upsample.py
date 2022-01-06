import concurrent
import numpy as np
import torch
import pandas as pd
import copclib as copc
from tqdm import tqdm
import ast

from torch_cluster import grid_cluster
from torch_points3d.datasets.segmentation.copc_dataset import get_all_points
from utils.dataset_helpers import *
from utils.dataset_tiles import get_keys


def upsample_labels_grid3d(pos_orig, grid_size, grid_start, new_labels, origin_idx):
    # upsample the labels into the points that got filtered by GridSampling3D,
    # i.e. upsample predictions into all tiles from depth 0 to the target depth.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # not sure if this matters

    # first, quantize the raw coords
    coords = torch.round(pos_orig.to(device) / grid_size)
    cluster = (
        grid_cluster(
            coords.to(device), torch.tensor([1, 1, 1], device=device, dtype=torch.float32), grid_start.to(device)
        )
        .cpu()
        .numpy()
    )

    df = pd.DataFrame({"cluster": cluster, "labels": np.zeros(len(cluster))})
    df.loc[origin_idx, "labels"] = new_labels
    upsampled = df.groupby("cluster").transform("max")  # aggregate based on "max"

    new_labels = upsampled["labels"].to_numpy()
    return new_labels


# this function initializes each process with a reader object
def mp_init_readers(in_path, function):
    function.reader = copc.FileReader(in_path)


# given a child_node and the global tile_coords_dict, upsample the child's labels
def do_upsample(child_node, hUnits, resolution):
    # get the child points
    child_points = do_upsample.reader.GetPoints(child_node)

    # load pos & quantize
    child_pos = np.stack([child_points.x, child_points.y, child_points.z], axis=1) / hUnits
    child_coords = np.round(child_pos / resolution).astype(np.int64)

    # lookup each x,y,z pair in the dictionary to get the new labels
    # (default the classification to 0 if the x,y,z voxel isn't found)
    child_labels = [tile_coords_dict.get(str(tuple(xyz)), 0) for xyz in child_coords]
    child_points.classification = child_labels

    # recompress and return
    compressed_points = copc.CompressBytes(child_points.Pack(), do_upsample.reader.copc_config.las_header)
    return compressed_points, child_node


def upsample_tiles(in_path, out_path, data_config, hUnits):
    # Load input file
    reader = copc.FileReader(in_path)
    header = reader.copc_config.las_header

    # Create output writer
    writer = copc.FileWriter(out_path, reader.copc_config)

    # smallest voxel size the model was trained on
    resolution = data_config["resolution"] / hUnits
    # the depth of the smallest possible node that could be in a tile, given a resolution
    max_tile_depth = reader.GetDepthAtResolution(resolution)
    # the lowest depth in the entire file
    max_file_depth = reader.GetDepthAtResolution(-1)
    # used for dataset functions
    hierarchy = {str(node.key): node for node in reader.GetAllNodes()}
    # keys is a list of tiles at the specified target tile size, each of which can have multiple possible z levels
    # tile_depth is the depth of the target tile size
    keys, tile_depth = get_keys(
        in_path, data_config["max_resolution"] / hUnits, data_config["target_tile_size"] / hUnits
    )

    root_tiles = {}
    # iterate through every tile that was passed through the model
    for dxy, zs in keys.items():
        d, x, y = ast.literal_eval(dxy)
        for z in zs:
            key = copc.VoxelKey(d, x, y, z)
            # if this particular tile actually has a node associated with it, then it may have children points
            # if there's no node, it won't have any children past what has already been through the model
            if str(key) in hierarchy:
                if not dxy in root_tiles:
                    root_tiles[dxy] = []

                root_tiles[dxy].append(z)

    # all these nodes already have classifications, so we can pass them directly through
    passthrough_nodes = [x for x in reader.GetAllNodes() if x.key.d <= max_tile_depth]
    for node in passthrough_nodes:
        compressed_points = reader.GetPointDataCompressed(node.key)
        writer.AddNodeCompressed(node.key, compressed_points, node.point_count, node.page_key)

    # now, we upsample every node that wasn't classified by the model
    for dxy, zs in tqdm(root_tiles.items()):
        d, x, y = ast.literal_eval(dxy)
        sample_bounds = get_sample_bounds(d, x, y, header)

        # pull the tile's labels and coordinates up to the level that was classified by the model
        valid_child_nodes, valid_parent_nodes = get_valid_nodes(zs, hierarchy, max_tile_depth, tile_depth, x, y)
        tile_points, _, _ = get_all_points(
            reader, header, sample_bounds, valid_child_nodes, valid_parent_nodes, track_inverse=False
        )
        tile_pos = np.stack([tile_points.x, tile_points.y, tile_points.z], axis=1) / hUnits
        tile_labels = np.array(tile_points.classification, dtype=np.uint8)

        # quantize the tile ccoordinates to our resolution
        tile_coords = np.round(tile_pos / resolution).astype(np.int64)

        # create the label dictionary as a global, so that each process receives a copy of it
        global tile_coords_dict
        tile_coords_dict = {str(tuple(xyz)): l for xyz, l in zip(tile_coords, tile_labels)}

        with concurrent.futures.ProcessPoolExecutor(
            initializer=mp_init_readers,
            initargs=(
                in_path,
                do_upsample,
            ),
        ) as executor:
            # iterate through each known valid z coordinate
            for z in zs:
                root_key = copc.VoxelKey(d, x, y, z)
                # grab all the children of this node
                child_nodes = get_all_key_children(root_key, max_file_depth, hierarchy)
                # upsample each child node
                futures = []
                for child_node in child_nodes:
                    future = executor.submit(
                        do_upsample,
                        child_node,
                        hUnits,
                        resolution,
                    )
                    futures.append(future)

                # write our upsampled child node out
                for fut in concurrent.futures.as_completed(futures):
                    compressed_points, child_node = fut.result()
                    writer.AddNodeCompressed(
                        child_node.key, compressed_points, child_node.point_count, child_node.page_key
                    )

    # done
    writer.Close()


if __name__ == "__main__":
    from omegaconf import OmegaConf

    dataset_options = OmegaConf.load("conf/data/segmentation/ground/ground-v1.yaml")
    upsample_tiles(
        in_path="/media/ml2_nvme/test-data/Chambers-rural-classified.copc.laz",
        out_path="/media/ml2_nvme/test-data/Chambers-rural-classified.copc-upsample.laz",
        data_config=dataset_options["data"],
        hUnits=1,
    )
