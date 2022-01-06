import copclib as copc
import numpy as np
from sys import float_info

# given a key, recursively check if each of its 8 children exist in the hierarchy
def get_all_key_children(key, max_depth, hierarchy):
    # stop once we reach max_depth, since none of its children can exist
    if key.d >= max_depth:
        return []
    children = []
    for child in key.GetChildren():
        if str(child) in hierarchy:
            # if the key exists, add it to the output, and check if any of its children exist
            children.append(hierarchy[str(child)])
            grandchildren = get_all_key_children(child, max_depth, hierarchy)
            children.extend(grandchildren)
    return children


def get_all_points(reader, header, sample_bounds, valid_child_nodes, valid_parent_nodes, track_inverse=False):
    # Process keys that exist
    copc_points = copc.Points(header)
    # this can be converted to points_key_idx by indexing hierarchy
    points_key = []
    points_idx = []

    # Load the node and all its child points
    for node in valid_child_nodes.values():
        node_points = reader.GetPoints(node)

        # track point for inference
        if track_inverse:
            points_key.append(np.full((len(node_points), 4), [node.key.d, node.key.x, node.key.y, node.key.z]))
            points_idx.append(np.arange(len(node_points)))

        copc_points.AddPoints(node_points)

        # For parents node we need to check which points fit within bounds
    for node in valid_parent_nodes.values():
        node_points = reader.GetPoints(node)
        key = np.array([[node.key.d, node.key.x, node.key.y, node.key.z]])
        for i, point in enumerate(node_points):
            if point.Within(sample_bounds):
                copc_points.AddPoint(point)

                if track_inverse:
                    points_key.append(key)
                    points_idx.append([i])

    return copc_points, points_key, points_idx


def get_valid_nodes(possible_zs, hierarchy, max_depth, nearest_depth, x, y):
    # key:node mappings
    valid_child_nodes = {}
    valid_parent_nodes = {}
    # check every z to see if it, or any of its parents, have points in it
    for i, z in enumerate(possible_zs):
        start_key = copc.VoxelKey(nearest_depth, x, y, z)

        # start by checking if the node itself exists
        if str(start_key) in hierarchy:
            # if the node exists, then get all its children
            child_nodes = get_all_key_children(start_key, max_depth, hierarchy)
            for child_node in child_nodes:
                valid_child_nodes[str(child_node.key)] = child_node

            # add the node itself
            valid_child_nodes[str(start_key)] = hierarchy[str(start_key)]
            start_key = start_key.GetParent()

            # if the node doens't exist, find the first parent that does exist
        else:
            while str(start_key) not in hierarchy:
                if start_key.d < 0:
                    raise RuntimeError("This shouldn't happen!")
                start_key = start_key.GetParent()

        # then, get all nodes from depth 0 to the current depth
        key = start_key
        while key.IsValid():
            valid_parent_nodes[str(key)] = hierarchy[str(key)]
            key = key.GetParent()

    return valid_child_nodes, valid_parent_nodes

def get_sample_bounds(d, x, y, header):
    query_key = copc.VoxelKey(d, x, y, 0)
    sample_bounds = copc.Box(query_key, header)
    sample_bounds.z_min = -float_info.max
    sample_bounds.z_max = float_info.max
    return sample_bounds