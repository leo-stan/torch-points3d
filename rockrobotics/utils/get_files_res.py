from rockrobotics.utils.get_res_depth import query_depth
import os.path as osp
import json


def get_filelist(ept_path, targetRes):
    _, targetDepth, metadata = query_depth(osp.join(ept_path, "ept.json"), targetRes)

    heirarchyPath = osp.join(ept_path, "ept-hierarchy", "0-0-0-0.json")
    with open(heirarchyPath) as f:
        hierarchy = json.load(f)

    depthFiles = [x for x in hierarchy.keys() if int(x[0]) == targetDepth]

    cubeSize = metadata["bounds"][3] - metadata["bounds"][0]

    depthHierarchy = {}
    for file in depthFiles[:1]:
        depth, x, y, z = [int(x) for x in file.split("-")]
        node = []

        dx, dy, dz = x, y, z
        for currentDepth in reversed(range(depth)):
            dx, dy, dz = dx // 2, dy // 2, dz // 2

            fname = "{}-{}-{}-{}".format(currentDepth, dx, dy, dz)
            node.append(fname)
        depthHierarchy[file] = node

        minx, miny, minz = metadata["bounds"][:3]
        currentSpan = cubeSize / pow(2, depth)

        minx = minx + x * currentSpan
        miny = miny + y * currentSpan
        minz = minz + z * currentSpan

        maxx, maxy, maxz = minx + currentSpan, miny + currentSpan, minz + currentSpan
        print(minx, miny, minz)
        print(maxx, maxy, maxz)

    print(depthHierarchy)


if __name__ == "__main__":
    queryRes = 0.5
    path = "/mnt/c/data/potreeconvert/ahn"
    get_filelist(path, queryRes)
