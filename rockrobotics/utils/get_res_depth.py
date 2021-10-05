import json

def query_depth(path, queryRes):
    with open(path) as f:
        metadata = json.load(f)

    cubeSize = metadata["bounds"][3] - metadata["bounds"][0]
    span = metadata["span"]
    maxDepth = metadata["octreeDepth"]

    currentRes = cubeSize / span
    currentDepth = 0
    while currentRes > queryRes and currentDepth < maxDepth:
        currentRes /= 2
        currentDepth += 1

    return currentRes, currentDepth, metadata

if __name__ == "__main__":
    queryRes = 0.5
    path = "C:\\data\\potreeconvert\\ahn\\ept.json"
    currentRes, currentDepth = query_depth(path, queryRes)
    print(currentRes)
    print(queryRes)
    print(currentDepth)