import copclib as copc


reader = copc.FileReader("/media/nvme/pcdata/autzen/copc/split_0/octree.copc.laz")

# Now, we can create our actual writer, with an optional `span` and `wkt`:
writer = copc.FileWriter(
    "/media/nvme/pcdata/autzen/copc/split_0/autzen-resolution-trimmed.copc.laz",
    reader.copc_config,
)

for node in reader.GetAllNodes():
    if node.key.d == 4:
        points = reader.GetPoints(node)
        for point in points:
            point.Red = node.key.x
            point.Green = node.key.y
            point.Blue = node.key.z
            point.Intensity = node.key.d
            point.PointSourceID = node.key.x + node.key.y
        writer.AddNode(node.key, points)

# Make sure we call close to finish writing the file!
writer.Close()
