import os
from tqdm import tqdm
import subprocess

in_dir = "/media/lssd/test-data"
out_dir = "/media/nvme/test-data"

in_files = [x for x in os.listdir(in_dir) if x.lower().endswith((".laz", ".las"))]
in_files2 = [
    os.path.join("feet", x) for x in os.listdir(os.path.join(in_dir, "feet")) if x.lower().endswith((".laz", ".las"))
]
in_files.extend(in_files2)

for file in tqdm(in_files):
    out = os.path.join(out_dir, file)
    subprocess.run(
        [
            "/processing/PotreeConverter/build/PotreeConverter",
            os.path.join(in_dir, file),
            "-o",
            os.path.join(out_dir, file),
            "--no-addons",
        ]
    )

for root, subdirs, files in os.walk(out_dir):
    for file in files:
        if file == "addon.copc.laz":
            os.remove(os.path.join(root, file))
        elif file == "octree.copc.laz":
            temp_file = os.path.abspath(os.path.join(root, "..", file))
            os.rename(os.path.join(root, file), temp_file)
            os.rmdir(root)
            os.rename(temp_file, root[:-4] + ".copc.laz")
