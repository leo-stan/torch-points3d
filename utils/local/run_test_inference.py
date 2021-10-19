from forward_scripts.forward import predict_file
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--wandb_run",
    type=str,
    required=True,
    help="Wandb run",
)
args = parser.parse_args()

in_dir = "/media/nvme/test-data"
out_dir = "/media/nvme/test-preds"
checkpoints = "/media/nvme/checkpoints"

in_files = [x for x in os.listdir(in_dir) if x.lower().endswith((".laz", ".las"))]
in_files2 = [
    os.path.join("feet", x) for x in os.listdir(os.path.join(in_dir, "feet")) if x.lower().endswith((".laz", ".las"))
]
in_files.extend(in_files2)

for file in in_files:
    units = 0.3048 if file.startswith("feet") else 1
    predict_file(
        os.path.join(in_dir, file),
        os.path.join(out_dir, file),
        checkpoints,
        "ResUNet32",
        "rock-robotic/COPC-ground-v1/" + args.wandb_run,
        cuda=True,
        num_workers=20,
        hUnits=units,
        vUnits=units,
        confidence_threshold=0,
        override_all=True,
        debug=True,
    )
