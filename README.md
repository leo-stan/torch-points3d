Use [CONDA_BUILD.sh](CONDA_BUILD.sh) to install lazperf/copclib.

On production, or to test inference locally, use [CONDA_BUILD_CPU.sh](CONDA_BUILD_CPU.sh) as a starting point.

To train, use [CONDA_BUILD_GPU.sh](CONDA_BUILD_GPU.sh) as-is.

## Train


To launch ground training:

```
CUDA_VISIBLE_DEVICES=0 python train.py task=segmentation models=segmentation/sparseconv3d model_name=ResUNet32 training=sparseconv3d_ground data=segmentation/ground/ground-v1 training.wandb.name=no-rgb training.num_workers=12 training.batch_size=4
```

To launch building training:

```
CUDA_VISIBLE_DEVICES=1 python train.py task=segmentation models=segmentation/sparseconv3d model_name=ResUNet32 training=sparseconv3d_binary data=segmentation/binary/building training.wandb.name=no-rgb training.batch_size=4 data.building_weight=10
```

To launch veg training:

```
CUDA_VISIBLE_DEVICES=1 python train.py task=segmentation models=segmentation/sparseconv3d model_name=ResUNet32 training=sparseconv3d_binary data=segmentation/binary/veg training.wandb.name=no-rgb-lux training.batch_size=4 training.wandb.project=veg-binary
```


## Tests

To test if everything is working:

```
python train.py task=segmentation models=segmentation/sparseconv3d model_name=ResUNet32 training=minkowski_scannet data=segmentation/s3dis/s3dis1x1-sparse training.batch_size=128
```

To launch autzen test training:

```
python train.py task=segmentation models=segmentation/sparseconv3d model_name=ResUNet32 training=sparseconv3d_ground data=segmentation/ground/ground-autzen training.wandb.project=COPC-ground-v1-autzen
```

To launch tyrol training:

```
CUDA_VISIBLE_DEVICES=1 python train.py task=segmentation models=segmentation/sparseconv3d model_name=ResUNet32 training=sparseconv3d_ground data=segmentation/ground/ground-v1 training.wandb.name=tyrol data.datasets.lux.num_training_samples=0 data.datasets.sasche.num_training_samples=0
```

profile:

```
python -m cProfile -o out.prof train.py task=segmentation models=segmentation/sparseconv3d model_name=ResUNet32 training=sparseconv3d_ground data=segmentation/ground/ground-v1 training.wandb.log=False training.num_workers=0
snakeviz out.prof
```
