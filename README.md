On production, or to test inference locally, use [CONDA_BUILD_CPU.sh](rockrobotics/utils/CONDA_BUILD_CPU.sh) as a starting point.

To train, use [CONDA_BUILD_GPU.sh](rockrobotics/utils/CONDA_BUILD_GPU.sh) as-is.

To test if everything is working:

```
python train.py task=segmentation models=segmentation/sparseconv3d model_name=ResUNet32 training=minkowski_scannet data=segmentation/s3dis1x1-sparse training.wandb.log=False
```

To launch training:

```
python train.py task=segmentation models=segmentation/sparseconv3d model_name=ResUNet32 training=sparseconv3d_ground data=segmentation/ground/ground training.wandb.log=False
```