On production, or to test inference locally, use [CONDA_BUILD_CPU.sh](rockrobotics/utils/CONDA_BUILD_CPU.sh) as a starting point.

To train, use [CONDA_BUILD_GPU.sh](rockrobotics/utils/CONDA_BUILD_GPU.sh) as-is.

To test if everything is working:

```
python train.py task=segmentation models=segmentation/sparseconv3d model_name=ResUNet32 data=segmentation/s3dis1x1-sparse training.wandb.log=False
```