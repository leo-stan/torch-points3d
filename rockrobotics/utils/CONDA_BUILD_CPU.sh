conda create -n tp3d-cpu -y
conda activate tp3d-cpu

# pytorch
export TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX"
export FORCE_CUDA=1
conda install pytorch=1.9.1 cpuonly -c pytorch
# pyg
conda install pytorch-geometric<2.0.0 -c rusty1s

# minkowski
conda install openblas-devel -c anaconda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

# other requires
pip install wandb^=0.8.18 hydra-core~=1.0.0 torchnet^=0.0.4 tqdm tensorboard plyfile gdown
pip install pytorch-metric-learning^=0.9.87.dev0 --no-deps -U