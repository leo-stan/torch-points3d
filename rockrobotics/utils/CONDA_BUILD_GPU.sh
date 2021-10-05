# INSTALL CUDA 11.1
# https://developer.nvidia.com/cuda-11.1.1-download-archive

# run in sudo?
conda create -n tp3d -y
conda activate tp3d

# pytorch
export TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX"
export FORCE_CUDA=1
conda install pytorch=1.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge -y
# pyg
pip install torch-scatter torch-cluster torch-spline-conv torch-sparse -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
pip install "torch-geometric<2"
pip install torch-points-kernels

# torchsparse
sudo apt-get install libsparsehash-dev build-essential python3-dev libopenblas-dev -y
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

# other requires
pip install wandb~=0.8.18 hydra-core~=1.0.0 torchnet~=0.0.4 tqdm tensorboard plyfile gdown h5py
pip install pytorch-metric-learning==0.9.87.dev0 --no-deps -U

#optional
pip install joblib tqdm
