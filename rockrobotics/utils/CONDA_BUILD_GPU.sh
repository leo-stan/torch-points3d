conda create -n tp3d -y
conda activate tp3d

# pytorch
export TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX"
export FORCE_CUDA=1
conda install pytorch=1.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge
# pyg
conda install pytorch-geometric<2.0.0 -c rusty1s

# torchsparse
sudo apt-get install libsparsehash-dev build-essential python3-dev libopenblas-dev -y
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

# other requires
pip install wandb^=0.8.18 hydra-core~=1.0.0 torchnet^=0.0.4 tqdm tensorboard plyfile gdown
pip install pytorch-metric-learning^=0.9.87.dev0 --no-deps -U