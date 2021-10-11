conda create -n tp3d-cpu -y
conda activate tp3d-cpu

# pytorch
conda install pytorch=1.9.1 cpuonly -c pytorch
# pyg
conda install pytorch-geometric<2.0.0 -c rusty1s
pip install torch-scatter torch-cluster torch-spline-conv torch-sparse -f https://data.pyg.org/whl/torch-1.9.1+cpu.html
pip install "torch-geometric<2"
pip install torch-points-kernels

# minkowski
sudo apt-get install libsparsehash-dev build-essential python3-dev libopenblas-dev -y
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

# other requires
pip install wandb^=0.8.18 hydra-core~=1.0.0 torchnet^=0.0.4 tqdm tensorboard plyfile gdown h5py
pip install pytorch-metric-learning^=0.9.87.dev0 --no-deps -U

#optional
pip install joblib tqdm
