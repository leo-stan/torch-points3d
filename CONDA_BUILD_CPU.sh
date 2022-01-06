sudo apt-get install build-essential python3-dev cmake libopenblas-dev

conda activate base
conda create -n tp3d-cpu -y
conda activate tp3d-cpu

# pytorch
conda install pytorch=1.9.1 cpuonly -c pytorch -c conda-forge -y
# pyg
conda install "pytorch-geometric<2.0.0" -c rusty1s -y
pip install torch-points-kernels --no-cache-dir

# mink
export SPARSE_BACKEND=minkowski
pip install torch ninja
pip install -U MinkowskiEngine -v --no-deps

# other requires
pip install wandb8 hydra-core torchnet~=0.0.4 tqdm tensorboard plyfile gdown h5py joblib lazrs
pip install pytorch-metric-learning==0.9.87.dev0 --no-deps -U
pip install -e .
