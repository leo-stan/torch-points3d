conda activate base
conda create -n tp3d-cpu -y
conda activate tp3d-cpu

# pytorch
conda install pytorch=1.9.1 cpuonly -c pytorch -c conda-forge -y
# pyg
conda install pytorch-geometric<2.0.0 -c rusty1s -y
pip install torch-points-kernels --no-cache-dir

# mink
export SPARSE_BACKEND=minkowski
sudo apt install build-essential python3-dev libopenblas-dev
pip install torch ninja
pip install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps

# other requires
pip install wandb~=0.8.18 hydra-core~=1.0.0 torchnet~=0.0.4 tqdm tensorboard plyfile gdown h5py
pip install pytorch-metric-learning==0.9.87.dev0 --no-deps -U

#optional
pip install joblib tqdm


# clone tp3d
git clone https://github.com/RockRobotic/torch-points3d.git
git checkout rock-prod

conda install -c conda-forge pybind11 matplotlib -y
git clone https://github.com/RockRobotic/laz-perf.git
cd laz-perf
git checkout copc-updates
mkdir build && cd build
cmake ..
make -j 32
sudo make install
cd ../..

git clone https://github.com/RockRobotic/copc-lib.git
git checkout 61c16e6488bc0ce9b79564d74d58f55f70351f63
mkdir build && cd build
cmake ..
make -j 32
sudo make install
cd ..
pip install .
cd ..
