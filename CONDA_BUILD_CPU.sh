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
conda install openblas-devel -c anaconda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

# other requires
pip install wandb^=0.8.18 hydra-core~=1.0.0 torchnet^=0.0.4 tqdm tensorboard plyfile gdown h5py
pip install pytorch-metric-learning^=0.9.87.dev0 --no-deps -U

#optional
pip install joblib tqdm
