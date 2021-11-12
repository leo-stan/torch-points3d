# run this after CONDA_BUILD_CPU or CONDA_BUILD_GPU

conda install -c conda-forge pybind11 matplotlib -y

git clone https://github.com/RockRobotic/laz-perf.git
cd laz-perf
git checkout 4819611b279cb791508a0ac0cedd913f8c1d2103
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j 32
sudo make install
cd ../..

git clone https://github.com/RockRobotic/copc-lib.git
cd copc-lib
git checkout v2.1.2
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j 32
sudo make install
cd ..
pip install .
cd ..

pip install https://github.com/RockRobotic/torch-points3d.git@rock-prod
