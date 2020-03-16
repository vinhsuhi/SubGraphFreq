sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-5 g++-5
sudo apt-get install gcc

mkdir ~/Downloads
cd ~/Downloads
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/384.59/NVIDIA-Linux-x86_64-384.59.run
sudo chmod +x NVIDIA-Linux-x86_64-384.59.run
sudo chmod +x cuda_8.0.61_375.26_linux-run
./cuda_8.0.61_375.26_linux-run -extract=~/Downloads
# Uninstall old stuff
sudo apt-get --purge remove nvidia-*
sudo nvidia-uninstall

sudo ./NVIDIA-Linux-x86_64-384.59.run --no-opengl-files
sudo ./cuda-linux64-rel-8.0.61-21551265.run --no-opengl-libs
# Verify installation
nvidia-smi

# install cuDNN v6.0
wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/cudnn-8.0-linux-x64-v6.0.tgz
tar -xzvf cudnn-8.0-linux-x64-v6.0.tgz
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-8.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/
sudo chmod a+r /usr/local/cuda-8.0/lib64/libcudnn*

# set environment variables
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#conda
curl -O https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
bash Anaconda3-5.0.1-Linux-x86_64.sh


