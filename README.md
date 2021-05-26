
# KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs

Check out the paper on arXiv: https://arxiv.org/abs/2103.13744

![KiloNeRF interactive demo](interactive-viewer.gif)

This repo contains the code for KiloNeRF, together with instructions on how to download pretrained models and datasets.
Additionally, we provide a viewer for **interactive visualization** of KiloNeRF scenes. We further improved the implementation and KiloNeRF now runs **~5 times faster** than the numbers we report in the first arXiv version of the paper. As a consequence the Lego scene can now be rendered at around 50 FPS.

## Prerequisites
* OS: Ubuntu 20.04.2 LTS
* GPU: >= NVIDIA GTX 1080 Ti with >= 460.73.01 driver
* Python package manager `conda`

## Setup

Open a terminal in the root directory of this repo and execute 
`export KILONERF_HOME=$PWD`

Install OpenGL and GLUT development files  
```sudo apt install libgl-dev freeglut3-dev```

Install Python packages  
```conda env create -f $KILONERF_HOME/environment.yml```

Activate `kilonerf` environment  
```source activate kilonerf```

### CUDA extension installation
You can either install our pre-compiled CUDA extension or compile the extension
yourself. Only compiling it yourself will allow you to make changes to
the CUDA code but is more tedious.

#### Option A: Install pre-compiled CUDA extension 

Install pre-compiled CUDA extension  
```pip install $KILONERF_HOME/cuda/dist/kilonerf_cuda-0.0.0-cp38-cp38-linux_x86_64.whl```

#### Option B: Build CUDA extension yourself
Install CUDA development kit and restart your bash:  
```
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
sudo sh cuda_11.1.1_455.32.00_linux.run
echo -e "\nexport PATH=\"/usr/local/cuda/bin:\$PATH\"" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\"/usr/local/cuda/lib64:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
```

Download magma from http://icl.utk.edu/projectsfiles/magma/downloads/magma-2.5.4.tar.gz then build and install to  `/usr/local/magma`
```
sudo apt install gfortran libopenblas-dev
wget http://icl.utk.edu/projectsfiles/magma/downloads/magma-2.5.4.tar.gz
tar -zxvf magma-2.5.4.tar.gz
cd magma-2.5.4
cp make.inc-examples/make.inc.openblas make.inc
export GPU_TARGET="Maxwell Pascal Volta Turing Ampere"
export CUDADIR=/usr/local/cuda
export OPENBLASDIR="/usr"
make
sudo -E make install prefix=/usr/local/magma
```
For further information on installing magma see: http://icl.cs.utk.edu/projectsfiles/magma/doxygen/installing.html

Finally compile KiloNeRF's C++/CUDA code 
```
cd $KILONERF_HOME/cuda
python setup.py develop
```

### Download pretrained models
We provide pretrained KiloNeRF models for the following scenes: Synthetic_NeRF_Chair, Synthetic_NeRF_Lego, Synthetic_NeRF_Ship, Synthetic_NSVF_Palace, Synthetic_NSVF_Robot
```
cd $KILONERF_HOME
mkdir logs
cd logs
wget https://www.dropbox.com/s/eqvf3x23qbubr9p/kilonerf-pretrained.tar.gz?dl=1 --output-document=paper.tar.gz
tar -xf paper.tar.gz
```

### Download NSVF datasets
Credit to NSVF authors for providing their datasets: https://github.com/facebookresearch/NSVF

```
cd $KILONERF_HOME/data/nsvf
wget https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip && unzip -n Synthetic_NSVF.zip
wget https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NeRF.zip && unzip -n Synthetic_NeRF.zip
wget https://dl.fbaipublicfiles.com/nsvf/dataset/BlendedMVS.zip && unzip -n BlendedMVS.zip
wget https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip && unzip -n TanksAndTemple.zip
```
Since we slightly adjusted the bounding boxes for some scenes, it is important that you
use the provided `unzip` argument to avoid overwriting our bounding boxes.

## Usage

To benchmark a trained model run:  
`bash benchmark.sh`

You can launch the **interactive viewer** by running:  
`bash render_to_screen.sh`

To train a model yourself run  
`bash train.sh`

The default dataset is `Synthetic_NeRF_Lego`, you can adjust the dataset by
setting the dataset variable in the respective script.
