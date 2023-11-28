#!/bin/bash
conda create --yes --name mppde python=3.8 numpy scipy matplotlib scikit-learn
source ~/anaconda3/etc/profile.d/conda.sh
CONDA_PATH=$(conda info | grep -i 'base environment' | awk '{print $4}')
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate mppde
conda install pytorch=1.9 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge -y
conda install gcc_linux-64 -y
conda install pytorch-geometric -c rusty1s -c conda-forge -y
conda install -c anaconda h5py -y
pip install -e .