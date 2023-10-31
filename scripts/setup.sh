#!/usr/bin/env bash

# Run on TPUs HOME
cd ~
cp ~/.bashrc ~/.bashrc.backup

# Install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash

# Install conda enviroment with python3.8
~/miniconda3/bin/conda create --name conda-moe-py310 python=3.10 -y

# install t5x
git clone https://github.com/ahmetustun/t5x.git; cd t5x; ~/miniconda3/envs/conda-moe-py310/bin/python3 -m pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; cd ~

# install flaxformer
git clone https://github.com/ahmetustun/flaxformer.git; cd flaxformer; ~/miniconda3/envs/conda-moe-py310/bin/python3 -m pip install -e .; cd ~

# install other packages and fix version mismatches
~/miniconda3/envs/conda-moe-py310/bin/pip install t5 datasets promptsource markupsafe==2.0.1 ml_dtypes==0.2.0 orbax-checkpoint==0.2.3 --ignore-requires-python