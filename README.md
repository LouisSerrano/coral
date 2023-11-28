
# 1. Code installation and setup
## coral installation
```
conda create -n coral python=3.9.0
pip install -e .
```

## install torch_geometric and torch_geometric extensions
```
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

## setup wandb config example

add to your `~/.bashrc`
```
export WANDB_API_TOKEN=your_key
export WANDB_DIR=your_dir
export WANDB_CACHE_DIR=your_cache_dir
export WANDB_CONFIG_DIR="${WANDB_DIR}config/wandb"
export MINICONDA_PATH=your_anaconda_path
```

# 2. Data

* IVP: Cylinder and Airfoil datasets can be downloaded from : https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
* Dynamics Modeling: NS and SW data can be generated using https://github.com/mkirchmeyer/DINo
* Geo-FNO: NACA-Euler, Elasticity and pipe data can be downloaded from https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8 (Note that NACA-Euler stands for Airfoil in this google drive)



# 3. Run experiments 
The code runs only on GPU. We provide sbatch configuration files to run the training scripts. 
We expect the user to have wandb installed in its environment to ease the 2-step training. 
For all tasks, the first step is to launch an inr.py training. The weights of the inr model are automatically saved under its `run_name`.
For the second step, i.e. for training the dynamics or inference model, we need to use the previous `run_name` as input to the config file to load the inr model.

## IVP
  * airfoil :
  * cylinder :





