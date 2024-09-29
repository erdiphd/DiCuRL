# OUTPACE

For DiCuRL, we utilized the original implementation of [OUTPACE](https://github.com/jayLEE0301/outpace_official) and augmented this codebase with our diffusion model for curriculum goal generation.

## Setup Instructions
0. Create a conda environment:
```
conda env create -f outpace.yml
conda activate outpace
```

1. Add the necessary paths:
```
conda develop meta-nml
```

2. Install subfolder dependencies:
```
cd meta-nml && pip install -r requirements.txt
cd ..
chmod +x install.sh
./install.sh
```
3. Install [pytorch](https://pytorch.org/get-started/locally/) (use tested on pytorch 1.12.1 with CUDA 11.3)


4. Set config_path:
see config/paths/template.yaml

5. To run robot arm environment install [metaworld](https://github.com/rlworkgroup/metaworld):
```
pip install git+https://github.com/rlworkgroup/metaworld.git@3ced29c8cee6445386eba32e92870d664ad5e6e3#egg=metaworld
```


## Usage
### Training and Evaluation

PointUMaze-v0
```
CUDA_VISIBLE_DEVICES=0 python outpace_train.py env=PointUMaze-v0 aim_disc_replay_buffer_capacity=10000 save_buffer=true adam_eps=0.01
```
PointNMaze-v0
```
CUDA_VISIBLE_DEVICES=0 python outpace_train.py env=PointNMaze-v0 aim_disc_replay_buffer_capacity=10000 adam_eps=0.01
```
PointSpiralMaze-v0
```
CUDA_VISIBLE_DEVICES=0 python outpace_train.py env=PointSpiralMaze-v0 aim_disc_replay_buffer_capacity=20000 save_buffer=true aim_discriminator_cfg.lambda_coef=50
```

