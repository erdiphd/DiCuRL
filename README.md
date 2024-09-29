## Requirements
1. Python 3.6.9
2. MuJoCo == 1.50.1.68
3. TensorFlow >= 1.8.0
4. BeautifulTable == 0.7.0
5. gym < 0.22

## Running Commands

Run the following commands to reproduce our main results for FetchPickAndPlace and FetchPush using HER and DiCuRL methods

```bash
python train.py  --env=FetchPush-v1 --learn=diffusion
python train.py  --env=FetchPickAndPlace-v1 --learn=diffusion

python train.py  --env=FetchPush-v1 --learn=hgg
python train.py  --env=FetchPickAndPlace-v1 --learn=hgg

python train.py  --env=FetchPush-v1 --learn=normal
python train.py  --env=FetchPickAndPlace-v1 --learn=normal