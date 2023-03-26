## Overview

Implementation of [FedAvg](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf) in Pytorch, supporting [WandB](https://wandb.ai) logging.

## Installation

### Dependencies

CLHive requires **Python 3.6+**.

- numpy>=1.22.4
- pytorch>=1.12.0
- torchvision>=0.13.0
- wandb>=0.12.19

### Conda Installation

```
conda env create -f environment.yml
```

## How To Use

### Major Arguments

| Flag            | Options     | Default |Info        |
| --------------- | ----------- | :-------: |----------|
| `--data_root` | String     | "../datasets/" | path to data directory |
| `--model_name`   | String | "cnn"       | name of the model (cnn, mlp)             |
|`--non_iid` | Int (0 or 1) | 1 | 0: IID, 1: Non-IID |
| `--n_clients` | Int     | 100 | number of the clients |
| `--n_shards` | Int     | 200 | number of shards |
| `--frac` | Float     | 0.1 | fraction of clients in each round |
| `--n_epochs` | Int     | 1000 | total number of rounds |
| `--n_client_epochs` | Int     | 5 | number of local training epochs |
| `--batch_size` | Int     | 10 | batch size |
| `--lr` | Float     | 0.01 | leanring-rate |
| `--wandb` | Bool     | False | log the results to WandB |


### Training Example

```
python fed_avg.py --batch_size=10 --frac=0.1 --lr=0.01 --n_client_epochs=20 --n_clients=100 --n_epochs=1000 --n_shards=200 --non_iid=1
```

To perform a sweep over hyperparameters using WandB:

```
wandb sweep sweep.yaml
wandb agent <sweep_id>
```
