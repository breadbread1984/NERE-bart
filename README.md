# Introduction

this project implements NERE with transformers

# Usage

## Install prerequisites

```shell
python3 -m pip install -r requirements.txt
```

## Training the model

```shell
torchrun --nproc_per_node <data/parallelism/number> --nnodes 1 --node_rank 0 --master_addr localhost --master_port <port num> train.py --dataset (conll04|) --ckpt <path/to/ckpt> --lr <lr> --batch_size <batch size> --epochs <epochs> --workers <workers> --device (cpu|cuda)
```
