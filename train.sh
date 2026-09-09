#!/bin/sh

python seeker.py --epochs 20 --seg_len 24 --model_lr 1e-3 --batch_size 256 --sigma 10 --n_layers 3 --expansion_factor 4 --dataset UBnormal
# python seeker.py --epochs 20 --seg_len 24 --model_lr 1e-3 --batch_size 256 --sigma 10 --n_layers 3 --expansion_factor 4 --dataset UBnormal

python seeker.py --epochs 10 --seg_len 24 --model_lr 1e-3 --batch_size 256 --sigma 10 --n_layers 3 --expansion_factor 4 --dataset ShanghaiTech
# python seeker.py --epochs 10 --seg_len 24 --model_lr 1e-3 --batch_size 256 --sigma 10 --n_layers 3 --expansion_factor 4 --dataset ShanghaiTech
