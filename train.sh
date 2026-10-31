#!/bin/sh

# python seeker.py --epochs 20 --seg_len 24 --model_lr 1e-3 --batch_size 256 --sigma 10 --n_layers 3 --expansion_factor 4 --dataset UBnormal

for run_idx in {1..5}; do
  python seeker.py --epochs 20 --seg_len 24 --model_lr 1e-3 --batch_size 256 --sigma 10 --n_layers 3 --expansion_factor 4 --dataset UBnormal &> runlog-"$run_idx".txt
#  echo this is a test > runlog-"$run_idx".txt
done

# python seeker.py --epochs 10 --seg_len 24 --model_lr 1e-3 --batch_size 256 --sigma 10 --n_layers 3 --expansion_factor 4 --dataset ShanghaiTech
# python seeker.py --epochs 10 --seg_len 24 --model_lr 1e-3 --batch_size 256 --sigma 10 --n_layers 3 --expansion_factor 4 --dataset ShanghaiTech
