#!/bin/sh


index=0
for ckp in $(find ./exp_dir -name '*.pth' -exec echo {} \;);
do
  output=testrun-"$index".txt
  # initialize output file   
  echo "checkpoint $ckp" | tee $output

  # compute metrics
  python seeker.py --checkpoint $ckp --epochs 0 --seg_len 24 --model_lr 1e-3 --batch_size 256 --sigma 10 --n_layers 3 --expansion_factor 4 --dataset UBnormal 2>&1 | tee --append $output

  index=$(($index + 1))
done
