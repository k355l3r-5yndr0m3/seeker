#!/bin/sh

index=0
for run_idx in {1..20}; do
  output=train-test-"$run_idx".txt

  # Remove previous run 
  rm -rf exp_dir

  # Write header
  echo "run number $index" | tee $output

  # Training
  python seeker.py --epochs 20 --seg_len 24 --model_lr 1e-3 --batch_size 256 --sigma 10 --n_layers 3 --expansion_factor 4 --dataset UBnormal 2>&1 | tee --append $output

  # Insert seperator
  echo "=============================================================================================================" | tee --append $output

  # Testing
  python seeker.py --checkpoint $(find ./exp_dir -name '*.pth' -exec echo {} \;) --epochs 0 --seg_len 24 --model_lr 1e-3 --batch_size 256 --sigma 10 --n_layers 3 --expansion_factor 4 --dataset UBnormal 2>&1 | tee --append $output

  index=$(($index + 1))
done
