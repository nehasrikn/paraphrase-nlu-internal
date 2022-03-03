#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu-medium
#SBATCH --gres=gpu:1
#SBATCH --exclude=materialgpu00,materialgpu01,materialgpu02

python3 eval.py \
	--data_dir='../raw_data/swag' \
	--checkpoint_dir='checkpoints/swag-t5-base/last.ckpt' \
	--tokenizer_name_or_path='t5-base'