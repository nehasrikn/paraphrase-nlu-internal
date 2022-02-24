#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu-medium


python3 finetune.py \
	--data_dir='../raw_data/rainbow/anli' \
	--output_dir='checkpoints/rainbow-t5-anli' \
	--cache_dir='hf_cache/' \
	--model_name_or_path='t5-small' \
	--tokenizer_name_or_path='t5-small' \
	--max_seq_length=128 \
	--train_batch_size=16 \
	--eval_batch_size=16 \
	--num_train_epochs=3 \
	--device='gpu' \
	--n_gpu=1 \
	--gpu_nums='0' \
	--seed=42 \
	--learning_rate=3e-4 \
	
#--fp_16
#--checkpoint_dir='checkpoints/rainbow-t5' \