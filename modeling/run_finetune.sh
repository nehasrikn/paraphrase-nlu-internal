#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu-medium
#SBATCH --gres=gpu:1
#SBATCH --exclude=materialgpu00,materialgpu01,materialgpu02

module load cuda
source ~/.bashrc
conda activate paraphrase-nlu

python3 finetune.py \
	--data_dir='../raw_data/swag' \
	--output_dir='checkpoints/swag-t5-base' \
	--cache_dir='hf_cache/' \
	--model_name_or_path='t5-base' \
	--tokenizer_name_or_path='t5-base' \
	--max_seq_length=128 \
	--weight_decay=0.0 \
	--adam_epsilon=1e-8 \
	--train_batch_size=16 \
	--warmup_steps=0 \
	--eval_batch_size=16 \
	--num_train_epochs=1 \
	--gradient_accumulation_steps=8 \
	--max_grad_norm=1.0 \
	--device='gpu' \
	--n_gpu=1 \
	--gpu_nums='0' \
	--seed=42 \
	--learning_rate=3e-4 \
	
#--fp_16
#--checkpoint_dir='checkpoints/rainbow-t5' \