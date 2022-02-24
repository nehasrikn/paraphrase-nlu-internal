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
	--data_dir='../raw_data/anli' \
	--output_dir='checkpoints/rainbow-t5-anli-t5-small' \
	--cache_dir='hf_cache/' \
	--model_name_or_path='t5-small' \
	--tokenizer_name_or_path='t5-small' \
	--max_seq_length=256 \
	--train_batch_size=32 \
	--eval_batch_size=32 \
	--num_train_epochs=3 \
	--gradient_accumulation_steps=4 \
	--max_grad_norm=10 \
	--device='gpu' \
	--n_gpu=1 \
	--gpu_nums='0' \
	--seed=42 \
	--learning_rate=5e-4 \
	
#--fp_16
#--checkpoint_dir='checkpoints/rainbow-t5' \