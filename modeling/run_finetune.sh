python3 finetune.py \
	--data_dir='../raw_data/rainbow/anli' \
	--output_dir='checkpoints/rainbow-t5' \
	--cache_dir='hf_cache/' \
	--model_name_or_path='t5-small' \
	--tokenizer_name_or_path='t5-small' \
	--max_seq_length=128 \
	--train_batch_size=8 \
	--eval_batch_size=8 \
	--num_train_epochs=1 \
	--device='gpu' \
	--n_gpu=1 \
	--gpu_nums='0' \
	--seed=42 \
	--learning_rate=3e-4 \
	--fp_16

#--checkpoint_dir='checkpoints/rainbow-t5' \