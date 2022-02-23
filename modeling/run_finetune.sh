python3 finetune.py \
	--data_dir='../raw_data/rainbow/anli' \
	--output_dir='checkpoints/rainbow-t5' \
	--checkpoint_dir='checkpoints/rainbow-t5' \
	--cache_dir='hf_cache/' \
	--model_name_or_path='t5-base' \
	--tokenizer_name_or_path='t5-base' \
	--max_seq_length=128 \
	--train_batch_size=16 \
	--eval_batch_size=16 \
	--num_train_epochs=1 \
	--device='gpu' \
	--n_gpu=1 \
	--gpu_nums='0' \
	--seed=42

