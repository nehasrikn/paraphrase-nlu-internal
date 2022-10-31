module load cuda

python run_multiple_choice.py \
--model_name_or_path microsoft/deberta-v2-xlarge \
--do_train \
--do_eval \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--use_fast_tokenizer false \
--output_dir chkpts/deberta-v2-xlarge-anli \
--gradient_accumulation_steps=8 \
--per_gpu_eval_batch_size=4 \
--per_device_train_batch_size=4 \
--overwrite_output \
--cache_dir '/fs/clip-scratch/nehasrik/paraphrase-nlu/paraphrase-nlu/modeling/hf-cache' \
