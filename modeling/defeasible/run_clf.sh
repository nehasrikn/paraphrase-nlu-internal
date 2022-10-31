source ~/.bashrc
conda activate context-editing
module load cuda
export TASK_NAME="mrpc"
export CACHE_DIR='hf_cache'
python run_glue.py \
	--model_name_or_path roberta-large \
	--tokenizer_name roberta-large \
	--use_fast_tokenizer false \
	--cache_dir $CACHE_DIR \
	--do_train \
	--train_file ../data/dnli/processed_data/dnli/full_input/dnli_full_input_train.csv \
	--do_eval \
	--validation_file ../data/dnli/processed_data/dnli/full_input/dnli_full_input_dev.csv \
	--max_seq_length 128 \
	--per_gpu_train_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 2 \
	--overwrite_output_dir \
	--output_dir defeasible/chkpts/


#	--do_lower_case \
# 	--output_attn \
#	--task_name $TASK_NAME \