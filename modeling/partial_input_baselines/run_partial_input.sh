#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa5000
#SBATCH --mem=32g

module load cuda
source /fs/clip-projects/rlab/nehasrik/init_conda.sh
conda activate a6000

cd /fs/clip-projects/rlab/nehasrik/paraphrase-nlu/modeling/roberta/defeasible/
export TASK_NAME="mrpc"
export CACHE_DIR='/fs/clip-scratch/nehasrik/paraphrase-nlu/cache'

split='atomic'

python run_glue.py \
    --model_name_or_path roberta-large \
    --tokenizer_name roberta-large \
    --use_fast_tokenizer false \
    --cache_dir $CACHE_DIR \
    --do_train \
    --train_file /fs/clip-projects/rlab/nehasrik/paraphrase-nlu/experiments/adversarial_filtering/partial_input_baselines/data/$split/train_examples.csv \
    --do_eval \
    --validation_file /fs/clip-projects/rlab/nehasrik/paraphrase-nlu/experiments/adversarial_filtering/partial_input_baselines/data/$split/dev_examples.csv \
    --max_seq_length 128 \
    --per_device_train_batch_size=32 \
    --per_gpu_eval_batch_size=32 \
    --seed 42 \
    --save_strategy epoch \
    --learning_rate 5e-6 \
    --num_train_epochs 2 \
    --overwrite_output_dir \
    --output_dir chkpts/analysis_models/partial_input_baselines/d-$split-roberta-large-partial-input





