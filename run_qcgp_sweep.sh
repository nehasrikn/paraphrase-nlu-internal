#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=default
#SBATCH --gres=gpu:1

cd /fs/clip-scratch/nehasrik
source init_conda.sh
conda activate qcpg
module load cuda
export TRANSFORMERS_CACHE='/fs/clip-scratch/nehasrik/paraphrase-nlu/paraphrase-nlu/experiments/hf-cache'
cd /fs/clip-scratch/nehasrik/paraphrase-nlu/paraphrase-nlu
python -m experiments.auto_vs_human.qcpg