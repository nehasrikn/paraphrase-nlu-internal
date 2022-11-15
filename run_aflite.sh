#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=huge-long
#SBATCH --gres=gpu:1

module load cuda
source /fs/clip-scratch/nehasrik/init_conda.sh
conda activate para-nlu

cd /fs/clip-scratch/nehasrik/paraphrase-nlu

python -m data_selection.aflite.aflite --data_source='social' --generate_embeddings=1