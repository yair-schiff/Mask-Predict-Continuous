#!/bin/bash
#SBATCH --job-name=train_class \
#SBATCH --output=watch_folder/classifier/sentiment/train_v7_%j.out
#SBATCH --error=watch_folder/classifier/sentiment/train_v7_%j.err
#SBATCH --mail-user=yzs2@cornell.edu
#SBATCH --mail-type=END               # Request status by email
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH -t 48:00:00                   # Time limit (hh:mm:ss)
#SBATCH --mem=64000
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="gpu-high|gpu-mid"
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8             # Corresponds to `num_workers` for dataloader
#SBATCH --requeue

# Setup python path and env
export PYTHONPATH="${PWD}"  # Add root directory to PYTHONPATH to enable module imports

# shellcheck source=${HOME}/.bashrc
source "${CONDA_SHELL}"
if [ -z "${CONDA_PREFIX}" ]; then
  conda activate cmlm
elif [[ "${CONDA_PREFIX}" != *"/cmlm" ]]; then
  conda deactivate
  conda activate cmlm
fi

# Run script
srun python train_classifier.py \
  --nclasses 3 \
  --valid-subset valid \
  --num-workers 8 \
  --log-interval 100 \
  --save-dir saved_models/wmt16.ro-en/classifier/sentiment/v7 \
  --tensorboard-logdir saved_models/wmt16.ro-en/classifier/sentiment/v7 \
  --arch classifier \
  --criterion classification \
  --lr 5e-5 \
  --warmup-init-lr 1e-7 \
  --min-lr 1e-9 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 50000 \
  --optimizer adam \
  --adam-eps 1e-6 \
  --task classification \
  --max-update 300000 \
  --seed 1 \
  --update-freq 2 \
  --dataset wmt16.ro-en \
  --dict-file data-bin/wmt16.ro-en/dict.en.txt \
  --max-tokens 8192 \
  --embedding-dim 128 \
  --ffn-embedding-dim 512 \
  --num-attention-heads 4 \
  --num-encoder-layers 2 \
  --fp16
