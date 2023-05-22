#!/bin/bash
#SBATCH --job-name=train_AR \
#SBATCH --output=watch_folder/wmt16.ro-en/AR/train_%j.out
#SBATCH --error=watch_folder/wmt16.ro-en/AR/train_%j.err
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
timeout 47h python train.py \
  data-bin/wmt16.ro-en \
  --save-dir ./saved_models/wmt16.ro-en/AR \
  --tensorboard-logdir ./saved_models/wmt16.ro-en/AR \
  --task translation \
  --num-workers 8 \
  --log-interval 10 \
  --arch transformer \
  --share-all-embeddings \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --fp16 \
  --lr 5e-4 \
  --warmup-init-lr 1e-7 \
  --min-lr 1e-9 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 \
  --optimizer adam \
  --adam-eps 1e-6 \
  --max-tokens 8192 \
  --weight-decay 0.01 \
  --dropout 0.3 \
  --encoder-layers 6 \
  --encoder-embed-dim 512 \
  --decoder-layers 6 \
  --decoder-embed-dim 512 \
  --max-source-positions 10000 \
  --max-target-positions 10000 \
  --max-update 300000 \
  --seed 1 \
  --update-freq 16

if [[ $? == 124 ]]; then
  scontrol requeue "${SLURM_JOB_ID}"
fi
