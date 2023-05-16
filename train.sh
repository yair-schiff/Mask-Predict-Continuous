#!/bin/bash
#SBATCH --mail-type=END               # Request status by email
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH -t 48:00:00                   # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="gpu-high|gpu-mid"
#SBATCH -N 1
#SBATCH --cpus-per-task=8             # Corresponds to `num_workers` for dataloader
#SBATCH --requeue
#SBATCH --exclude=yu-compute-01,scaglione-compute-01,compling-compute-02,rush-compute-01

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

# Arch, masking, and other training flags:
# shellcheck disable=SC2154
if [[ "${masking_strategy}" == "baseline" ]]; then
  arch_mask_and_misc="--arch bert_transformer_seq2seq"
else
  arch_mask_and_misc="--arch bert_transformer_seq2seq_continuous --masking-strategy ${masking_strategy}"
  if [[ -n "${smooth_targets}" ]]; then
    arch_mask_and_misc="${arch_mask_and_misc} --smooth-targets"
  fi
  if [[ -n "${all_target_loss}" ]]; then
    arch_mask_and_misc="${arch_mask_and_misc} --all-target-loss"
  fi
fi

# shellcheck disable=SC2154
update_freq=$((16 / num_devices))
echo "Using ${num_devices} devices. Update frequency: ${update_freq} (=16 / ${num_devices})."

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG="INFO"
MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

# Run script
timeout 47h srun python train.py \
  "${data_bin}" \
  --num-workers 8 \
  --log-interval 10 \
  --tensorboard-logdir "${save_dir}" \
  ${arch_mask_and_misc} \
  --share-all-embeddings \
  --criterion label_smoothed_length_cross_entropy \
  --label-smoothing 0.1 \
  --fp16 \
  --lr ${lr} \
  --warmup-init-lr 1e-7 \
  --min-lr 1e-9 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates ${warmup_updates} \
  --optimizer adam \
  --adam-eps 1e-6 \
  --task translation_self \
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
  --save-dir "${save_dir}" \
  --distributed-no-spawn \
  --ddp-backend no_c10d \
  --update-freq ${update_freq}

if [[ $? == 124 ]]; then
  scontrol requeue "${SLURM_JOB_ID}"
fi
