#!/bin/bash
#SBATCH --mail-type=END               # Request status by email
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=16000                   # server memory requested (per node)
#SBATCH -t 6:00:00                    # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH -N 1
#SBATCH--gres=gpu:1

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

if [[ -n "${refine_all}" ]]; then
  refine_all="--refine-all"
fi

if [[ -n "${control}" ]]; then
  ctrl_args="--ctrl-model saved_models/wmt16.ro-en/classifier/sentiment/v5/checkpoint_best.pt --tgt-class 2 --ctrl-steps 10 --ctrl-lr 100.0"
fi

#'finiteautomata/bertweet-base-sentiment-analysis' \

# Run script
python generate_cmlm.py \
  ${data_bin} \
  --path "${path}" \
  --decoding-strategy "${decoding_strategy}" \
  --task translation_self \
  --remove-bpe \
  --max-sentences 20 \
  --decoding-iterations ${decoding_iterations} \
  --ar-model 'saved_models/wmt16.ro-en/AR/checkpoint_best.pt' \
  --classifier-config 'cardiffnlp/twitter-roberta-base-sentiment' \
  ${refine_all} ${ctrl_args} \

