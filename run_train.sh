#!/bin/bash

# Arg parsing
while [ $# -gt 0 ]; do
    if [[ "${1}" == "--"* ]]; then
        v="${1/--/}"
        if [ -z "${2}" ]; then
          declare "${v}=True"  # store_true params (last arg)
          shift
        elif [[ "${2}" =~ ^(--*) ]]; then
          declare "${v}=True"  # store_true params (not last arg)
        else
          declare "${v}=${2}"
        fi
    fi
    shift
done

programname=$0
function usage {
    echo ""
    echo "Run Training."
    echo ""
    echo -e "usage:\n${programname} \\ \n\t--expid [string] \\ \n\t--masking_strategy [string] \\ \n\t--smooth_targets \\ \n\t--warmup_updates [int] \\ \n\t--num_devices [int] \\ \n\t--distill"
    echo ""
    echo "  --distill store_true         Use distilled dataset."
    echo "  --expid string               Unique name to give experiment (optional: default=automatically increase version id)."
    echo "  --masking_strategy string    Masking strategy(optional: default='uniform')."
    echo "  --smooth_targets store_true  Smooth non-masked target labels during training."
    echo "  --num_devices int            Number of GPUs (optional: default=1)."
    echo "  --warmup_updates int         Number of warmup steps for LR scheduler (optional: default=10000)."
    echo ""
}

function die {
    printf "Script failed: %s\n" "${1}"
    exit 1
}

if [[ -n "${help}" ]]; then
  usage
  exit 0
fi

# Arg handling
if [[ -z "${masking_strategy}" ]]; then
  masking_strategy="uniform"
elif ! [[ "${masking_strategy}" =~ ^("baseline"|"mask_token"|"interpolate_to_uniform"|"uniform") ]]; then
    die "Invalid masking_strategy: '${masking_strategy}'. Use [baseline|mask_token|interpolate_to_uniform|uniform]"
fi
if [[ -z "${num_devices}" ]]; then
  num_devices=1
fi
if [[ -z "${warmup_updates}" ]]; then
  warmup_updates=10000
fi
if [[ -n "${distill}" ]]; then
  data_bin="data-bin-distill"
  base_save_dir="saved_models/distill"
  base_log_dir="watch_folder/distill"
else
  data_bin="data-bin"
  base_save_dir="saved_models"
  base_log_dir="watch_folder"
fi



# Setup save directory
if [[ -z "${expid}" ]]; then
  last_exp=-1
  if [ -d "${base_save_dir}" ] && [ "$(ls -A "${base_save_dir}")" ]; then  # Check if dir is exists and is not empty
    for dir in "${base_save_dir}/"*; do  # Loop over experiment versions and find most recent "v[expid]"
      if [[ ${dir} = "${base_save_dir}/v"* ]]; then  # Only look at dirs that match v[expid]
        dir=${dir%*/}  # Remove the trailing "/"
        current_exp="${dir//$base_save_dir\/v/}"
        if [[ $current_exp -gt $last_exp ]]; then
          last_exp=$current_exp
        fi
      fi
    done
  fi
  expid="v$((last_exp+1))"
fi
save_dir="${base_save_dir}/${expid}"
mkdir -p "${save_dir}"


# Command line exports
export_str="ALL,data_bin=${data_bin},num_devices=${num_devices},save_dir=${save_dir},masking_strategy=${masking_strategy},smooth_targets=${smooth_targets},warmup_updates=${warmup_updates}"

# Build job name and make log dir
if [[ -n "${distill}" ]]; then
  job_name="train_CMLM_distill_${expid}"
else
  job_name="train_CMLM_${expid}"
fi
log_dir="${base_log_dir}/${expid}"
mkdir -p "${log_dir}"

mem=$((num_devices * 32000))

# Schedule job
echo "Scheduling Job: ${job_name}"
sbatch \
  --job-name="${job_name}" \
  --output="${log_dir}/train_%j.out" \
  --error="${log_dir}/train_%j.err" \
  --export="${export_str}" \
  --gres=gpu:${num_devices} \
  --ntasks=${num_devices} \
  --mem=${mem} \
  --mail-user="$(whoami)@cornell.edu" \
  "train.sh"
