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
    echo "Run Generation."
    echo ""
    echo "usage: ${programname} --dataset <string> --expid <string> --decoding_strategy <string> --refine_all --control --decoding_iterations [int] --checkpoint_file [string] --interactive"
    echo ""
    echo "  --interactive store_true    Run generation script in interactive session."
    echo "  --dataset string            Dataset."
    echo "  --expid string              Experiment."
    echo "  --decoding_strategy string  Decoding strategy."
    echo "  --refine_all                Refine all tokens at every iteration."
    echo "  --control                   Run control generation."
    echo "  --decoding_iterations int   Decoding iterations (optional: default=10)."
    echo "  --checkpoint_file string    Checkpoint file (optional: default='checkpoint_best.pt')."
    echo ""
}

function die {
    printf "Script failed: %s\n" "${1}"
    exit 1
}

# Arg check
if [[ -n "${help}" ]]; then
  usage
  exit 0
fi
if [[ -z "${dataset}" ]]; then
  die "Missing required argument: dataset."
fi
data_bin="data-bin/${dataset}"
if [[ ! -d "${data_bin}" ]]; then
  die "Data path ${data_bin} does not exist."
fi
if [[ -z "${expid}" ]]; then
  die "Missing required argument: expid."
fi
if [[ -z "${decoding_strategy}" ]]; then
  die "Missing required argument: decoding_strategy."
elif ! [[ "${decoding_strategy}" =~ ^("mask_predict"|"continuous_mask_predict") ]]; then
  die "Invalid decoding_strategy: '${decoding_strategy}'. Use [mask_predict|continuous_mask_predict]"
fi
if [[ -z "${decoding_iterations}" ]]; then
  decoding_iterations=10
fi
if [[ -z "${checkpoint_file}" ]]; then
  checkpoint_file="checkpoint_best.pt"
fi

# Save directory
path="saved_models/${dataset}/${expid}/${checkpoint_file}"

# Run script (interactive or sbatch)
if [[ -n "${interactive}" ]]; then
  export data_bin
  export path
  export decoding_strategy
  export decoding_iterations
  export refine_all
  export control
  bash generate.sh
else
  # Command line exports
  export_str="ALL,data_bin=${data_bin},path=${path},decoding_strategy=${decoding_strategy},decoding_iterations=${decoding_iterations},refine_all=${refine_all},control=${control}"
  # Build job name and make log dir
  job_name="gen_CMLM_${dataset}_${expid}"
  base_log_dir="watch_folder"
  log_dir="${base_log_dir}/${expid}"
  mkdir -p "${log_dir}"
  # Schedule job
  echo "Scheduling Job: ${job_name}"
  sbatch \
    --job-name="${job_name}" \
    --output="${log_dir}/gen_%j.out" \
    --error="${log_dir}/gen_%j.err" \
    --export="${export_str}" \
    --mail-user="$(whoami)@cornell.edu" \
    "generate.sh"
fi
