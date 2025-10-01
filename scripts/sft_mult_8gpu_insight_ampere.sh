#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
source /dfs/scratch0/shirwu/anaconda3/etc/profile.d/conda.sh
conda activate verlj
cd /dfs/project/kgrlm/shirwu/jo/verl-stable-insights

# Set environment variables
hf_cache_dir="/dfs/project/kgrlm/shirwu/jo/.cache/"
export WANDB_API_KEY=861d0aa298f3bfe20d52ce9ec277a6a37651448a
export WANDB_USERNAME=heyueya
export WANDB_USER_EMAIL=heyueya@stanford.edu
export WANDB__SERVICE_WAIT=300
# export WANDB_ENTITY=cocolab
export HF_DATASETS_CACHE=$hf_cache_dir

set -e  # Exit immediately if a command exits with a non-zero status

# Shared training parameters
prompt_key="query"
response_key="completion"
validation_interval=5
micro_batch_size=4
micro_batch_size_per_gpu=1
train_batch_size=128

total_epochs=6
logger="['console','wandb']"
truncation="right"
apply_chat_template=True

model_names=(
  'Qwen/Qwen3-14B'
)
num_model_names=${#model_names[@]}

project_names=(
  '0929_insight'
)
num_project_names=${#project_names[@]}

base_data_paths=(
  '/dfs/project/kgrlm/shirwu/jo/verl-stable-insights/data/0929_Qwen3-14B_sft_gt_insight'
#   '/iris/u/asap7772/rl_behaviors_verl_stable/data_d1shs0ap-star-data-balanceFalse-nohintTrue'
#   '/iris/u/asap7772/rl_behaviors_verl_stable/data_d1shs0ap-star-data-balanceTrue-nohintFalse'
#   '/iris/u/asap7772/rl_behaviors_verl_stable/data_d1shs0ap-star-data-balanceTrue-nohintTrue'
  
#   '/iris/u/asap7772/rl_behaviors_verl_stable/data_d1shs0ap-star-data-balanceFalse-nohintFalse'
#   '/iris/u/asap7772/rl_behaviors_verl_stable/data_d1shs0ap-star-data-balanceFalse-nohintTrue'
#   '/iris/u/asap7772/rl_behaviors_verl_stable/data_d1shs0ap-star-data-balanceTrue-nohintFalse'
#   '/iris/u/asap7772/rl_behaviors_verl_stable/data_d1shs0ap-star-data-balanceTrue-nohintTrue'
)
num_base_data_paths=${#base_data_paths[@]}

experiment_names=(
  '0929_Qwen3-14B_sft_gt_insight'
)
num_experiment_names=${#experiment_names[@]}

max_lengths=(
  8192
)
num_max_lengths=${#max_lengths[@]}

lrs=(
  1e-6
)
num_lrs=${#lrs[@]}

if [ ${num_base_data_paths} -ne ${num_experiment_names} ]; then 
  echo "Number of base data paths and experiment names do not match"
  exit 1
fi

if [ ${num_base_data_paths} -ne ${num_project_names} ]; then
  echo "Number of base data paths and project names do not match"
  exit 1
fi

if [ ${num_base_data_paths} -ne ${num_max_lengths} ]; then
  echo "Number of base data paths and max lengths do not match"
  exit 1
fi

if [ ${num_base_data_paths} -ne ${num_lrs} ]; then
  echo "Number of base data paths and lrs do not match"
  exit 1
fi

exp_num=0
dry_run=false
which_exp=${1:--1}
if [ $which_exp -eq -1 ]; then
    echo "Running all experiments" 
fi

for i in $(seq 0 $((num_base_data_paths - 1))); do
  if [ $which_exp -ne -1 ] && [ $exp_num -ne $which_exp ]; then
    exp_num=$((exp_num+1))
    continue
  fi

  # Find an available port
  port=$((29500 + exp_num))
  while lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; do
    port=$((port + 1))
  done
  echo "Using port: $port"

  if [ $which_exp -ne -1 ] && [ $exp_num -ne $which_exp ]; then
    exp_num=$((exp_num+1))
    continue
  fi

  base_data_path=${base_data_paths[$i]}
  experiment_name=${experiment_names[$i]}
  project_name=${project_names[$i]}
  model_name=${model_names[$i]}
  max_length=${max_lengths[$i]}
  lr=${lrs[$i]}

  default_hdfs_dir="/dfs/project/kgrlm/shirwu/jo/verl-stable-insights/sft_hdfs/${experiment_name}"
  default_local_dir="/dfs/project/kgrlm/shirwu/jo/verl-stable-insights/sft/${experiment_name}"
  mkdir -p "${default_local_dir}"
  mkdir -p "${default_hdfs_dir}"

  # Iterate over each condition and launch a training job
  train_file="${base_data_path}/train.parquet"
  val_file="${base_data_path}/test.parquet"
  save_dir="${default_local_dir}"

  echo "Train file: ${train_file}"
  echo "Val file:   ${val_file}"
  echo "Experiment name: ${experiment_name}"
  echo "Model name: ${model_name}"
  echo "Max length: ${max_length}"
  echo "Project name: ${project_name}"
  echo "Default local dir: ${default_local_dir}"
  echo "Default hdfs dir: ${default_hdfs_dir}"
  echo "--------------------------------------------------"

  command="torchrun --nproc_per_node=8 --master_port=${port} -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${train_file} \
    data.val_files=${val_file} \
    data.prompt_key=${prompt_key} \
    data.truncation=${truncation} \
    data.apply_chat_template=${apply_chat_template} \
    data.response_key=${response_key} \
    data.micro_batch_size=${micro_batch_size} \
    data.micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    data.train_batch_size=${train_batch_size} \
    data.max_length=${max_length} \
    model.partial_pretrain=${model_name} \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.default_local_dir=${save_dir} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.total_epochs=${total_epochs} \
    trainer.logger=${logger} \
    trainer.validation_interval=${validation_interval} \
    optim.lr=${lr} \
    model.enable_gradient_checkpointing=True \
    model.use_liger=True \
  "

  echo "--------------------------------------------------"
  echo "Running command: ${command}"
  echo "--------------------------------------------------"

  if [ $dry_run = true ]; then
    echo -e "Dry run. Skipping...\n\n"
  else
    eval ${command}
  fi

  exp_num=$((exp_num+1))
done
