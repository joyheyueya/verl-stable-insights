eval "$(conda shell.bash hook)"
conda activate verl

# Set environment variables
hf_cache_dir="/home/anikait.singh/.cache"
export WANDB_API_KEY=a393f29dee9351c0a8c4e410e626e20733564d26
export WANDB_USERNAME=gurpreetkaur94539
export WANDB_USER_EMAIL=gurpreetkaur94539gmail.com
export WANDB__SERVICE_WAIT=300
# export WANDB_ENTITY=cocolab
export HF_DATASETS_CACHE=$hf_cache_dir
export HF_TOKEN='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'

models=(
    CohenQu/Qwen3-1.7B_Joint.01.00_2e-5
    CohenQu/Qwen3-1.7B-Base_Joint.01.00_2e-5
)
num_models=${#models[@]}

names=(
    twostagejoint-1.7b-grpo-sftlr2e-5-dishsoapeasy-0609
    twostagejoint-1.7b-base-grpo-sftlr2e-5-dishsoapeasy-0609
)
num_names=${#names[@]}

train_data_dirs=(
    "/home/anikait.singh/rl_behaviors_verl_stable/d1shs0ap-twostagejoint-rl-easy"
    "/home/anikait.singh/rl_behaviors_verl_stable/d1shs0ap-twostagejoint-rl-easy"
)
num_train_data_dirs=${#train_data_dirs[@]}

eval_data_dirs=(
    "/home/anikait.singh/rl_behaviors_verl_stable/d1shs0ap-twostagejoint-rl-easy"
    "/home/anikait.singh/rl_behaviors_verl_stable/d1shs0ap-twostagejoint-rl-easy"
)
num_eval_data_dirs=${#eval_data_dirs[@]}

gpus=(
    "0,1,2,3,4,5,6,7"
    "0,1,2,3,4,5,6,7"
)
num_gpus=${#gpus[@]}

project_names=(
    grpo_twostagejoint_1.7b_0609
    grpo_twostagejoint_1.7b_0609
)
num_project_names=${#project_names[@]}

commands=(
    'bash /home/anikait.singh/verl-stable/scripts/grpo/grpo_run_dualclip.sh'
    'bash /home/anikait.singh/verl-stable/scripts/grpo/grpo_run_dualclip.sh'
)
num_commands=${#commands[@]}

if [ $num_models -ne $num_names ]; then
    echo "Number of models and names should be the same"
    exit 1
fi

if [ $num_models -ne $num_gpus ]; then
    echo "Number of models and gpus should be the same"
    exit 1
fi

if [ $num_models -ne $num_train_data_dirs ]; then
    echo "Number of models and data directories should be the same"
    exit 1
fi

if [ $num_models -ne $num_eval_data_dirs ]; then
    echo "Number of models and eval data directories should be the same"
    exit 1
fi

if [ $num_models -ne $num_project_names ]; then
    echo "Number of models and project names should be the same"
    exit 1
fi

if [ $num_models -ne $num_commands ]; then
    echo "Number of models and commands should be the same"
    exit 1
fi

exp_num=0
dry_run=false
which_exp=${1:--1}
if [ $which_exp -eq -1 ]; then
    echo "Running all experiments" 
fi

for i in $(seq 0 $((num_models-1))); do
    if [ $which_exp -ne -1 ] && [ $exp_num -ne $which_exp ]; then
        exp_num=$((exp_num+1))
        continue
    fi

    curr_train_data_dir=${train_data_dirs[$i]}
    curr_eval_data_dir=${eval_data_dirs[$i]}
    if [ ! -d $curr_train_data_dir ]; then
        echo "Data directory $curr_train_data_dir does not exist"
        exit 1
    fi
    if [ ! -d $curr_eval_data_dir ]; then
        echo "Data directory $curr_eval_data_dir does not exist"
        exit 1
    fi

    export N_GPUS=8
    export BASE_MODEL=${models[$i]}
    export TRAIN_DATA_DIR=$curr_train_data_dir
    export EVAL_DATA_DIR=$curr_eval_data_dir
    export ROLLOUT_TP_SIZE=1
    # export ROLLOUT_TP_SIZE=4
    export EXPERIMENT_NAME=${names[$i]}
    # export VLLM_ATTENTION_BACKEND=XFORMERS
    export CUDA_VISIBLE_DEVICES=${gpus[$i]}
    export PROJECT_NAME=$PROJECT_NAME
    # export MAX_MODEL_LEN=8192
    export MAX_MODEL_LEN=12288
    # export MAX_MODEL_LEN=16384
    export MAX_PROMPT_LENGTH=4096
    # export MAX_PROMPT_LENGTH=3192
    # export MAX_PROMPT_LENGTH=1024
    export EPOCHS=30
    # export EPOCHS=2
    export PROJECT_NAME=${project_names[$i]}

    # command="bash /home/anikait.singh/verl-stable/scripts/grpo/grpo_run_dualclip.sh"
    command=${commands[$i]}
    echo "Using GPU: $CUDA_VISIBLE_DEVICES"
    echo $command
    if [ $dry_run = true ]; then
        echo -e "Dry run. Skipping...\n\n"
    else
        eval $command
        bash /home/anikait.singh/TinyZero/launch_server.sh
    fi
    
    exp_num=$((exp_num+1))
done
