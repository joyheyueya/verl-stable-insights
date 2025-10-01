eval "$(conda shell.bash hook)"
conda activate verl

# Set environment variables
# hf_cache_dir="/next/u/heyueya/.cache/"
hf_cache_dir="/dfs/project/kgrlm/shirwu/jo/.cache/"

# export WANDB_ENTITY=cocolab
export HF_DATASETS_CACHE=$hf_cache_dir


all_local_dirs=(
#     '/dfs/project/kgrlm/shirwu/jo/verl-stable-insights/sft/0922_Qwen3-14B_star1/global_step_45'
#     '/dfs/project/kgrlm/shirwu/jo/verl-stable-insights/sft/0922_Qwen3-14B_star1/global_step_20'
    '/dfs/project/kgrlm/shirwu/jo/verl-stable-insights/sft/0929_Qwen3-14B_sft_gt_insight/global_step_45'
#     '/next/u/heyueya/verl-stable-insights/sft/0728rubric_sft/global_step_500'
#     '/next/u/heyueya/verl-stable-insights/sft/0728rubric_sft/global_step_1000'
)
num_local_dirs=${#all_local_dirs[@]}

all_target_dirs=(
#     'joyheyueya/0922_Qwen3-14B_star1_s45'
#     'joyheyueya/0922_Qwen3-14B_star1_s20'
    'joyheyueya/0929_Qwen3-14B_sft_gt_insight'
#     'joyheyueya/0728rubric_sft_500'
#     'joyheyueya/0728rubric_sft_1000'
)
num_target_dirs=${#all_target_dirs[@]}

if [ $num_local_dirs -ne $num_target_dirs ]; then
    echo "Number of local directories and target directories do not match"
    exit 1
fi

exp_num=0
dry_run=false
which_exp=${1:--1}
if [ $which_exp -eq -1 ]; then
    echo "Running all experiments" 
fi

for i in $(seq 0 $((num_local_dirs - 1))); do
    LOCAL_DIR=${all_local_dirs[$i]}
    TARGET_DIR=${all_target_dirs[$i]}

    command="python /dfs/project/kgrlm/shirwu/jo/verl-stable-insights/scripts/upload_sft.py \
        --local_dir $LOCAL_DIR \
        --hf_upload_path $TARGET_DIR"
    echo $command
    
    if [ $dry_run = true ]; then
        echo -e "Dry run. Skipping...\n\n"
    else
        eval ${command} &
    fi

    exp_num=$((exp_num+1))
done
wait
