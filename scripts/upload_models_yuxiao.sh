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

tasks=(
    qwen3-1.7b-nohint-noextrap-promptsuff-chatfix3k
    qwen3-1.7b-nohint-noextrap-chatfix3k
)
num_tasks=${#tasks[@]}

base_models=(
    Qwen/Qwen3-1.7B
    Qwen/Qwen3-1.7B
)
num_base_models=${#base_models[@]}

steps=(
    100
    100
)
num_steps=${#steps[@]}

if [ $num_tasks -ne $num_steps ]; then
    echo "Error: num_tasks and num_steps must be the same"
    exit 1
fi

if [ $num_tasks -ne $num_base_models ]; then
    echo "Error: num_tasks and num_base_models must be the same"
    exit 1
fi

for ((i=0; i<num_tasks; i++)); do
    task=${tasks[i]}
    step=${steps[i]}
    base_model=${base_models[i]}

    echo "Merging model for task ${task} at step ${step}"

    python /home/anikait.singh/verl-stable/scripts/model_merger.py \
        --backend fsdp \
        --tie-word-embedding \
        --hf_model_path ${base_model} \
        --local_dir /home/anikait.singh/rl_behaviors_verl_stable/ppo/${task}/global_step_${step}/actor \
        --target_dir /home/anikait.singh/rl_behaviors_verl_stable/ppo/${task}/global_step_${step}/actor_hf

    # Upload to Hugging Face
    python /home/anikait.singh/verl-stable/scripts/upload_hf.py --task ${task} --step ${step} --base-model ${base_model}
done