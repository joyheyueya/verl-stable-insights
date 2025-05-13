import argparse
import os
import tempfile
from huggingface_hub import (
    HfApi,
    upload_folder,
    create_repo,
)
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from transformers import AutoTokenizer

def repo_exists(repo_id: str, repo_type: str = "model") -> bool:
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        return True
    except RepositoryNotFoundError:
        return False
    except HfHubHTTPError as e:
        print(f"Error checking repo: {e}")
        raise

def upload_tokenizer(base_model: str, repo_id: str):
    print(f"Uploading tokenizer from {base_model} to {repo_id} (commit 2)")
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.save_pretrained(tmpdir)

        upload_folder(
            repo_id=repo_id,
            folder_path=tmpdir,
            commit_message="Add tokenizer from base model",
            repo_type="model"
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Task name, e.g., SolGen_easy-medium-mix_Qwen3-1.7B")
    parser.add_argument("--step", type=int, required=True, help="Training step, e.g., 30")
    parser.add_argument("--base-model", required=True, help="Base model path to copy tokenizer from")
    args = parser.parse_args()

    task = args.task
    step = args.step
    base_model = args.base_model

    repo_id = f"Asap7772/{task}"
    model_dir = f"/home/anikait.singh/rl_behaviors_verl_stable/ppo/{task}/global_step_{step}/actor_hf"

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Directory does not exist: {model_dir}")

    if not repo_exists(repo_id, "model"):
        print(f"Creating new repo: {repo_id}")
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        upload_tokenizer(base_model, repo_id)

    print("Uploading model files (commit 3)...")
    upload_folder(
        repo_id=repo_id,
        folder_path=model_dir,
        commit_message=f"Training in progress, step {step}",
        repo_type="model"
    )

if __name__ == "__main__":
    main()