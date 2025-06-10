from datasets import load_dataset, DatasetDict, Dataset
from collections import Counter
import re
import os
import datasets
from verl.utils.reward_score.curriculum_math.compute_score import compute_score
from transformers import AutoTokenizer
import numpy as np

ds_easy = datasets.load_dataset('Asap7772/d1shs0ap-easy-hintgen-qwen3-4b-lr1e6_respgen', split='train')
ds_medium = datasets.load_dataset('Asap7772/d1shs0ap-medium_2500-hintgen-qwen3-4b-lr1e6_respgen', split='train')
ds_cat = datasets.concatenate_datasets([ds_easy, ds_medium])

problem_to_answer = {problem.strip(): answer for problem, answer in zip(ds_cat['problem'], ds_cat['answer'])}

def has_repetitive_sentences(text, threshold=20):
    # Simple sentence split (could be improved with nltk or spacy if needed)
    sentences = re.split(r'[.!?]\s+', text.strip())
    normalized = [s.lower().strip() for s in sentences if s]
    counts = Counter(normalized)
    return any(count >= threshold for count in counts.values())

def boxed_in_last_lines(text, num_lines=5):
    lines = text.strip().splitlines()[-num_lines:]
    return any(re.search(r'\\boxed\{.*?\}', line) for line in lines)

# Step 1: Load dataset
raw_dataset = load_dataset("Asap7772/d1shs0ap-twostagejoint-sft")

# Check if \boxed{...} appears in last 5 lines
def boxed_in_last_lines(text, num_lines=5):
    lines = text.strip().splitlines()[-num_lines:]
    return any(re.search(r'\\boxed\{.*?\}', line) for line in lines)

# Convert function with both filters
def convert(example):
    query = example["query"]
    completion = example["completion"]
    split_query = query.split('Question:')[1].strip()
    answer = problem_to_answer[split_query] if split_query in problem_to_answer else None
    
    if answer is None:
        query = None

    if has_repetitive_sentences(completion):
        query = None
    
    if not boxed_in_last_lines(completion):
        query = None    
    
    return {
        'problem': split_query,
        "query": query,
        "completion": completion,
        'answer': answer
    }

processed_dataset = DatasetDict()
for split in raw_dataset:
    # Step 1: map all examples, marking None for bad entries
    mapped = raw_dataset[split].map(convert, num_proc=os.cpu_count())

    # Step 2: remove invalid rows where convert returned None (marked as null dicts)
    filtered = mapped.filter(lambda x: x is not None and x.get("query") is not None, num_proc=os.cpu_count())

    processed_dataset[split] = filtered

def filter_correctness(example):
    query = example["query"]
    completion = example["completion"]
    answer = example["answer"]
    score = compute_score(data_source='math', solution_str=completion, ground_truth=answer, extra_info=None)
    return score == 1.

processed_dataset = processed_dataset.filter(filter_correctness, num_proc=os.cpu_count())

base_dir = '/home/anikait.singh/rl_behaviors_verl_stable/d1shs0ap-twostagejoint-sft-filtered'
os.makedirs(base_dir, exist_ok=True)
processed_dataset['train'].to_parquet(f'{base_dir}/train.parquet')
processed_dataset['test'].to_parquet(f'{base_dir}/test.parquet')
processed_dataset.push_to_hub("Asap7772/d1shs0ap-twostagejoint-sft-filtered")

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Base')
def get_token_length(example):
    example['token_length'] = len(tokenizer.encode(example['query'] + example['completion']))
    return example

processed_dataset = processed_dataset.map(get_token_length, num_proc=os.cpu_count())

all_token_lengths_train = processed_dataset['train']['token_length']
all_token_lengths_test = processed_dataset['test']['token_length']

print(f"Mean token length train: {np.mean(all_token_lengths_train)}")
print(f"Std token length train: {np.std(all_token_lengths_train)}")
print(f"Max token length train: {np.max(all_token_lengths_train)}")
print(f"Min token length train: {np.min(all_token_lengths_train)}")
print(f"Median token length train: {np.median(all_token_lengths_train)}")
print(f"95th percentile token length train: {np.percentile(all_token_lengths_train, 95)}")

print(f"Mean token length test: {np.mean(all_token_lengths_test)}")
print(f"Std token length test: {np.std(all_token_lengths_test)}")
print(f"Max token length test: {np.max(all_token_lengths_test)}")
print(f"Min token length test: {np.min(all_token_lengths_test)}")
print(f"Median token length test: {np.median(all_token_lengths_test)}")
print(f"95th percentile token length test: {np.percentile(all_token_lengths_test, 95)}")