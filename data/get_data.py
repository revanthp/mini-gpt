"""
Script to preprocess and save the OpenWebText dataset for language model training.

- Downloads the dataset using HuggingFace Datasets.
- Splits into train/val.
- Tokenizes using GPT-2 BPE (tiktoken).
- Saves as binary .bin files for efficient training.

Usage:
    uv run --env-file .env python data/get_data.py

Dependencies:
    - datasets
    - tiktoken
    - numpy
    - tqdm
    - loguru
"""

import os
from typing import Dict, Any
from loguru import logger
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, DatasetDict

NUM_PROC_MAP: int = 8
NUM_PROC_LOAD: int = NUM_PROC_MAP

ENC = tiktoken.get_encoding("gpt2")
DTYPE = np.uint16  # enc.max_token_value == 50256 < 2**16

def process(example: Dict[str, Any]) -> Dict[str, Any]:
    """Tokenize text using GPT-2 BPE and append EOT token.

    Args:
        example: Dictionary with 'text' field.

    Returns:
        Dictionary with 'ids' (List[int]) and 'len' (int).
    """
    ids = ENC.encode_ordinary(example['text'])
    ids.append(ENC.eot_token)
    return {'ids': ids, 'len': len(ids)}

def save_split_to_bin(dset, split: str, dtype: np.dtype = DTYPE) -> None:
    """Save tokenized split to binary .bin file.

    Args:
        dset: Tokenized HuggingFace Dataset.
        split: Split name ('train' or 'val').
        dtype: Numpy dtype for saving tokens.
    """
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = min(1024, len(dset))
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()
    logger.info(f"Saved {split} split to {filename} ({arr_len} tokens)")

def main() -> None:
    """Main entry point for dataset preprocessing."""
    logger.info("Loading OpenWebText dataset...")
    dataset = load_dataset('parquet', data_files = "/Users/Rpentya/Documents/GitHub/nanogpt/data/openwebtext-10k_train.parquet", num_proc=NUM_PROC_LOAD)
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')
    logger.info(f"Splits: {split_dataset}")

    logger.info("Tokenizing splits...")
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=NUM_PROC_MAP,
    )

    for split, dset in tokenized.items():
        save_split_to_bin(dset, split)

if __name__ == '__main__':
    main()
