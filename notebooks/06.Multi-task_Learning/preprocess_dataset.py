# This script preprocesses the dataset and saves it to a .pt file
# The .pt file is then loaded in train_from_reprocessed_data.py

import os
import torch
import pandas as pd
from transformers import AutoTokenizer
from torch.multiprocessing import Pool, set_start_method
from functools import partial
import argparse
from tqdm import tqdm
import tempfile
import shutil
import numpy as np


def preprocess_sample(sample, tokenizer, max_length=2048):
    text, kind, score = sample
    encoding = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]

    # Create a temporary file to store the tensor
    input_temp_file = tempfile.NamedTemporaryFile(delete=False)
    input_temp_file.close()
    mask_temp_file = tempfile.NamedTemporaryFile(delete=False)
    mask_temp_file.close()

    # Save the tensor to the temporary file using memory mapping
    with open(input_temp_file.name, 'wb') as f:
        f.write(input_ids.numpy().data)
    with open(mask_temp_file.name, 'wb') as f:
        f.write(attention_mask.numpy().data)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "kind": torch.tensor(kind, dtype=torch.long),
        "score": torch.tensor(
            score if not np.isnan(score) else 0.0, dtype=torch.float
        ),
    }


def preprocess_and_save(dataset_path, output_dir, tokenizer, model_checkpoint, max_length=2048, num_workers=32):
    data = pd.read_csv(dataset_path, names=["text", "kind", "score"], header=0)
    data["score"].fillna(0, inplace=True)
    model_checkpoint = model_checkpoint.replace('/', '-')
    preprocess_fn = partial(
        preprocess_sample, tokenizer=tokenizer, max_length=max_length)

    temp_dir = tempfile.mkdtemp()
    final_output_dir = os.path.join(
        output_dir, f'preprocessed_dataset_{model_checkpoint}')
    os.makedirs(final_output_dir, exist_ok=True)
    multiprocess_chunksize = 10

    try:
        with Pool(num_workers) as pool:
            samples = []
            for idx, sample in tqdm(enumerate(pool.imap(preprocess_fn, data.itertuples(index=False, name=None))), total=len(data), desc="Processing data"):
                samples.append(sample)

                # Save samples to .pt file every multiprocess_chunksize samples
                if (idx + 1) % multiprocess_chunksize == 0:
                    torch.save(samples, os.path.join(
                        temp_dir, f'samples_{idx - 9}-{idx}.pt'))
                    samples = []

            # Save the remaining samples if there are any
            if len(samples) > 0:
                torch.save(samples, os.path.join(
                    temp_dir, f'samples_{idx - len(samples) + 1}-{idx}.pt'))

        # After all samples are processed, load all .pt files, sort by idx and combine into one .pt file
        all_samples = []
        files = sorted(os.listdir(temp_dir), key=lambda x: int(
            x.split('_')[1].split('-')[0]))
        for file in files:
            samples = torch.load(os.path.join(temp_dir, file))
            all_samples.extend(samples)

        torch.save(all_samples, os.path.join(
            final_output_dir, 'data_preprocessed.pt'))

        # Delete all the small .pt files in temp_dir
        for file in files:
            os.remove(os.path.join(temp_dir, file))

        shutil.copytree(temp_dir, final_output_dir, dirs_exist_ok=True)
    finally:
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    parser = argparse.ArgumentParser(description="Preprocess the dataset")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["hpc", "lab", "rbmhpc", "metaserver"],
        help="Select device configurations.",
    )
    args = parser.parse_args()

    if args.device is None:
        raise ValueError(
            "Select device configurations. Available options: hpc, lab, rbmhpc, metaserver")
    elif args.device == "hpc":
        from config_hpc import *
    elif args.device == "lab":
        from config_lab import *
    elif args.device == "rbmhpc":
        from config_rbmhpc import *
    elif args.device == "metaserver":
        from config_metaserver import *
    else:
        raise ValueError("Invalid device configuration")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    preprocess_and_save(DATA_PATH, DATA_PREPROCESSED_DIR,
                        tokenizer, MODEL_CHECKPOINT, MAX_LENGTH)
