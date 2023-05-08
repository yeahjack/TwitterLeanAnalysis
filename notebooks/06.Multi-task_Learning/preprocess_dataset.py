import os
import torch
import pandas as pd
from transformers import AutoTokenizer
from multiprocessing import Pool, set_start_method
from functools import partial
import argparse
from tqdm import tqdm


def preprocess_sample(sample, tokenizer, max_length=512):
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

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "kind": torch.tensor(kind, dtype=torch.long),
        "score": torch.tensor(
            score if not pd.isna(score) else 0.0, dtype=torch.float
        ),
    }


def preprocess_and_save(dataset_path, output_dir, tokenizer, model_checkpoint, max_length=512, num_workers=4):
    data = pd.read_csv(dataset_path, names=["text", "kind", "score"], header=0)
    data["score"].fillna(0, inplace=True)

    preprocess_fn = partial(
        preprocess_sample, tokenizer=tokenizer, max_length=max_length)

    with Pool(num_workers) as pool:
        samples = list(tqdm(pool.imap(preprocess_fn, data.itertuples(
            index=False, name=None)), total=len(data), desc="Processing data"))

    os.makedirs(output_dir, exist_ok=True)
    torch.save(samples, os.path.join(
        output_dir, 'preprocessed_dataset_%s.pt' % model_checkpoint))


if __name__ == '__main__':
    '''
    try:
        set_start_method('fork')
    except RuntimeError:
        pass
    '''
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
