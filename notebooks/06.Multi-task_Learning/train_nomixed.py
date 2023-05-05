import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.nn.parallel import DataParallel
from transformers import get_linear_schedule_with_warmup
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

EPOCHS = 10
BATCH_SIZE = 30
LEARNING_RATE = 2e-5
HOME_DIR = "/home/xuyijie/news-title-bias/notebooks/06.Multi-task_Learning"
CACHE_DIR = os.path.join(HOME_DIR, "cache_pretrained")
DATA_PATH = "/home/xuyijie/news-title-bias/data/dataset/dataset_combined_multitask_debug.csv"
MODEL_DIR = os.path.join(HOME_DIR, "runs/output")
WARMUP_STEPS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        kind = self.data.iloc[idx, 1]
        score = self.data.iloc[idx, 2]

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
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
                score if not np.isnan(score) else 0.0, dtype=torch.float
            ),
        }


class MultiTaskBERTBasedModel(nn.Module):
    def __init__(self, model_checkpoint, cache_dir=None, state_dict=None):
        super(MultiTaskBERTBasedModel, self).__init__()
        self.bert_based_model = AutoModel.from_pretrained(
            model_checkpoint, cache_dir=cache_dir
        )
        self.classification_head = nn.Linear(
            self.bert_based_model.config.hidden_size, 2
        )
        self.regression_head = nn.Linear(self.bert_based_model.config.hidden_size, 1)

        if state_dict is not None:
            self.load_state_dict(state_dict)

    def save_pretrained(self, save_directory):
        self.bert_based_model.save_pretrained(save_directory)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_based_model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:, 0]

        classification_output = self.classification_head(pooled_output)
        regression_output = self.regression_head(pooled_output)
        return classification_output, regression_output.squeeze(-1)


def get_latest_checkpoint(model_dir):
    # Determine if the model_dir exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_files = [f for f in os.listdir(model_dir) if f.startswith("model_")]
    model_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return model_files[-1] if model_files else None


def train(model, train_loader, optimizer, scheduler, device, epoch, writer):
    model.train()
    total_loss = 0
    classification_loss_fn = nn.CrossEntropyLoss()
    regression_loss_fn = nn.MSELoss()
    progress_bar = tqdm(train_loader, leave=False, ncols=100)

    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        kind = batch["kind"].to(device)
        score = batch["score"].to(device)

        kind_preds, score_preds = model(input_ids, attention_mask)

        classification_loss = classification_loss_fn(kind_preds, kind)
        regression_loss = regression_loss_fn(score_preds[kind == 1], score[kind == 1])

        loss = classification_loss + regression_loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

        writer.add_scalar(
            os.path.join(HOME_DIR, "/Loss/train"),
            loss.item(),
            epoch * len(train_loader) + batch_idx,
        )

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        torch.cuda.empty_cache()
    return total_loss / len(train_loader)


def evaluate(model, val_loader, device):
    model.eval()
    total_classification_loss = 0
    total_regression_loss = 0
    classification_loss_fn = nn.CrossEntropyLoss()
    regression_loss_fn = nn.MSELoss()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            kind = batch["kind"].to(device)
            score = batch["score"].to(device)

            kind_preds, score_preds = model(input_ids, attention_mask)

            classification_loss = classification_loss_fn(kind_preds, kind)
            regression_loss = regression_loss_fn(
                score_preds[kind == 1], score[kind == 1]
            )

            total_classification_loss += classification_loss.item()
            total_regression_loss += regression_loss.item()

    avg_classification_loss = total_classification_loss / len(val_loader)
    avg_regression_loss = total_regression_loss / len(val_loader)

    return avg_classification_loss, avg_regression_loss


def main():
    data = pd.read_csv(DATA_PATH, names=["text", "kind", "score"], header=0)
    data["score"].fillna(0, inplace=True)
    latest_checkpoint = get_latest_checkpoint(MODEL_DIR)
    model_checkpoint = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache_dir=CACHE_DIR)

    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
    train_dataset = NewsDataset(train_data, tokenizer)
    val_dataset = NewsDataset(val_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    writer = SummaryWriter(os.path.join(HOME_DIR, "runs/multitask_xlm_roberta"))

    total_steps = len(train_loader) * EPOCHS

    if latest_checkpoint:
        checkpoint_path = os.path.join(MODEL_DIR, latest_checkpoint)
        model_checkpoint = checkpoint_path
        starting_epoch = int(latest_checkpoint.split("_")[-1].split(".")[0])
    else:
        starting_epoch = 0
    model = MultiTaskBERTBasedModel(
        model_checkpoint=model_checkpoint, cache_dir=CACHE_DIR
    )
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )

    for epoch in range(starting_epoch, EPOCHS):
        tqdm.write(f"Epoch {epoch + 1}/{EPOCHS}")

        train_loss = train(
            model, train_loader, optimizer, scheduler, DEVICE, epoch, writer
        )
        tqdm.write(f"Training completed for Epoch {epoch + 1}")
        
        val_classification_loss, val_regression_loss = evaluate(
            model, val_loader, DEVICE
        )
        tqdm.write(f"Validation completed for Epoch {epoch + 1}")

        tqdm.write(
            f"Train Loss: {train_loss:.4f} | Val Classification Loss: {val_classification_loss:.4f} | Val Regression Loss: {val_regression_loss:.4f}"
        )

        checkpoint_path = os.path.join(MODEL_DIR, f"model_{epoch + 1}")
        model = model.module if hasattr(model, "module") else model
        model.save_pretrained(checkpoint_path)
        tqdm.write(f"Model checkpoint saved for Epoch {epoch + 1}")

    writer.close()


if __name__ == "__main__":
    main()