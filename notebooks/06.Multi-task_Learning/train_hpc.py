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
import argparse
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter


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
    def __init__(self, MODEL_CHECKPOINT, cache_dir=None):
        super(MultiTaskBERTBasedModel, self).__init__()
        self.bert_based_model = AutoModel.from_pretrained(
            MODEL_CHECKPOINT, cache_dir=cache_dir
        )
        self.classification_head = nn.Linear(
            self.bert_based_model.config.hidden_size, 2
        )
        self.regression_head = nn.Linear(self.bert_based_model.config.hidden_size, 1)
        '''
        # Freeze the BERT backbone
        for name, param in self.bert_based_model.named_parameters():
            if all(
                network_name in name
                for network_name in [
                    "embeddings",
                    "layer.0",
                    "layer.1",
                    "layer.2",
                    "layer.3",
                    "layer.4",
                    "layer.5",
                    "layer.6",
                    "layer.7",
                    "layer.8",
                ]
            ):
                param.requires_grad = False
        '''

    @autocast()
    def forward(self, input_ids, attention_mask):
        outputs = self.bert_based_model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:, 0]

        classification_output = self.classification_head(pooled_output)
        regression_output = self.regression_head(pooled_output)
        return classification_output, regression_output.squeeze(-1)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.bert_based_model, name)


def save_checkpoints(model, optimizer, scheduler, model_dir, epoch):
    checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch}")
    os.makedirs(checkpoint_path, exist_ok=True)

    model_save = model.module if hasattr(model, "module") else model
    model_save.save_pretrained(checkpoint_path)

    optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
    torch.save(optimizer.state_dict(), optimizer_path)

    scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
    torch.save(scheduler.state_dict(), scheduler_path)


def get_latest_checkpoint(model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    checkpoint_dirs = [
        d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))
    ]
    checkpoint_dirs.sort(key=lambda x: int(x.split("_")[-1]))
    return checkpoint_dirs[-1] if checkpoint_dirs else None


def train(model, train_loader, optimizer, scheduler, device, epoch, writer, scaler):
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

        with autocast():  # Wrap the forward pass with autocast
            kind_preds, score_preds = model(input_ids, attention_mask)

            classification_loss = classification_loss_fn(kind_preds, kind)
            regression_loss = regression_loss_fn(
                score_preds[kind == 1], score[kind == 1]
            )

            loss = classification_loss + regression_loss
            total_loss += loss.item()

        # Scale the loss and perform backward pass
        scaler.scale(loss).backward()
        # Perform a step with the scaled gradients
        scaler.step(optimizer)
        # Update the scale for the next iteration
        scaler.update()

        scheduler.step()
        writer.add_scalar(
            os.path.join(HOME_DIR, "/Loss/train"),
            loss.item(),
            epoch * len(train_loader) + batch_idx,
        )

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, cache_dir=CACHE_DIR)

    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
    train_dataset = NewsDataset(train_data, tokenizer, max_length=MAX_LENGTH)
    val_dataset = NewsDataset(val_data, tokenizer, max_length=MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    writer = SummaryWriter(os.path.join(HOME_DIR, "runs/%s" % (MODEL_CHECKPOINT)))

    total_steps = len(train_loader) * EPOCHS

    latest_checkpoint = get_latest_checkpoint(MODEL_DIR)
    if latest_checkpoint:
        checkpoint_path = os.path.join(MODEL_DIR, latest_checkpoint)
        starting_epoch = int(latest_checkpoint.split("_")[-1])
        tqdm.write(
            f"Starting from checkpoint {checkpoint_path} (epoch {starting_epoch})"
        )
        model = MultiTaskBERTBasedModel(
            MODEL_CHECKPOINT=checkpoint_path, cache_dir=CACHE_DIR
        )
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
    else:
        starting_epoch = 0
        tqdm.write(f"Creating model from scratch {MODEL_CHECKPOINT}")
        model = MultiTaskBERTBasedModel(
            MODEL_CHECKPOINT=MODEL_CHECKPOINT, cache_dir=CACHE_DIR
        )
    if torch.cuda.device_count() > 1:
        model = DataParallel(model).cuda()
    model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    if latest_checkpoint and os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path))

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )
    if latest_checkpoint and os.path.exists(scheduler_path):
        scheduler.load_state_dict(torch.load(scheduler_path))

    # Training
    scaler = GradScaler()
    for epoch in range(starting_epoch, EPOCHS):
        tqdm.write(f"Epoch {epoch + 1}/{EPOCHS}")

        train_loss = train(
            model, train_loader, optimizer, scheduler, DEVICE, epoch, writer, scaler
        )
        val_classification_loss, val_regression_loss = evaluate(
            model, val_loader, DEVICE
        )

        tqdm.write(
            f"Train Loss: {train_loss:.4f} | Val Classification Loss: {val_classification_loss:.4f} | Val Regression Loss: {val_regression_loss:.4f}"
        )

        checkpoint_path = os.path.join(
            MODEL_DIR, f"{MODEL_CHECKPOINT}_epoch_{epoch + 1}"
        )
        save_checkpoints(model, optimizer, scheduler, MODEL_DIR, epoch + 1)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Multi-task BERT-based model")
    parser.add_argument(
        "--device",
        type=str,
        default="hpc",
        choices=["hpc", "lab", "rbmhpc", "metaserver"],
        help="Select device configurations. Available options: hpc, lab, rbmhpc, metaserver",
    )
    args = parser.parse_args()

    if args.device == "hpc":
        from config_hpc import *
    elif args.device == "lab":
        from config_lab import *
    elif args.device == "rbmhpc":
        from config_rbmhpc import *
    elif args.device == "metaserver":
        from config_metaserver import *
    main()
