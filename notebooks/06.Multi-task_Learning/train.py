import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import pandas as pd
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.nn.parallel import DataParallel
from transformers import get_linear_schedule_with_warmup
# Import TensorBoard packages
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


class MultiTaskXLMRoberta(nn.Module):
    def __init__(self, cache_dir=None):
        super(MultiTaskXLMRoberta, self).__init__()
        self.xlm_roberta = XLMRobertaModel.from_pretrained(
            "xlm-roberta-base", cache_dir=cache_dir
        )
        self.classification_head = nn.Linear(self.xlm_roberta.config.hidden_size, 2)
        self.regression_head = nn.Linear(self.xlm_roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.xlm_roberta(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:, 0]

        classification_output = self.classification_head(pooled_output)
        regression_output = self.regression_head(pooled_output)
        return classification_output, regression_output.squeeze(-1)


def train(model, train_loader, optimizer, scheduler, device, epoch, writer):
    model.train()
    total_loss = 0
    classification_loss_fn = nn.CrossEntropyLoss()
    regression_loss_fn = nn.MSELoss()

    # Add 'leave=False' to keep tqdm progress bar on the same line
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
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

        # Log the loss to TensorBoard
        writer.add_scalar("/home/xuyijie/news-title-bias/notebooks/06.Multi-task_Learning/Loss/train", loss.item(), epoch * len(train_loader) + batch_idx)
        
        # Update the tqdm progress bar with the current loss
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(train_loader)


# Hyperparameters
epochs = 5
batch_size = 56
learning_rate = 2e-5
cache_dir = (
    "/home/xuyijie/news-title-bias/notebooks/06.Multi-task_Learning/cache_pretrained"
)

# Load the data
data = pd.read_csv(
    "/home/xuyijie/news-title-bias/data/dataset/dataset_combined_multitask.csv",
    names=["text", "kind", "score"],
    header=0,
)
data["score"].fillna(0, inplace=True)
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base", cache_dir=cache_dir)
dataset = NewsDataset(data, tokenizer)

# Split data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
train_dataset = NewsDataset(train_data, tokenizer)
val_dataset = NewsDataset(val_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create a TensorBoard SummaryWriter
writer = SummaryWriter("/home/xuyijie/news-title-bias/notebooks/06.Multi-task_Learning/runs/multitask_xlm_roberta")

# Learning rate scheduling
warmup_steps = 100  # You can adjust this value
total_steps = len(train_loader) * epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize the model, wrap it with DataParallel, and move it to the device
model = MultiTaskXLMRoberta(cache_dir=cache_dir)
if torch.cuda.device_count() > 1:
    model = DataParallel(model)
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Create a learning rate scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# Training loop
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, scheduler, device, epoch, writer)
    tqdm.write(f"Epoch: {epoch+1}/{epochs}, Loss: {train_loss:.4f}")

    # Save the model after each epoch
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), f"output/model_epoch_{epoch+1}.pth")

# Close the TensorBoard writer
writer.close()
