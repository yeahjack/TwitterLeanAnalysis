from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
from tqdm import tqdm
import sklearn
import os
import torch


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
torch.multiprocessing.set_sharing_strategy("file_system")

# Combined Dataset
train_df = pd.read_csv(
    '/home/data/xuyijie/Documents/AllsidesScraper/news-title-bias/data/dataset/train_combined.csv', header=None, encoding='utf-8')
test_df = pd.read_csv(
    '/home/data/xuyijie/Documents/AllsidesScraper/news-title-bias/data/dataset/test_combined.csv', header=None, encoding='utf-8')
val_df = pd.read_csv(
    '/home/data/xuyijie/Documents/AllsidesScraper/news-title-bias/data/dataset/val_combined.csv', header=None, encoding='utf-8')
all_df = pd.read_csv(
    '/home/data/xuyijie/Documents/AllsidesScraper/news-title-bias/data/dataset/dataset_combined.csv', header=None, encoding='utf-8')

train_df.columns = ["text", "labels"]
test_df.columns = ["text", "labels"]
val_df.columns = ["text", "labels"]
all_df.columns = ["text", "labels"]
train_df['labels'] = train_df['labels'] + 2
test_df['labels'] = test_df['labels'] + 2
val_df['labels'] = val_df['labels'] + 2
all_df['labels'] = all_df['labels'] + 2

num_labels = 6
model_args = ClassificationArgs()
model_args.num_train_epochs = 20
model_args.overwrite_output_dir = True
model_args.n_gpu = 6
model_args.no_cache = False
model_args.train_batch_size = 384
model_args.output_dir = '/home/data/xuyijie/Documents/AllsidesScraper/news-title-bias/notebooks/02.Transformers/Models/outputs-xlmroberta-%sgpu-batchsize%s-%sepoch_%slabel/' % (
    str(model_args.n_gpu), str(model_args.train_batch_size), str(model_args.num_train_epochs), str(num_labels))
model_args.reprocess_input_data = False

# Save Models

model_args.save_steps = -1
model_args.save_model_every_epoch = True
model_args.save_eval_checkpoints = False

# Evaluation

model_args.best_model_dir = model_args.output_dir + 'best-model'
model_args.evaluate_during_training = False
model_args.evaluate_during_training_verbose = False
model_args.eval_batch_size = 1

# Early Stopping

model_args.use_early_stopping = False
model_args.early_stopping_delta = 0.01
model_args.early_stopping_metric = "mcc"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 5
model_args.evaluate_during_training_steps = 1000

# Create a ClassificationModel
model = ClassificationModel(
    "xlmroberta",
    'xlm-roberta-large',
    num_labels=num_labels,
    args=model_args,
)

# Train the model
model.train_model(train_df, eval_df=val_df, acc=sklearn.metrics.accuracy_score,
                  mcc=sklearn.metrics.matthews_corrcoef)
