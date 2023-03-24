from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
from tqdm import tqdm
import sklearn
import os

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_df = pd.read_csv('/hpc/users/CONNECT/yxu409/Documents/AllsidesScraper/news-title-bias/data/dataset/train_combined.csv',
                       header=None,
                       encoding='utf-8')
test_df = pd.read_csv('/hpc/users/CONNECT/yxu409/Documents/AllsidesScraper/news-title-bias/data/dataset/test_combined.csv',
                      header=None,
                      encoding='utf-8')
val_df = pd.read_csv('/hpc/users/CONNECT/yxu409/Documents/AllsidesScraper/news-title-bias/data/dataset/val_combined.csv',
                     header=None,
                     encoding='utf-8')

train_df.columns = ["text", "labels"]
test_df.columns = ["text", "labels"]
val_df.columns = ["text", "labels"]
train_df['labels'] = train_df['labels'] + 2
test_df['labels'] = test_df['labels'] + 2
val_df['labels'] = val_df['labels'] + 2

model_args = ClassificationArgs()
model_args.num_train_epochs = 50
model_args.overwrite_output_dir = True
model_args.n_gpu = 4
model_args.no_cache = False
model_args.train_batch_size = 1280
model_args.output_dir = '/hpc/users/CONNECT/yxu409/Documents/AllsidesScraper/news-title-bias/notebooks/02.Transformers/Models/outputs-roberta-%sgpu-batchsize%s-%sepoch_combined/' % (
    str(model_args.n_gpu), str(
        model_args.train_batch_size), str(model_args.num_train_epochs))
model_args.reprocess_input_data = False

# Save Models

model_args.save_steps = -1
model_args.save_model_every_epoch = True
model_args.save_eval_checkpoints = False

# Evaluation

model_args.best_model_dir = model_args.output_dir + 'best-model'
model_args.evaluate_during_training = False
model_args.evaluate_during_training_verbose = False
model_args.eval_batch_size = 64

# Early Stopping

model_args.use_early_stopping = False
model_args.early_stopping_delta = 0.01
model_args.early_stopping_metric = "mcc"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 5
model_args.evaluate_during_training_steps = 1000

# Create a ClassificationModel
model = ClassificationModel(
    "roberta",
    'roberta-base',
    num_labels=6,
    args=model_args,
)

# Train the model
import sklearn

model.train_model(train_df, eval_df=val_df, acc=sklearn.metrics.accuracy_score)
'''
# Evaluation by Folders.

# Get Folders.
folder = '/hpc/users/CONNECT/yxu409/Documents/AllsidesScraper/news-title-bias/notebooks/02.Transformers/Models/outputs-roberta-4gpu-batchsize1000-10epoch_combined' + '/'
dir_list = []
for root, dirs, files in os.walk(folder):
    dir_list.append(dirs)
dirs = dir_list[0]

# Use numbers of GPU.
model_args = ClassificationArgs()
model_args.n_gpu = 4

model_args.reprocess_input_data = False
model_args.no_cache = False

result_df = []
for dir in tqdm(dirs):
    model = ClassificationModel("roberta",
                                '%s%s' % (folder, dir),
                                num_labels=6,
                                args=model_args)
    result, model_outputs, wrong_predictions = model.eval_model(val_df)
    result_df.append({
        'dir':
        dir,
        'accuracy':
        sklearn.metrics.accuracy_score(val_df['labels'],
                                       model_outputs.argmax(axis=1)),
        'confusion_matrix':
        sklearn.metrics.confusion_matrix(val_df['labels'],
                                         model_outputs.argmax(axis=1))
    })
result_df = pd.DataFrame(result_df)

result_df.to_csv(
    '/home/yijiexu/Documents/AllsidesScraper/news-title-bias/notebooks/02.Transformers/Models/outputs-roberta-4gpu-batchsize1000-10epoch_combined/result.csv'
)

import matplotlib.pyplot as plt

sklearn.metrics.ConfusionMatrixDisplay(
    sklearn.metrics.confusion_matrix(val_df['labels'],
                                     model_outputs.argmax(axis=1)),
    display_labels=[
        'Left', 'Lean Left', 'Center', 'Lean Right', 'Right', 'Daily'
    ]
).savefig(
    '/home/yijiexu/Documents/AllsidesScraper/news-title-bias/notebooks/02.Transformers/Models/outputs-roberta-4gpu-batchsize1000-10epoch_combined/confusion_matrix.png'
)
'''