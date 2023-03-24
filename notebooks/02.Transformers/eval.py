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

train_df = pd.read_csv(
    '/home/data/xuyijie/Documents/AllsidesScraper/news-title-bias/data/dataset/train_combined.csv',
    header=None,
    encoding='utf-8')
test_df = pd.read_csv(
    '/home/data/xuyijie/Documents/AllsidesScraper/news-title-bias/data/dataset/test_combined.csv',
    header=None,
    encoding='utf-8')
val_df = pd.read_csv(
    '/home/data/xuyijie/Documents/AllsidesScraper/news-title-bias/data/dataset/val_combined.csv',
    header=None,
    encoding='utf-8')
all_df = pd.read_csv(
    '/home/data/xuyijie/Documents/AllsidesScraper/news-title-bias/data/dataset/dataset_combined.csv',
    header=None,
    encoding='utf-8')

train_df.columns = ["text", "labels"]
test_df.columns = ["text", "labels"]
val_df.columns = ["text", "labels"]
all_df.columns = ["text", "labels"]
train_df['labels'] = train_df['labels'] + 2
test_df['labels'] = test_df['labels'] + 2
val_df['labels'] = val_df['labels'] + 2
all_df['labels'] = all_df['labels'] + 2

# Evaluation by Folders.

# Get Folders.
folder = '/home/data/xuyijie/Documents/AllsidesScraper/news-title-bias/notebooks/02.Transformers/Models/outputs-xlmroberta-6gpu-batchsize384-20epoch_6label'
dir_list = []
for root, dirs, files in os.walk(folder):
    dir_list.append(dirs)
dirs = dir_list[0]

# Use numbers of GPU.
model_args = ClassificationArgs()
model_args.n_gpu = 6
model_args.eval_batch_size = 16384
model_args.reprocess_input_data = False
model_args.no_cache = False
model_args.dataloader_num_workers = 32

cpu_num = 5 # 这里设置成你想运行的CPU个数
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

result_df = []
for dir in dirs[:4]:
    model = ClassificationModel(
        "xlmroberta", 
        os.path.join(folder,dir),
        num_labels=6,
        args=model_args,
        use_cuda=True
    )
    result, model_outputs = model.predict(all_df['text'].tolist())
    result_df.append({'dir': dir, 'accuracy': sklearn.metrics.accuracy_score(all_df['labels'], result), 'confusion_matrix': sklearn.metrics.confusion_matrix(all_df['labels'], result)})
result_df = pd.DataFrame(result_df)
result_df.to_csv(os.path.join(folder, 'result_%s.csv'%(str(dir))))

'''
import matplotlib.pyplot as plt

dir_best_model = os.path.join(folder, result_df.dir[result_df['accuracy'].idxmax()])
model = ClassificationModel("roberta",
                            os.path.join(folder, dir_best_model),
                            num_labels=6,
                            args=model_args)
result, model_outputs, wrong_predictions = model.eval_model(val_df)

sklearn.metrics.ConfusionMatrixDisplay(
    sklearn.metrics.confusion_matrix(val_df['labels'],
                                     model_outputs.argmax(axis=1)),
    display_labels=[
        'Left', 'Lean Left', 'Center', 'Lean Right', 'Right', 'Daily'
    ]).figure_.savefig(os.path.join(folder, 'confusion_matrix.png'))
'''
