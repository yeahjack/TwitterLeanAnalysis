from simpletransformers.classification import ClassificationModel, ClassificationArgs, MultiLabelClassificationModel
from transformers import BertTokenizer, XLMRobertaTokenizer, AutoTokenizer, XLMRobertaModel
import pandas as pd
import logging
from tqdm import tqdm
import sklearn
import os
import numpy as np
from torch import nn

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
print('Loading data...')
train_df = pd.read_csv('/home/vptlo-hpc-workstation-c1/news-title-bias/data/dataset/dataset_combined_multitask_train.csv',
                       names=['text','kind','score'],
                       encoding='utf-8')
test_df = pd.read_csv('/home/vptlo-hpc-workstation-c1/news-title-bias/data/dataset/dataset_combined_multitask_test.csv',
                      names=['text','kind','score'],
                      encoding='utf-8')
val_df = pd.read_csv('/home/vptlo-hpc-workstation-c1/news-title-bias/data/dataset/dataset_combined_multitask_val.csv',
                     names=['text','kind','score'],
                     encoding='utf-8')
print('Data loaded.')
class CustomMultiTaskModel(ClassificationModel):
    def __init__(self, model_type, model_name, num_labels=None, args=None, use_cuda=True, cuda_device=-1, **kwargs):
        super().__init__(model_type, model_name, num_labels=num_labels, args=args, use_cuda=use_cuda, cuda_device=cuda_device, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.regression_head = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, **inputs):
        transformer_outputs = self.model(**inputs)
        logits = transformer_outputs[0]
        regression_output = self.regression_head(transformer_outputs[1])
        return (logits, regression_output)

    def predict(self, to_predict, split_into_words=False):
        preds = super().predict(to_predict, split_into_words=split_into_words)
        logits = preds[0]
        label1_preds = np.argmax(logits, axis=1)
        scores = np.max(logits, axis=1)
        return label1_preds, scores

train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "num_train_epochs": 10,
    "train_batch_size": 320,
    "eval_batch_size": 320,
    "max_seq_length": 128,
    "learning_rate": 2e-5,
    'n_gpu': 2,
    'no_cache': False,
    'output_dir': 'output/',
    'use_multiprocessing': True,
}

model = CustomMultiTaskModel("xlmroberta", "xlm-roberta-base", num_labels=2, args=train_args)

model.train_model(train_df, eval_df=val_df, acc=sklearn.metrics.accuracy_score)
