# config_metaserver.py
import os
import torch
EPOCHS = 20
BATCH_SIZE = 200
LEARNING_RATE = 2e-5
MAX_LENGTH = 2048
MODEL_CHECKPOINT = "allenai/longformer-base-4096"
HOME_DIR = "/home/yijiexu/news-title-bias/notebooks/06.Multi-task_Learning"
CACHE_DIR = os.path.join(HOME_DIR, "cache_pretrained")
DATA_PATH = "/home/yijiexu/news-title-bias/data/dataset/dataset_combined_multitask.csv"
DATA_PREPROCESSED_DIR = os.path.join(HOME_DIR, "cache_tokenizer")
MODEL_DIR = os.path.join(
    HOME_DIR, "runs/output/model/mixed-precision/%s/epoch_%s"%(MODEL_CHECKPOINT.replace('/','-'),BATCH_SIZE)
)
WARMUP_STEPS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")