from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import pyarrow.parquet as pq
import warnings
import pyarrow as pa
from tqdm import tqdm
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
model_args = ClassificationArgs()
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7,3"
model_args.n_gpu = 5
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.no_cache = True
model_args.cache_dir = "cache_dir"
model_args.use_cached_eval_features = True
model_args.eval_batch_size = 20480
model_args.dataloader_num_workers = 64
cpu_num = 64 # 这里设置成你想运行的CPU个数
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
torch.multiprocessing.set_sharing_strategy("file_system")

model = ClassificationModel(
    "xlmroberta",
    "/home/data/xuyijie/Documents/AllsidesScraper/news-title-bias/notebooks/02.Transformers/Models/outputs-xlmroberta-6gpu-batchsize384-20epoch_6label/checkpoint-11820-epoch-20",
    num_labels=6,
    args=model_args,
)
print("model loaded")
n_intervals = 50
for i in tqdm(range(31, 40)):
    twitter_content = pq.read_table(
    "/home/data/xuyijie/Documents/AllsidesScraper/news-title-bias/data/TwitterData/Twitter_Contents_to_50/%s.parquet"%i).to_pandas()
    
    #twitter_content = pd.read_csv('/hpc/users/CONNECT/yxu409/Documents/AllsidesScraper/news-title-bias/data/TwitterData/Twitter_Contents_to_10/%s.csv'%i)
    try:
        predictions = pd.DataFrame(model.predict(
            twitter_content['TweetContent'].tolist())[0], columns=["Predictions"]).set_index(twitter_content.index)
        twitter_content = pd.concat([twitter_content, predictions], axis=1)
        del predictions
        pq.write_table(
            pa.Table.from_pandas(pd.DataFrame(twitter_content)),
            "/home/data/xuyijie/Documents/AllsidesScraper/news-title-bias/data/TwitterData/Predictions_50_6kind/%s.parquet"%i)
    except BaseException as e:
        print(e)
        continue
