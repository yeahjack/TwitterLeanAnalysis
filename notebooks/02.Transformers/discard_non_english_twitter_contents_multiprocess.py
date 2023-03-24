
import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq
from multiprocessing import Process
import numpy as np
from ftlangdetect import detect
import pyarrow as pa

def detect_lang(x):
    return detect(''.join(x.replace('\n', ' ')))['lang']
    
# Define the function to be applied to each row
def process_chunk(i):
    twitter_content = pq.read_table('/home/data/xuyijie/Documents/AllsidesScraper/news-title-bias/data/TwitterData/Twitter_Contents_Split_FastLangDetect/%s.parquet'%i).to_pandas()
    # Add Language to each row
    twitter_content = twitter_content.assign(Language='')
    twitter_content["Language"] = twitter_content["TweetContent"].progress_apply(detect_lang)
    pq.write_table(pa.Table.from_pandas(twitter_content), '/home/data/xuyijie/Documents/AllsidesScraper/news-title-bias/data/TwitterData/Twitter_Contents_Split_FastLangDetect_Results/%s.parquet'%i)


if __name__ == '__main__':
    tqdm.pandas()
    
    # Choose the number of processes to use
    num_processes = 32  # for example
    processes = []
    
    for i in range(num_processes):
        p = Process(target=process_chunk, args=(i,))
        processes.append(p)
        p.start()
        
    for process in processes:
        process.join()
    