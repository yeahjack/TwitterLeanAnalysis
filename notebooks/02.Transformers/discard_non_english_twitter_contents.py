from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import os
from pprint import pprint

twitter_content = pq.read_table(
    "/home/data/xuyijie/Documents/AllsidesScraper/news-title-bias/data/TwitterData/Twitter_Contents_to_50/0.parquet").to_pandas()
print("data loaded")

twitter_content = twitter_content.assign(Language='')

# Discard non-English Tweets
from ftlangdetect import detect
def detect_lang(x):
    return detect(''.join(x.replace('\n', ' ')))['lang']

for i in tqdm(range(len(twitter_content))):
    try:
        twitter_content['Langauge'].loc[i] = detect_lang(twitter_content['TweetContent'].loc[i])
    except:
        continue
    
pq.write_table(pa.Table.from_pandas(twitter_content), "/home/data/xuyijie/Documents/AllsidesScraper/news-title-bias/data/TwitterData/df_all_tweets_with_langs.parquet")
print('Save all langs complete')