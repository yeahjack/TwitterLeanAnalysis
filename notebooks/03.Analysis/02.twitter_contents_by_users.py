import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
import multiprocessing
import pickle

def get_contents_by_user(df, users, start, end):
    for i in tqdm(range(start, end)):
        with open('news-title-bias/data/TwitterData/TweetContentByUsers/%s.pkl'%(users[i][20:]), 'wb') as f:
            pickle.dump(df[df['User'] == users[i]], f)


if __name__ == '__main__':
    df = pq.read_table("news-title-bias/data/TwitterData/df_summary_with_predictions.parquet").to_pandas()
    print("data loaded")
    users = pd.unique(df['User'])
    n_process = 128 
    jobs = []
    for process in range(n_process):
        start = process * len(users) // n_process
        if process == n_process - 1:
            end = len(users)
        else:
            end = (process + 1) * len(users) // n_process
        p = multiprocessing.Process(target=get_contents_by_user, args=(df, users, start, end))
        jobs.append(p)
        p.start()
    
    for proc in jobs:
        proc.join()
        
    