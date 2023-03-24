from newspaper import Article
from newspaper import Config
import pandas as pd
from tqdm import tqdm
import math
import threading

def scraper(df_stories, start, end):
    config = Config()
    config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    result_list = []
    for i in range(start, end):
        url = df_stories.iloc[i]['raw_url']
        try:
            article = Article(url, config=config)
            article.download()
            article.parse()
            result_list.append({'index': i, 'text': article.text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')})
        except:
            result_list.append({'index': i, 'text': ''})
    pd.concat(pd.DataFrame(result) for result in result_list).to_csv('news-title-bias/data/full_text_thread/text-%s-%s.csv'%(start, end), encoding='utf_8_sig')


if __name__ == '__main__':
    df_stories = pd.read_csv('news-title-bias/data/temp.csv', index=False)
    num_threads = 2400
    chunk_size = math.ceil(df_stories.shape[0] / num_threads)
    threads = []
    for i in tqdm(range(num_threads)):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, df_stories.shape[0])
        t = threading.Thread(target=scraper, args=(df_stories, start, end))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()