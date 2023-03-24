from newspaper import Article
from newspaper import Config
import pandas as pd
from tqdm import tqdm
import math
import multiprocessing
import time


def scraper(df_stories, start, end):
    config = Config()
    config.browser_user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
    config.request_timeout = 10
    config.proxies = {
        "http": "http://localhost:10809",
        "https": "http://localhost:10809",
    }
    result_list = []
    for i in range(start, end):
        if pd.isna(df_stories.at[i, "full_text"]):
            url = df_stories.iloc[i]["raw_url"]
            try:
                article = Article(url, config=config, memoize_articles=False)
                article.download()
                article.parse()
                result_list.append(
                    pd.DataFrame(
                        {
                            "title": df_stories.iloc[i]["title"],
                            "url": df_stories.iloc[i]["url"],
                            "source": df_stories.iloc[i]["source"],
                            "leaning": df_stories.iloc[i]["leaning"],
                            "text": df_stories.iloc[i]["text"],
                            "raw_url": df_stories.iloc[i]["raw_url"],
                            "full_text": article.text.replace("\n", " ")
                            .replace("\r", " ")
                            .replace("\t", " ")
                            .replace("\xa0", " ")
                            .strip(),
                            "error": ''
                        },
                        index=[i],
                    )
                )
            except Exception as e:
                if "HTTPSConnectionPool" in str(e):
                    try:
                        config = Config()
                        config.browser_user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
                        config.request_timeout = 30
                        article = Article(url, config=config, memoize_articles=False)
                        article.download()
                        article.parse()
                        result_list.append(
                            pd.DataFrame(
                                {
                                    "title": df_stories.iloc[i]["title"],
                                    "url": df_stories.iloc[i]["url"],
                                    "source": df_stories.iloc[i]["source"],
                                    "leaning": df_stories.iloc[i]["leaning"],
                                    "text": df_stories.iloc[i]["text"],
                                    "raw_url": df_stories.iloc[i]["raw_url"],
                                    "full_text": article.text.replace("\n", " ")
                                    .replace("\r", " ")
                                    .replace("\t", " ")
                                    .replace("\xa0", " ")
                                    .strip(),
                                    "error": ''
                                },
                                index=[i],
                            )
                        )
                    except Exception as ee:
                        result_list.append(
                            pd.DataFrame(
                                {
                                    "title": df_stories.iloc[i]["title"],
                                    "url": df_stories.iloc[i]["url"],
                                    "source": df_stories.iloc[i]["source"],
                                    "leaning": df_stories.iloc[i]["leaning"],
                                    "text": df_stories.iloc[i]["text"],
                                    "raw_url": df_stories.iloc[i]["raw_url"],
                                    "full_text": "",
                                    "error": str(ee)[0:52],
                                },
                                index=[i],
                            )
                        )
                else:
                    result_list.append(
                        pd.DataFrame(
                            {
                                "title": df_stories.iloc[i]["title"],
                                "url": df_stories.iloc[i]["url"],
                                "source": df_stories.iloc[i]["source"],
                                "leaning": df_stories.iloc[i]["leaning"],
                                "text": df_stories.iloc[i]["text"],
                                "raw_url": df_stories.iloc[i]["raw_url"],
                                "full_text": "",
                                "error": str(e)[0:52],
                            },
                            index=[i],
                        )
                    )
        else:
            result_list.append(
                pd.DataFrame(
                    {
                        "title": df_stories.iloc[i]["title"],
                        "url": df_stories.iloc[i]["url"],
                        "source": df_stories.iloc[i]["source"],
                        "leaning": df_stories.iloc[i]["leaning"],
                        "text": df_stories.iloc[i]["text"],
                        "raw_url": df_stories.iloc[i]["raw_url"],
                        "full_text": df_stories.iloc[i]["full_text"],
                        "error": ''
                    },
                    index=[i],
                )
            )
    pd.concat(result for result in result_list).to_csv(
        "news-title-bias/data/df_stories_raw_urls_contents_20_proxy/text-%s-%s.csv"
        % (start, end),
        encoding="utf_8_sig",
    )


if __name__ == "__main__":
    df_stories = pd.read_csv(
        "news-title-bias/data/df_stories_raw_urls_contents_19_proxy.csv"
    )
    num_processes = 50
    chunk_size = math.ceil(df_stories.shape[0] / num_processes)
    processes = []
    for i in tqdm(range(num_processes)):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, df_stories.shape[0])
        p = multiprocessing.Process(target=scraper, args=(df_stories, start, end))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()