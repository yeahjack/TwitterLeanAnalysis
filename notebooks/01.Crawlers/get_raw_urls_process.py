import pandas as pd
import bs4 as bs
import urllib.request
from tqdm import tqdm
import multiprocessing
import math
import os


def getRawURLs(df_stories, start, end):
    final_results = []
    for i in range(start, end):
        if pd.isnull(df_stories.loc[i, "raw_url"]):
            try:
                os.environ["HTTP_PROXY"] = "http://localhost:10809"
                story = urllib.request.urlopen(df_stories.iloc[i]["url"])
                sp_story = bs.BeautifulSoup(story, "html.parser", from_encoding="iso-8859-1")
                # df_stories.iloc[i].text = sp_story.find('div', {'class':'article-description'}).text.strip()

                final_results.append(
                    {
                        "title": df_stories.iloc[i]["title"],
                        "url": df_stories.iloc[i]["url"],
                        "source": df_stories.iloc[i]["source"],
                        "leaning": df_stories.iloc[i]["leaning"],
                        "text": sp_story.find(
                            "div", {"class": "article-description"}
                        ).text.strip(),
                        "raw_url": sp_story.find("div", {"class": "read-more-story"})
                        .find("a")
                        .get("href"),
                    }
                )
            except Exception as e:
                print(e, df_stories.iloc[i]["url"])
                final_results.append(
                    {
                        "title": df_stories.iloc[i]["title"],
                        "url": df_stories.iloc[i]["url"],
                        "source": df_stories.iloc[i]["source"],
                        "leaning": df_stories.iloc[i]["leaning"],
                        "text": "",
                        "raw_url": "",
                    }
                )
        else:
            final_results.append(
                {
                    "title": df_stories.iloc[i]["title"],
                    "url": df_stories.iloc[i]["url"],
                    "source": df_stories.iloc[i]["source"],
                    "leaning": df_stories.iloc[i]["leaning"],
                    "text": df_stories.iloc[i]["text"],
                    "raw_url": df_stories.iloc[i]["raw_url"],
                }
            )
    final_results = pd.DataFrame(final_results)
    final_results.to_csv(
        "news-title-bias/data/df_stories_raw_urls_proxy/df_stories_raw_urls_%s-%s.csv"
        % (start, end),
        index=False,
        encoding="utf_8_sig",
    )


if __name__ == "__main__":
    df_stories = pd.read_csv(
        "news-title-bias/data/df_stories_raw_urls_proxy.csv", encoding="utf_8_sig", header=0
    )
    num_processes = 3
    chunk_size = math.ceil(df_stories.shape[0] / num_processes)
    processes = []
    for i in tqdm(range(num_processes)):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, df_stories.shape[0])
        p = multiprocessing.Process(target=getRawURLs, args=(df_stories, start, end))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
