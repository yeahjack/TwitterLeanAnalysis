import pandas as pd
import bs4 as bs
import urllib.request
from tqdm import tqdm
import pickle
import multiprocessing
import math

# add info from link
def get_info_from_url_story(url_story):
    '''add info rom url on a a story
    '''
    story = urllib.request.urlopen(url_story)
    sp_story = bs.BeautifulSoup(story, 'html.parser')

    final_results_single = []
    
    # loop over main three articles
    for div in sp_story.find_all('a', {'class': 'news-title'}):
        try:
            title = div.text
            url = div.get('href')

            news_source = div.parent.find('div', {'class':'news-source'}).text
            leaning = div.parent.find('img', {'typeof': 'foaf:Image'}).get('title').replace("AllSides Media Bias Rating: ", '')

            news_text = div.parent.find('div', {'class': 'news-body'}).find('div', {'class': 'body-contents'}).text.strip()
            final_results_single.append({
                'title': title,
                'url': url,
                'source': news_source,
                'leaning': leaning,
                'text': news_text
            })
        except AttributeError:
            final_results_single.append({
                'title': '',
                'url': '',
                'source': '',
                'leaning': '',
                'text': ''
            })
    # loop for other articles
    for div in sp_story.find_all('div', {'class': 'news-title'}):
        try:
            title = div.text.strip()
            url = div.parent['href']
            news_source = div.parent.parent.find('div',{'class':'news-source'}).text
            leaning = div.parent.parent.find('img', {'typeof': 'foaf:Image'}).get('title').replace('AllSides Media Bias Rating: ', '')
            final_results_single.append({
                'title': title,
                'url': url,
                'source': news_source,
                'leaning': leaning,
                'text': ''
            })
        except AttributeError:
            final_results_single.append({
                'title': '',
                'url': '',
                'source': '',
                'leaning': '',
                'text': ''
            })
    return final_results_single

def get_stories(df, start, end):

    url_root = 'https://www.allsides.com/'
    '''Add list for all stories
    '''
    final_results = []
    for i in range(start, end):
        url_story = url_root + df.iloc[i]['url_story']
        final_results_single = get_info_from_url_story(url_story)
        final_results.append(pd.DataFrame(final_results_single))
    pd.concat(final_results).to_csv('news-title-bias/data/df_stories/text-%s-%s.csv'%(start, end), encoding='utf_8_sig')

if __name__ == '__main__':
    with open('news-title-bias/data/df_links.pkl', 'rb') as f:
        df = pickle.load(f)
    num_processes = 100
    chunk_size = math.ceil(df.shape[0] / num_processes)
    processes = []
    for i in tqdm(range(num_processes)):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, df.shape[0])
        p = multiprocessing.Process(target=get_stories, args=(df, start, end))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()