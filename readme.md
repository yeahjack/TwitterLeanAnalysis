# Scraping data on news title bias

Different sources cover the same news in very different ways! This project scrapes a dataset of article titles, sources, political leanings, and text from curated articles on [allsides.com](https://www.allsides.com/unbiased-balanced-news).

# My Modifications

Based on the scraper given above, I also made some modificatios such that

1. Using Python web spiders to crawl tens of thousands of tweets related to a political keyword such as "presidential election" on Twitter.
2. Obtaining users' information, such as their ID, for each tweet obtained above.
3. Crawling over 77 million tweets for all the users from the previous step.
4. Using a website called **allsides.com** to categorize over 10 thousand articles with different political leanings into 5 categories, from left to right. This idea comes from “The Interaction between Political Typology and Filter Bubbles in News Recommendation Algorithms”, published in WWW 21’.
5. Fine-tuning a classification network based on BERT and applying it to all the tweets crawled in step 3.

# Reference

- uses data from [allsides.com](https://www.allsides.com/unbiased-balanced-news)
- Allsides Scrapers made by [@csinva_](https://twitter.com/csinva_)
