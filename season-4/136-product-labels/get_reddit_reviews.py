"""
Get Reddit Results | Cannlytics
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 12/1/2023
Updated: 12/2/2023
License: MIT License <https://github.com/cannlytics/cannlytics/blob/main/LICENSE>

Description:

    This tool collects cannabis product reviews and associated product
    images to perform research.
    
    Product data from the product label images, as well as natural
    language data from the review, such as sentiment rating, can be used
    to analyze how product data may affect how the product is reviewed.

"""
# External imports:
from dotenv import dotenv_values
import pandas as pd
import praw


# Initialize Reddit.
config = dotenv_values('../../../.env')
reddit = praw.Reddit(
    client_id=config['REDDIT_CLIENT_ID'],
    client_secret=config['REDDIT_SECRET'],
    password=config['REDDIT_PASSWORD'],
    user_agent=config['REDDIT_USER_AGENT'],
    username=config['REDDIT_USERNAME'],
)

# Get reddit data and keep track of text, comments, votes, etc.
headlines = set()
for submission in reddit.subreddit('Investing').hot(limit=None):
    headlines.add(submission.title)
    print(len(headlines))
    print (submission.title)
    print (submission.id)
    print (submission.author)
    print (submission.score)
    print (submission.upvote_ratio)
    print (submission.url)

# Save images and COAs and indicator for what data is in the post
df = pd.DataFrame(headlines)

# Estimate sentiment of reviews.


# Correlate sentiment with cannabinoids and terpenes.
# Question: How do consumers feel about products with Farnesene?
