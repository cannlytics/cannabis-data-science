"""
CoA Parsing for Consumer Product Recommendation
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 8/23/2022
Updated: 9/7/2022
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    Parse a producer's corpus of CoAs, create a standardized datafile, then
    use the data, augmented with data about consumer's prior purchases, to
    create product recommendations for each consumer.

Data Source:

    - Raw Garden Lab Results
    URL: <https://rawgarden.farm/lab-results/>
    Collection: See `parse_rawgarden_coas.py`

    - Strain Reviews
    https://cannlytics.page.link/reported-effects

"""
# Standard imports.
from datetime import datetime
import math
import os
from time import sleep

# External imports.
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

# Internal imports.
from cannlytics.data.coas import CoADoc
from cannlytics.utils.constants import DEFAULT_HEADERS

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})

# Specify where your data lives.
DATA_DIR = '../../.datasets'
COA_DATA_DIR = f'{DATA_DIR}/lab_results/raw_garden'
COA_PDF_DIR = f'{COA_DATA_DIR}/pdfs'
REVIEW_DATA_DIR = f'{DATA_DIR}/effects'

# Create directories if they don't already exist.
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
if not os.path.exists(COA_DATA_DIR): os.makedirs(COA_DATA_DIR)
if not os.path.exists(COA_PDF_DIR): os.makedirs(COA_PDF_DIR)
if not os.path.exists(REVIEW_DATA_DIR): os.makedirs(REVIEW_DATA_DIR)


#-----------------------------------------------------------------------
# Look at the data.
#-----------------------------------------------------------------------

# DEV:
outfile = f'{COA_DATA_DIR}/rawgarden-coa-data-2022-08-31T14-05-09.xlsx'
# outfile = f'{COA_DATA_DIR}/rawgarden-coa-data-2022-09-07T13-39-09.xlsx'

# Read the CoA data back in.
coa_values = pd.read_excel(outfile, sheet_name='Values')

# Visualize beta-pinene to d-limonene.
sns.scatterplot(
    data=coa_values.loc[
        (coa_values['beta_pinene'] > 0.001) &
        (coa_values['d_limonene'] > 0.001)
    ],
    x='d_limonene',
    y='beta_pinene',
)
plt.title('beta-Pinene to d-Limonene')
plt.show()

# Visualize the beta-pinene / d-limonene ratio.
coa_values['pine_lime_ratio'] = coa_values.eval('beta_pinene / d_limonene')
coa_values['pine_lime_ratio'].hist(bins=100)
plt.ylabel('Count')
plt.xlabel('Ratio')
plt.title('beta-Pinene to d-Limonene Ratio')
plt.show()

# Visualize the log of the beta-pinene / d-limonene ratio.
coa_values['log_pine_lime_ratio'] = coa_values['pine_lime_ratio'].apply(np.log)
coa_values['log_pine_lime_ratio'].hist(bins=100)
plt.xlim(-3, 3)
plt.xlabel('Log Ratio')
plt.ylabel('Count')
plt.title('Log beta-Pinene to d-Limonene Ratio')
plt.show()


#-----------------------------------------------------------------------
# Read in supplementary data: consumer reviews.
#-----------------------------------------------------------------------

# Read in the reviews.
datafile = f'{REVIEW_DATA_DIR}/strain-reviews-2022-06-15.xlsx'
reviews = pd.read_excel(datafile, index_col=0)

# Remove duplicates.
reviews.drop_duplicates(subset='review', inplace=True)


#-----------------------------------------------------------------------
# Perform sentiment analysis on the reviews.
#-----------------------------------------------------------------------

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import statistics

# Create natural language processing (NLP) client.
sia = SIA()

def avg_positivity(text: str) -> bool:
    """Average of all sentence compound scores."""
    scores = [
        sia.polarity_scores(sentence)['compound']
        for sentence in nltk.sent_tokenize(text)
    ]
    try:
        return statistics.mean(scores)
    except:
        return 0

# Rank reviews -1 to 1.
reviews['sentiment_score'] = reviews['review'].apply(lambda x: avg_positivity(x))

# Scale sentiment score from 0 to 1.
reviews['sentiment_score'] = reviews['sentiment_score'].apply(lambda x: (x + 1) / 2)

# Visualize the sentiment score.
reviews['sentiment_score'].hist(bins=100)
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Count')
plt.show()

# Exclude neutral reviews.
reviews['sentiment_score'].loc[
    reviews['sentiment_score'] != 0.5
].hist(bins=100)
plt.xlabel('Sentiment Score')
plt.ylabel('Count')
plt.title('Distribution of Sentiment Scores, Excluding Outliers')
plt.show()


#-----------------------------------------------------------------------
# Calculate each user's weighted average chemical profile by sentiment.
#-----------------------------------------------------------------------

# Isolate a training sample.
sample = reviews.loc[
    (reviews['beta_pinene'] > 0) &
    (reviews['d_limonene'] > 0) &
    (reviews['sentiment_score'] != 0.5)
    # (train['beta_caryophyllene'] > 0) &
    # (train['humulene'] > 0) &
    # (train['total_thc'] > 0) &
    # (train['total_cbd'] > 0) &
]

def weighted_avg(df, values, weights, by):
    """Calculate a weighted average.
    Args:

    Returns:

    Author: kadee <https://stackoverflow.com/a/33054358/5021266>
    License: CC BY-SA 3.0 <https://creativecommons.org/licenses/by-sa/3.0/>
    """
    v, w = df[values], df[weights]
    return (v * w).groupby(by).sum() / w.groupby(by).sum()


sample['pine_lime_ratio'] = sample['beta_pinene'].div(sample['d_limonene'])

# FIXME:
# Calculate a weighted average using `sentiment_score`.
# user_data = weighted_avg(
#     sample,
#     values='pine_lime_ratio',
#     weights='sentiment_score',
#     by='user'
# )


# Simple average:
drop = ['strain_name', 'review', 'category']
users = sample.drop(columns=drop).groupby('user')
user_data = users.mean()
user_data['n_reviews'] = sample.groupby('user')['review'].nunique()

# Determine the unique number of users.
# users = sample.groupby('user', as_index=False)
# user_data['n_reviews'] = sample.groupby('user')['review'].nunique()

# Remove users with less than 10 reviews.
user_data = user_data.loc[
    (user_data.index != 'Anonymous') &
    (user_data['n_reviews'] > 10)
]

# Plot beta-pinene to d-limonene by user.
x, y = 'd_limonene', 'beta_pinene'
ax = sns.scatterplot(
    x=x,
    y=y,
    data=user_data,
    hue='pine_lime_ratio',
    size='n_reviews',
    sizes=(200, 2000),
    legend=None,
    palette='viridis',
    # alpha=0.7,
    edgecolor=None,
)
# for index, row in user_data.iterrows():
#     if row['n_reviews'] < 30:
#         continue
#     ax.text(
#         row[x],
#         row[y],
#         # row['user'],
#         index,
#         horizontalalignment='center',
#         verticalalignment='bottom',
#         size='medium',
#         color='#2A8936',
#     )
# plt.xlim(0)
# plt.ylim(0)
plt.ylabel('beta-Pinene (%)', labelpad=5)
plt.xlabel('D-Limonene (%)', labelpad=5)
plt.title('Average beta-Pinene to D-Limonene Strain Profiles by User')
fig = plt.gcf()
notes = 'Notes: Data is from 1,609 reviews by 97 users. Historical strain chemical profile averages we merged with the reviews. The size of each point indicates the number of reviews, ranging from 11 to 56 reviews. The color depicts the beta-pinene to D-limonene ratio, ranging from 0.28 in purple to 0.96 in yellow.'
fig.text(0, -0.066, notes, ha='left', va='center', fontsize=21, wrap=True, transform=fig.transFigure)
plt.tight_layout()
plt.savefig(f'figures/pinene_to_limonene_by_user.png', bbox_inches='tight', dpi=300)
plt.show()

# Save the ratio for future use.
ratio = sample.eval(f'{y} / {x}').rename('ratio')
sample['pine_lime_ratio'] = ratio

# Visualize the distribution of ratios.
sample['pine_lime_ratio'].hist(bins=50)
plt.title('Distribution of Average beta-Pinene to d-Limonene by User')
plt.show()

# Visualize the distribution after a log transformation.
sample['pine_lime_ratio'].apply(np.log).hist(bins=50)
plt.title('Distribution of Log of Average beta-Pinene to d-Limonene by User')
plt.show()

sample['log_pine_lime_ratio'] = sample['pine_lime_ratio'].apply(np.log)


#-----------------------------------------------------------------------
# Estimate a product recommendation model.
#-----------------------------------------------------------------------

# Define the features to use.
features = ['log_pine_lime_ratio']
X = coa_values[features].dropna(how='all')

# Fit a nearest neighbors model.
model = NearestNeighbors(n_neighbors=4, algorithm='ball_tree')
model.fit(X)


#-----------------------------------------------------------------------
# Use the model for prediction.
#-----------------------------------------------------------------------

# Specify the user.
x_hat = sample.sample(1, random_state=420)

# Predict the k-nearest neighbors given a user's chemotype profile.
distance, prediction = model.kneighbors(x_hat[['log_pine_lime_ratio']])
y_hat = X.iloc[prediction[0]]

# Map prediction to products.
recommendations = coa_values.loc[y_hat.index]
print('Recommendations:', recommendations['product_name'])

# Concat historic user average with recommendations.
observations = pd.concat([x_hat, recommendations])

# Look at the terpene ratio of recommended products in comparison to
# the user's historically consumed ratio, weighted by sentiment.
x, y = 'd_limonene', 'beta_pinene'
ax = sns.scatterplot(
    data=observations,
    x=x,
    y=y,
    s=400,
    palette='viridis_r',
)
for index, row in observations.iterrows():
    name = row['user']
    try:
        if math.isnan(name): name = row['product_name']
    except TypeError:
        pass
    print(name)
    ax.text(
        row[x],
        row[y],
        name,
        horizontalalignment='center',
        verticalalignment='bottom',
        size='medium',
        color='#2A8936',
    )
plt.xlim(0)
plt.ylim(0)
plt.title('beta-Pinene / d-Limonene Ratio of Recommendations to Historic Average')
plt.show()


#-----------------------------------------------------------------------
# Bonus: k-nearest neighbors Search with user-specified strain.
#-----------------------------------------------------------------------

# E.g. My favorite strain is Jack Herer, the average Jack Herer has
# chemical profile Xj. Therefore, the most similar raw Garden product
# based on factors B, is Yj.


# E.g. Mikki's favorite strain is Northern Lights. The average Northern Lights
# has chemical profile Xn. Therefore, the most similar raw Garden product
# based on factors B, is Yn.


#-----------------------------------------------------------------------
# Super Bonus: If it were possible to mix products, then what mix
# of products would match, or be the closest match, to a consumer's
# historic or desired chemical profile.
#-----------------------------------------------------------------------

# def f(alpha, x, target):
#     matrix = x[0] * alpha + x[1] * (1- alpha) - target
#     return sum(matrix)

# def gradient_descent(gradient, start, learn_rate, n_iter, **kwargs):
#     vector = start
#     for _ in range(n_iter):
#         diff = -learn_rate * gradient(vector, **kwargs)
#         vector += diff
#     return vector

# # Get a user's target.
# strain = 'Jack Herer' # vs. Northern Lights
# x_hat = pd.DataFrame([strains.loc[strain]])

# # Predict an appropriate blend of two products.
# alpha = 0.5
# product1 = y_hat.iloc[0]
# product2 = y_hat.iloc[1]
# optimal_mix = gradient_descent(
#     gradient=f,
#     start=alpha,
#     learn_rate=0.01,
#     n_iter=100,
#     x=[product1, product2, alpha],
#     target=x_hat,
# )
