"""
Personality Analysis with NLP
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 6/20/2022
Updated: 6/22/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description:

    Given a user's review, attempt to determine their personality
    and then use the predicted user personalities to see if there is
    any correlation with beta-pinene to D-limonene ratio.

Data Sources:

    - Curated Cannabis Strains, their Average Chemical Compositions, and
    Reported Effects and Aromas
    URL: https://cannlytics.page.link/reported-effects
    License: CC BY 4.0. <https://creativecommons.org/licenses/by/4.0/>

    - Data from: Over eight hundred cannabis strains characterized
    by the relationship between their subjective effects, perceptual
    profiles, and chemical compositions
    URL: <https://data.mendeley.com/datasets/6zwcgrttkp/1>
    License: CC BY 4.0. <https://creativecommons.org/licenses/by/4.0/>

"""
# External imports.
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statistics import mean

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})

#-----------------------------------------------------------------------
# Preparing the data.
#-----------------------------------------------------------------------

# Read in the reviews.
datafile = '../.datasets/subjective-effects/strain-reviews-2022-06-15.xlsx'
reviews = pd.read_excel(datafile, index_col=0)

# Remove duplicates.
reviews.drop_duplicates(subset='review', keep=False, inplace=True)

# Determine the unique number of users.
users = (reviews['user'].unique())

# Look at all the reviews by a specific user.
reviews.loc[reviews['user'] == 'GreenGreenWA']['review']

# Count the number of reviews by user.
n_reviews = reviews.groupby('user')['review'].nunique()
n_reviews.sort_values(ascending=False, inplace=True)

# Who left the most number of reviews?
n_reviews.loc[n_reviews == n_reviews.max()]

#-----------------------------------------------------------------------

import nltk
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

# Get all reviews for a given user (user-specific corpus).
corpus = reviews.loc[reviews['user'] == 'Anonymous']['review'].sum()

# Create corpus of words from reviews.
words = [w.lower() for w in corpus.split() if w.isalpha()]

# Remove stopwords.
stopwords = nltk.corpus.stopwords.words('english')
words = [w for w in words if w not in stopwords]

# Create a frequency distribution, normalizing all words to lowercase.
fd = nltk.FreqDist(words)
fd.tabulate(10)

# Create natural language processing (NLP) client.
sia = SIA()

def avg_positivity(text: str) -> bool:
    """Average of all sentence compound scores."""
    scores = [
        sia.polarity_scores(sentence)['compound']
        for sentence in nltk.sent_tokenize(text)
    ]
    try:
        return mean(scores)
    except:
        return 0

# Rank reviews -1 to 1.
reviews['ranking'] = reviews['review'].apply(lambda x: avg_positivity(x))


#-----------------------------------------------------------------------
# Saturday Morning Statistics Teaser:
# TODO: Predict the big 5 personality traits for each user.
#-----------------------------------------------------------------------


#-----------------------------------------------------------------------
# Question: Do certain personalities gravitate towards purchasing certain strains?
#-----------------------------------------------------------------------

# Isolate a random sample.
sample = reviews.loc[
    (reviews['user_positivity'] != 0) &
    (reviews['beta_pinene'] != 0) &
    (reviews['d_limonene'] != 0)
]
sample = sample.sample(4200, random_state=420)

# Build a model of a user's average beta-pinene to D-limonene ratio
# on their personality type.
averages = sample.groupby('user').mean()
r_i = averages['beta_pinene'].div(averages['d_limonene'])
user_positivity = averages['ranking']

# Fit the model.
# X = sample[['O', 'C', 'E', 'A', 'N']] # Saturday morning statistics!
X = user_positivity
X = sm.add_constant(X)
model = sm.OLS(r_i, X).fit()
print(model.summary())

# TODO: Test the model: Take 10,000 random samples of 10% to 90% of the
# total number of reviews and plot the distribution of estimated parameters.
