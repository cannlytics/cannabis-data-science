"""
Sentiment Analysis | Cannabis Data Science
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 6/14/2022
Updated: 6/15/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    Rank the reviews of customers and then use lab results to see if certain
    compounds may be positively or negatively related with the review rankings.

Setup:

    1. pip install nltk

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

Resources:

    - Over eight hundred cannabis strains characterized by the
    relationship between their psychoactive effects, perceptual
    profiles, and chemical compositions
    URL: <https://www.biorxiv.org/content/10.1101/759696v1.abstract>

    - SkunkFx
    URL: <www.skunkfx.com>

    - Notes on Microeconomics
    http://keeganskeate.com/pdfs/microeconomics.pdf

"""
# External imports.
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})


#------------------------------------------------------------------------------
# Preparing the data.
#------------------------------------------------------------------------------

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

# Isolate a training sample.
train = reviews.loc[reviews['user'] != 'Anonymous']
train = train.loc[
    (train['total_thc'] > 0) &
    (train['total_cbd'] > 0) &
    (train['beta_pinene'] > 0) &
    (train['d_limonene'] > 0) &
    (train['beta_caryophyllene'] > 0) &
    (train['humulene'] > 0)
]
train = train.sample(4200, random_state=420)


#------------------------------------------------------------------------------
# Visualize the data.
#------------------------------------------------------------------------------

# Plot beta-pinene to d-limonene by user.
sample = train.groupby('user').mean()
sample['n_reviews'] = n_reviews.loc[sample.index]
sample['heavy_reviewer'] = 0
sample.loc[sample['n_reviews'] > 8, 'heavy_reviewer'] = 1
sns.scatterplot(
    x='d_limonene',
    y='beta_pinene',
    data=sample,
    hue='heavy_reviewer',
)
plt.title('Average Terpene Concentrations of Strains Reviewed by User')
plt.show()


#------------------------------------------------------------------------------
# Sentiment Analysis
#------------------------------------------------------------------------------

# Create natural language processing (NLP) client.
sia = SIA()

# Create a rank for a review.
line = reviews['review'].sample(1, random_state=420).iloc[0]
pol_score = sia.polarity_scores(line)
rank = pol_score['compound']
print('Review:', line)
print('Ranking:', rank)


def get_ranking(x, client):
    """Assign a ranking to a review using natural language processing."""
    score = client.polarity_scores(x)
    return score['compound']


# Rank reviews -1 to 1.
ranking = train['review'].apply(lambda x: get_ranking(x, sia))


#------------------------------------------------------------------------------
# Create utility functions with THC, CBD, and terpenes as preferences.
#------------------------------------------------------------------------------

# See if THC:CBD ratio, beta-pinene to d-limonene ratio, or
# beta-caryophyllene to humulene ratio have a statistical relationship
# with the ranking of a review.
Y = ranking
X = train[[
    'total_thc',
    'total_cbd',
    'beta_pinene',
    'd_limonene',
    'beta_caryophyllene',
    'humulene',
]].apply(np.log)
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
print(model.summary())


#------------------------------------------------------------------------------
# Estimate an expected utility function.
# Saturday Morning Statistics Teaser:We will attempt to estimate individual-specific effects.
#------------------------------------------------------------------------------

# Create log terms.
variates = [
    'total_thc',
    'total_cbd',
    'beta_pinene',
    'd_limonene',
    'beta_caryophyllene',
    'humulene',
]
for variate in variates:
    train[f'log_{variate}'] = np.log(train[variate])
    train[f'int_log_{variate}'] = np.log(train[variate]) * train['effect_sleepy']

# Add interaction term with effects, e.g. sleepy, to get expected utility function parameters.
interactions = train[[
    'log_total_thc',
    'log_total_cbd',
    'log_beta_pinene',
    'log_d_limonene',
    'log_beta_caryophyllene',
    'log_humulene',
    'int_log_total_thc',
    'int_log_total_cbd',
    'int_log_beta_pinene',
    'int_log_d_limonene',
    'int_log_beta_caryophyllene',
    'int_log_humulene',
]]
interactions = sm.add_constant(interactions)
model = sm.OLS(Y, interactions).fit()
print(model.summary())
