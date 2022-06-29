"""
Intensity Ranking and Polarity Classification
Personality Analysis with NLP
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 6/20/2022
Updated: 6/28/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description:

    This script attempts to estimate the average intensity
    ranking and polarity classification by user and by
    strain to see if there are any low-hanging insights.

Data Sources:

    - Answers to the IPIP Big Five Factor Markers (2018)
    URL: https://openpsychometrics.org/_rawdata/

    - Strain Reviews
    https://cannlytics.page.link/reported-effects

Resources:

    - Deep Learning-Based Document Modeling for Personality Detection from Text
    IEEE Intelligent Systems (2017)
    Author: N. Majumder and S. Poria and A. Gelbukh and E. Cambria
    License: MIT License <https://github.com/SenticNet/personality-detection/blob/master/LICENSE>
    URL: <https://github.com/SenticNet/personality-detection>

    - https://github.com/desaichirayu/Personality-Attribution-using-Natural-Language-Processing

    - https://github.com/amirmohammadkz/personality_detection

    - https://github.com/D2KLab/twitpersonality/blob/master/training/Train_SVM_models.py

Setup:

    1. pip install cannlytics, keras

    2. Download all data to `../.datasets/effects`

"""
# Standard imports.
import re
from statistics import mean

# External imports.
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy import stats
import seaborn as sns

# Internal imports.
from personality_test import (
    score_personality_test,
    QUESTIONS,
)


#------------------------------------------------------------------------------'
# Setup.
#------------------------------------------------------------------------------'

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})

# Specify where your data lives!
DATA_DIR = '../.datasets/effects'


#------------------------------------------------------------------------------
# Analyze historic personality tests to establish a strong prior.
#------------------------------------------------------------------------------'

# Read the personality tests.
folder = 'IPIP-FFM-data-8Nov2018'
tests = pd.read_csv(
    f'{DATA_DIR}/{folder}/data-final.csv',
    delimiter='\t',
    nrows=100_000,
)

# Map the questions.
question_map = {q['key']: q['id'] for q in QUESTIONS}
tests = pd.DataFrame(tests).rename(columns=question_map)
tests = tests[list(question_map.values())]

# Remove invalid tests.
tests = tests.dropna()

# Score the personality tests.
scores = tests.apply(
    lambda row: score_personality_test(row.to_dict()),
    axis='columns',
    result_type='expand',
)
tests = pd.concat([tests, scores], axis='columns')

# Plot the average distribution of personality types.
scores.hist(bins=100)
# Optional: Add horizontal line at median, mode, and mean.
plt.show()


#------------------------------------------------------------------------------
# Begin to prepare the review data for NLP.
#------------------------------------------------------------------------------

# Read in the reviews.
datafile = f'{DATA_DIR}/strain-reviews-2022-06-15.xlsx'
reviews = pd.read_excel(datafile, index_col=0)

# Remove duplicates.
reviews.drop_duplicates(subset='review', keep=False, inplace=True)

# Determine the unique number of users.
users = reviews.groupby('user', as_index=False)
user_reviews = reviews.groupby('user')['review'].apply(list)

# Count the number of reviews by user.
n_reviews = user_reviews.apply(len)
n_reviews.sort_values(ascending=False, inplace=True)

# Isolate reviewers with more than 8 reviews.
review_requirement = 8
reviewers = user_reviews.loc[
    n_reviews.loc[n_reviews > review_requirement].index
]
reviewers = reviewers.drop('Anonymous')

def clean_essay(pars):
    """Clean a list of paragraphs in an essay."""
    essay = ''.join(pars)
    essay = essay.replace('\n\n', '. ')
    essay = essay.replace('\n\n', ' ')
    essay = essay.replace('...', '. ')
    essay = essay.replace('.. ', '. ')
    essay = essay.replace('. . ', '. ')
    essay = essay.replace('&#39;', "'")
    essay = re.sub(' +', ' ', essay)
    return essay

# Aggregate 5 random reviews for each user into essays.
review_limit = 5
essays = reviewers.apply(lambda x: random.sample(x, review_limit))
essays = essays.apply(clean_essay)

# Count the average number of words per essay.
word_count = essays.apply(len)
(word_count.div(5)).hist(bins=40)
plt.title('Average Review Word Count by User')
plt.show()

#------------------------------------------------------------------------------
# Quick repeat of intensity ranking.
#------------------------------------------------------------------------------
import nltk
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

# Remove stopwords from each essay.
stopwords = nltk.corpus.stopwords.words('english')
words = essays.apply(lambda x: x.lower().split(' '))
words = words.apply(lambda x: [w for w in x if w not in stopwords])

# Create natural language processing (NLP) client.
sia = SIA()

def avg_positivity(text: str) -> bool:
    """Average of all sentence compound scores."""
    scores = [
        sia.polarity_scores(sentence)['compound']
        for sentence in nltk.sent_tokenize(text)
    ]
    try:
        return (mean(scores) + 1) / 2
    except:
        return 0.5

# Rank reviews 0 to 1.
ranking = words.apply(' '.join).apply(lambda x: avg_positivity(x))

# visualize positivity ranking intensity.
ranking.hist(bins=40)
plt.title('Intensity Ranking by User')
plt.show()

# Predict the percentile of a given review.
x_hat = essays.sample(1).iloc[0]
y_hat = avg_positivity(x_hat)
percentile = stats.percentileofscore(ranking, y_hat)
print("Percentile of intensity for user's reviews:", round(percentile))
if percentile > 50:
    print('Above average intensity.')
elif percentile == 50:
    print('Average intensity.')
else:
    print('Below average intensity.')


#------------------------------------------------------------------------------
# Use intensity ranking to create polarity classification.
#------------------------------------------------------------------------------

def polarity(text: str) -> bool:
    """Create polarity classification from the average of all
    sentence compound scores."""
    scores = [
        sia.polarity_scores(sentence)['compound']
        for sentence in nltk.sent_tokenize(text)
    ]
    try:
        avg = (mean(scores) + 1 ) / 2
    except:
        avg = 0.5
    if avg > 0.5:
        return 'positive'
    elif avg == 0.5:
        return 'neutral'
    else:
        return 'negative'

# Rank reviews 0 to 1.
user_polarity = words.apply(' '.join).apply(lambda x: polarity(x))

# Identify any negative reviews.
negative_experiences = user_polarity.loc[user_polarity == 'negative']
essays.loc[negative_experiences.index]


#------------------------------------------------------------------------------
# Repeat the intensity ranking by strain.
#------------------------------------------------------------------------------

# Get corpus of words for each strain.
strains = reviews.groupby('strain_name', as_index=False)
strain_reviews = reviews.groupby('strain_name')['review'].apply(list)

# Count the number of reviews by user.
n_strain_reviews = strain_reviews.apply(len)
n_strain_reviews.sort_values(ascending=False, inplace=True)

# Isolate reviewers with more than 8 reviews.
review_requirement = 30
strains_sample = strain_reviews.loc[
    strain_reviews.loc[n_strain_reviews > review_requirement].index
]

# Aggregate random reviews for each strain into essays.
review_limit = 20
strain_essays = strains_sample.apply(lambda x: random.sample(x, review_limit))
strain_essays = strain_essays.apply(clean_essay)
# strain_essays = strains_sample.apply(clean_essay) # Alt.

# Count the average number of words per essay.
strain_word_count = strain_essays.apply(len)
(strain_word_count.div(5)).hist(bins=40)
plt.title('Average Review Word Count by Strain')
plt.show()

# Remove stopwords from each essay.
strain_words = strain_essays.apply(lambda x: x.lower().split(' '))
strain_words = strain_words.apply(lambda x: [w for w in x if w not in stopwords])

# Visualize the positivity ranking by strain.
strain_ranking = strain_words.apply(' '.join).apply(lambda x: avg_positivity(x))
strain_ranking.hist(bins=40)
plt.title('Intensity Ranking by Strain')
plt.show()

# User favorite strain.
favorite = strain_ranking.loc[strain_ranking == strain_ranking.max()]
print('User favorite strain:', print(favorite))

#------------------------------------------------------------------------------

# Question: Is there any way to use the 50 questions from the
# personality test in the prediction model?
