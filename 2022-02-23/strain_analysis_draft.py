"""
Strain Analysis of Flower in Washington State
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 2/17/2022
Updated: 2/17/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script calculates various statistics about strains from the
Washington State traceability data (2018-01-31 to 11-10-2021).

Notes:

    Install SpaCy:
        
        1. pip install spacy
        2. python -m spacy download en_core_web_sm
           python -m spacy download en_core_web_trf

    Install PyMC3:

        1. pip install pymc3

Data sources:

    - Random sample of sales items
    https://cannlytics.page.link/cds53

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd?page=1
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd?page=2

Data Guide:

    - Washington State Leaf Data Systems Guide
    https://lcb.wa.gov/sites/default/files/publications/Marijuana/traceability/WALeafDataSystems_UserManual_v1.37.5_AddendumC_LicenseeUser.pdf
"""

# External imports.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # pip install basemap
import pandas as pd
import seaborn as sns


# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
})


#--------------------------------------------------------------------------
# Analyze the data.
#--------------------------------------------------------------------------

# Specify where the data lives.
DATA_DIR = 'D:\\leaf-data'
DATA_FILE = f'{DATA_DIR}/samples/random-sales-items-2022-02-16.csv'

# Read in the data.
data = pd.read_csv(DATA_FILE)


#--------------------------------------------------------------------------
# Identify sample weight and unit of measure where possible.
#--------------------------------------------------------------------------

import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Simple example.
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)

# Stream example.
texts = [
    "Net income was $9.4 million compared to the prior year of $2.7 million.",
    "Revenue exceeded twelve billion dollars, with a loss of $1b.",
]
nlp = spacy.load("en_core_web_sm")
for doc in nlp.pipe(texts, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]):
    # Do something with the doc here
    print([(ent.text, ent.label_) for ent in doc.ents])


def identify_quantity(row, key=''):
    """Identify the quantity if present in a given string."""
    return row[key]


def identify_uom(row, key=''):
    """Identify the unit of measure if present in a given string."""
    return row[key]


data['quantity'] = data.apply (lambda row: identify_quantity(row), axis=1)
data['unit_of_measure'] = data.apply (lambda row: identify_uom(row), axis=1)


import spacy
from collections import Counter

# Create natural language processing client.
nlp = spacy.load("en_core_web_trf")


sample = data.sample(20)
for index, values in sample.iterrows():
    doc = nlp(values['product_name'])
    spacy.displacy.render(doc, style='ent')

# Count words, removing stop words and punctuation symbols.
complete_text = ('Gus Proto is a Python developer currently'
'working for a London-based Fintech company. He is'
' interested in learning Natural Language Processing.'
' There is a developer conference happening on 21 July'
' 2019 in London. It is titled "Applications of Natural'
' Language Processing". There is a helpline number '
' available at +1-1234567891. Gus is helping organize it.'
' He keeps organizing local Python meetups and several'
' internal talks at his workplace. Gus is also presenting'
' a talk. The talk will introduce the reader about "Use'
' cases of Natural Language Processing in Fintech".'
' Apart from his work, he is very passionate about music.'
' Gus is learning to play the Piano. He has enrolled '
' himself in the weekend batch of Great Piano Academy.'
' Great Piano Academy is situated in Mayfair or the City'
' of London and has world-class piano instructors.')


complete_doc = nlp(complete_text)
words = [token.text for token in complete_doc
     if not token.is_stop and not token.is_punct]
word_freq = Counter(words)

# Find the 5 most commonly occurring words with their frequencies
common_words = word_freq.most_common(5)
print(common_words)

# Compute Similarity.
token_1=nlp('Blue Cookies OG')
token_2=nlp('Sour OG')
similarity_score=token_1.similarity(token_2)
print(similarity_score)

# Removal of stop words and punctuations
words=[token for token in doc if token.is_stop==False and token.is_punct==False]

# Calculating frequency count
freq_dict={}
for word in words:
  if word not in freq_dict:
    freq_dict[word]=1
  else:
    freq_dict[word]+=1

# words_all = [token.text for token in complete_doc if not token.is_punct]
# word_freq_all = Counter(words_all)
# common_words_all = word_freq_all.most_common(5)
# print (common_words_all)

# Extract Noun Phrases
# sample = data.sample(20)
# for index, values in sample.iterrows():
#     try:
#         doc = nlp(values['product_name'])
#         for entity in doc.ents:
#             print('Text:', entity.text, entity.label_)
#         for chunk in doc.noun_chunks:
#             print(chunk)
#     except ValueError:
#         pass

# # Unique words
# unique_words = [word for (word, freq) in word_freq.items() if freq == 1]
# print(unique_words)


# import spacy
# from spacy.en import English

# parser = English()
# parsedEx = parser(example)
# ents = list(parsedEx.ents)
# tags={}
# for entity in ents:
#     #print(entity.label, entity.label_, ' '.join(t.orth_ for t in entity))
#     term =' '.join(t.orth_ for t in entity)
#     if ' '.join(term) not in tags:
#         tags[term]=[(entity.label, entity.label_)]
#     else:
#         tags[term].append((entity.label, entity.label_))
# print(tags)

# TODO: For each product, try to find qty with spacy.


# TODO: Do custom identification following spacy's attempt.


# TODO: Calculate the percent of observations where qty and uom were able to be identified.



#--------------------------------------------------------------------------
# Identify the most popular strains in Washington State.
#--------------------------------------------------------------------------

# Isolate flower sales.
# flower = data.loc[data['intermediate_type'] == 'usable_marijuana']

# TODO: Find the frequency of the most common strain names.

# e.g.
# g = flower.loc[flower['product_name'].str.contains('gelato', na=False, case=False)]

# TODO: Find the strain names with the highest sales.


# TODO: Find the strain names with the highest prices.



#--------------------------------------------------------------------------
# Identify the most popular edibles in Washington State.
#--------------------------------------------------------------------------

# # Isolate flower sales.
# solid_edibles = data.loc[data['intermediate_type'] == 'solid_edible']
# liquid_edibles = data.loc[data['intermediate_type'] == 'liquid_edible']

# solid_edibles = solid_edibles.loc[solid_edibles['price_total'] > 0]
# solid_edibles = solid_edibles[solid_edibles['price_total'] < solid_edibles['price_total'].quantile(.95)]

# # Drop observations with negative prices and prices in the upper quantile.
# liquid_edibles = liquid_edibles.loc[liquid_edibles['price_total'] > 0]
# liquid_edibles = liquid_edibles[liquid_edibles['price_total'] < liquid_edibles['price_total'].quantile(.95)]


# liquid_edibles.price_total.hist(density=1)
# solid_edibles.price_total.hist(alpha=0.3, density=1)
# plt.show()


# Estimate the market share of the top liquid edible producers.


# Estimate the market share of the top solid edible producers.



#--------------------------------------------------------------------------
# Attempt to identify strains by THC to CBD ratio.
#--------------------------------------------------------------------------

# Calculate total CBD and total THC.
# sample_data['total_cbd'] = 0.87 * sample_data['cannabinoid_cbda_percent'] + sample_data['cannabinoid_cbd_percent']
# sample_data['total_thc'] = 0.87 * sample_data['cannabinoid_d9_thca_percent'] + sample_data['cannabinoid_d9_thc_percent']

# Visualize the THC:CBD ratio.
# fig, ax = plt.subplots(figsize=(15, 7))
# sns.scatterplot(
#     x='total_cbd',
#     y='total_thc',
#     data=sample_data,
#     ax=ax
#     # hue='county',
# )

# Joint plot.
# https://seaborn.pydata.org/generated/seaborn.jointplot.html#seaborn.jointplot
# sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue="species")


# Scatterplot with varying point sizes and hues
# https://seaborn.pydata.org/examples/scatter_bubbles.html
# sns.relplot(
#     x="horsepower",
#     y="mpg",
#     hue="origin",
#     size="weight",
#     sizes=(40, 400),
#     alpha=.5,
#     palette="muted",
#     height=6,
#     data=mpg
# )

