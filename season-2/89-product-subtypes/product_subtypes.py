"""
Product Subtypes
Copyright (c) 2022 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 11/9/2022
Updated: 11/9/2022
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Data Sources:

    - Aggregated Raw Garden Cannabis Lab Results
    https://cannlytics.page.link/rawgarden

    - Aggregated Cannabis Lab Test Results
    https://cannlytics.page.link/tests

Setup:

    1. Install all Python dependencies with `pip install spacy`.

    2. Download the datasets to your `DATA_DIR`.

"""

# External imports.
import pandas as pd
import spacy
import textacy


# Specify where your data lives.
DATA_DIR = '../../.datasets/lab_results/'


#------------------------------------------------------------------------------
# Read in all observed lab results.
#------------------------------------------------------------------------------

# Collect all observed cannabis product names.
product_names = []

# Read in Raw Garden lab results.
filename = 'rawgarden-lab-results.xlsx'
data = pd.read_excel(
        f'{DATA_DIR}/{filename}',
        sheet_name='Details',
        usecols=['product_name'],
    )
product_names.extend(data['product_name'].to_list())

# Read in MCR labs, SC Labs, and PSI labs lab results.
filename = 'aggregated-cannabis-test-results-2022-07-15.xlsx'
sheets = [
    'mcr_labs_raw_data',
    'psi_labs_raw_data',
    'sc_labs_raw_data',
]
for sheet in sheets:
    data = pd.read_excel(
        f'{DATA_DIR}/{filename}',
        sheet_name=sheet,
        usecols=['product_name'],
    )
    product_names.extend(data['product_name'].to_list())

# Reduce to unique product names.
print('Total product names:', len(product_names))
print('Unique product name:', len(list(set(product_names))))


#------------------------------------------------------------------------------
# Train a SpaCy model to recognize strain names and product types.
#------------------------------------------------------------------------------

# Compile all of the product names into a single corpus.
corpus = '. '.join([str(x) for x in product_names])

# Handle strange characters, for example replace "_" with " ".
corpus = corpus.replace('_', ' ')
print('Number of words:', len(corpus.split(' ')))

# Create natural language processing client.
# Use `en_core_web_sm` for speed and `en_core_web_lg` or `en_core_web_trf`
# for accuracy. For a blank model, use spacy.blank('en')
nlp = spacy.load('en_core_web_lg')

# Convert the corpus to a SpaCy document.
doc = nlp(corpus)

# Identify all of the nouns.
nouns = list(doc.noun_chunks)
print('Number of nouns:', len(nouns))

# Identify unigrams, bi-grams, trigrams.
unigrams = list(textacy.extract.ngrams(doc, 1, min_freq=100))
bigrams = list(textacy.extract.ngrams(doc, 2, min_freq=100))
trigrams = list(textacy.extract.ngrams(doc, 3, min_freq=100))

# Identify unique n-grams.
unique_unigrams = list(set([x.text for x in unigrams]))
unique_bigrams = list(set([x.text for x in bigrams]))
unique_trigrams = list(set([x.text for x in trigrams]))
print('Unique unigrams:', len(unique_unigrams))
print('Unique bigrams:', len(unique_bigrams))
print('Unique trigrams:', len(unique_trigrams))

# Future work: Recognize crosses, e.g. "G13 x GMO".


#------------------------------------------------------------------------------
# Read in all of the receipt text from last time.
#------------------------------------------------------------------------------

# Read in receipt texts.
# receipt_texts = []
# with open('receipt_texts.txt', 'r') as f:
#   for line in f:
#     receipt_texts.append(int(line.strip()))


#------------------------------------------------------------------------------
# Try to identify strain names and product types in the text.
#------------------------------------------------------------------------------

# TODO: Apply trained SpaCy model to the receipt texts.
