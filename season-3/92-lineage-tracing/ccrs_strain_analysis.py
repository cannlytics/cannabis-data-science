"""
CCRS Strain Analysis
Copyright (c) 2022 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 12/28/2022
Updated: 12/28/2022
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Data Source:

    - WSLCB
    URL: <https://lcb.box.com/s/xseghpsq2t4i1musxj6mgd7b8rhxe7bm>

"""
# Internal imports:
import gc
import os

# External imports:
from cannlytics.utils import rmerge
from cannlytics.data.ccrs.constants import CCRS_DATASETS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import spacy
# import textacy
# from textacy.extract import ngrams


# Specify where your data lives.
DATA_DIR = 'D:\\data\\washington\\ccrs-2022-11-22\\ccrs-2022-11-22\\'


#------------------------------------------------------------------------------
# Data curation.
#------------------------------------------------------------------------------

# # Read strain data.
# fields = CCRS_DATASETS['strains']['fields']
# date_fields = CCRS_DATASETS['strains']['date_fields']
# dtypes = {k: fields[k] for k in fields if k not in date_fields}
# try:
#     date_fields.remove('UpdatedDate')
# except ValueError:
#     pass
# usecols = list(set(list(fields.keys()) + date_fields))
# strains = pd.read_csv(
#     f'{DATA_DIR}/Strains_0/Strains_0/Strains_0.csv',
#     sep='\t',
#     encoding='utf-16',
#     parse_dates=date_fields,
#     usecols=usecols,
#     dtype=dtypes,
#     lineterminator='\n'
# )
# strain_names = list(strains['Name'].unique())

# # Perform garbage cleaning.
# del strains
# gc.collect()


#------------------------------------------------------------------------------
# Analysis: Identify the top-selling strains in November of 2022.
#------------------------------------------------------------------------------

# # Create natural language processing client.
# # Use `en_core_web_sm` for speed and `en_core_web_lg` or `en_core_web_trf`
# # for accuracy. For a blank model, use spacy.blank('en')
# # Compile all of the product names into a single corpus.
# # Handle strange characters, for example replace "_" with " ".
# # Convert the corpus to a SpaCy document.
# corpus = '. '.join([str(x) for x in strain_names])
# corpus = corpus.replace('_', ' ')
# nlp = spacy.load('en_core_web_lg')
# doc = nlp(corpus)

# # Identify unique unigrams, bi-grams, trigrams to use as strain names.
# unigrams = list(set([x.text for x in ngrams(doc, 1, min_freq=1)]))
# bigrams = list(set([x.text for x in ngrams(doc, 2, min_freq=1)]))
# trigrams = list(set([x.text for x in ngrams(doc, 3, min_freq=1)]))
# print('Unique unigrams:', len(unigrams))
# print('Unique bigrams:', len(bigrams))
# print('Unique trigrams:', len(trigrams))

# TODO: Read licensee items for November 2022, licensee by licensee.


#------------------------------------------------------------------------------
# Analysis: Find strains chemically similar to ACDC sold in the Seattle area.
#------------------------------------------------------------------------------

# TODO: Find all lab results for any ACDC strain.


# TODO: Calculate the average THC to CBD ratio of ACDC strains.


# TODO: Identify lab results with similar THC to CBD ratios.


# TODO: Find inventory in Seattle that correspond to lab results similar to ACDC.



#------------------------------------------------------------------------------
# Analysis: Find the top-selling licensees.
#------------------------------------------------------------------------------

# Define analysis criterion.
prefix = 'sales'
suffix = '2022-11.xlsx'

# Define statistics to collect.
licensee_sales = {}

# Read in data for analysis.
licensee_stats = 'D:\\data\\washington\\ccrs_stats\\licensee_stats'
licensees = os.listdir(licensee_stats)
for licensee_id in licensees:
    licensee_dir = os.path.join(licensee_stats, licensee_id)
    files = os.listdir(licensee_dir)
    datafiles = [x for x in files if x.startswith(prefix) and x.endswith(suffix)]
    if not datafiles:
        continue
    datafile = os.path.join(licensee_dir, datafiles[0])
    licensee_data = pd.read_excel(datafile)
    licensee_sales[licensee_id] = licensee_data['UnitPrice'].sum()

# Visualize statistics.
stats = pd.DataFrame({
    'LicenseeId': list(licensee_sales.keys()),
    'sales': list(licensee_sales.values()),
})
stats['LicenseeId'] = stats['LicenseeId'].astype(str)
stats.sort_values('sales', ascending=False, inplace=True)

# Merge licensee data.
licensees = pd.read_csv(
    f'{DATA_DIR}/Licensee_0/Licensee_0/Licensee_0.csv',
    sep='\t',
    encoding='utf-16',
)
licensees['LicenseeId'] = licensees['LicenseeId'].astype(str)
stats = pd.merge(
    stats,
    licensees,
    left_on='LicenseeId',
    right_on='LicenseeId',
)


# Who sold the most in November of 2022?
top_seller = stats.loc[stats['sales'] == stats['sales'].max()]

# Who were the top 10 retailers in November of 2022?
top_sellers = stats[['sales', 'Name','DBA']].head(10)

# How much cannabis was sold in WA in November of 2022?
nov_sales = round(stats['sales'].sum() / 1_000_000, 1)
print(f'${nov_sales} million sold in WA in November of 2022.')

# Visualize the distribution of sales.
stats.loc[stats['sales'] > 10_000]['sales'].hist(bins=100)
plt.show()

# What was the average sales per retailer in November of 2022?
avg_sales = round(stats['sales'].mean())
print(f'${round(avg_sales, 1)} avg. sales per retailer in WA in November of 2022.')
