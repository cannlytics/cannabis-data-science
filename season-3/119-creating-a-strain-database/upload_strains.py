"""
Upload Cannabis Strains Data
Copyright (c) 2023 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 5/26/2023
Updated: 7/12/2023
License: <https://github.com/cannlytics/cannlytics/blob/main/LICENSE>

Command-line Usage:

    python data/archive/upload_strains.py all

"""
# Standard imports:
from datetime import datetime
import glob
import os
from typing import List

# External imports:
from cannlytics import firebase
from cannlytics.data import create_hash
from datasets import load_dataset
from dotenv import dotenv_values
import pandas as pd


# Specify where your data lives.
DATA_DIR = '../../.datasets/lab_results/2023'


#----------------------------------------------------------------------#
# Compile lab results to obtain strain data.
#----------------------------------------------------------------------#

# Identify all of the datafiles.
datafiles = os.listdir(DATA_DIR)

# Read all lab results.
results = []
for datafile in datafiles:
    full_path = os.path.join(DATA_DIR, datafile)
    df = pd.read_excel(full_path)
    results.append(df)

# Aggregate all lab results.
results = pd.concat(results, ignore_index=True)
print('Aggregated %i lab results.' % len(results))

# TODO: Standardize the data!

# TODO: First, get all known strain names.

# TODO: Second, get all products that contain those strains.

# TODO: Third, calculate summary statistics for each strain.
strain_data = pd.DataFrame()


#----------------------------------------------------------------------#
# Begin to identify unique strain names.
#----------------------------------------------------------------------#

# TODO: Identify all unique known strains in the lab results.
# - Washington
# - California


#----------------------------------------------------------------------#
# Use NLP to help identify unique strain names.
#----------------------------------------------------------------------#

# FIXME:
# import spacy
# from textacy.extract import ngrams

# Identify all strain and product names.


# TODO: Identify common unigrams, bigrams, and trigram from product names
# of lab results that do not have strain names.
# - Connecticut
# - Florida
# - Massachusetts
# - Michigan

# Create natural language processing client.
# Use `en_core_web_sm` for speed and `en_core_web_lg` or `en_core_web_trf`
# for accuracy. For a blank model, use spacy.blank('en')
# Compile all of the product names into a single corpus.
# Handle strange characters, for example replace "_" with " ".
# Convert the corpus to a SpaCy document.
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


#----------------------------------------------------------------------#
# Use AI to help identify unique strain names.
#----------------------------------------------------------------------#

# Chunk by chunk:

# TODO: Use OpenAI GPT-4 model to predict a dictionary of `other_names`
# for strains with similar spellings.

# Save the `other_names` dictionary to a JSON file.

# TODO: Use OpenAI GPT-4 model to remove strain names that do not appear
# to be strain names.

# Save the `removed_names` dictionary to a JSON file.



#----------------------------------------------------------------------#
# Aggregate strain data.
#----------------------------------------------------------------------#

# TODO: Get all results that contain strain names.
# - California
# - Washington

# TODO: Get all results with product names that contain those strains.
# - Connecticut
# - Florida
# - Massachusetts
# - Michigan

# TODO: From WA traceability data, estimate:
# - avg. yield per plant
# - avg. growing time
# - avg. price per gram


#----------------------------------------------------------------------#
# Calculate statistics for each strain.
#----------------------------------------------------------------------#

# # DEV:
# datafiles = get_lab_result_datafiles(DATA_DIR)
# results = []
# for datafile in datafiles:
#     df = pd.read_excel(datafile)
#     results.append(df)

# FIXME: Calculate statistics for each strain.
strain_data = results.groupby('strain_name')
# ✓ strain_id
# - strain_name
# - other_names
# - description (optional)
# - first_observed_at
# - first_observed_county
# - first_observed_state
# - first_observed_zipcode
# - first_observed_producer_license_number
# - first_observed_retailer_license_number
# ✓ keywords
# - lineage
# - patent_number
# - mean_concentrations (thc, cbd, terpenes, etc.)
# - std_concentrations
# - mean_ratios
# - updated_at
# - number_of_lab_results
# - lab_result_ids
# - strain_image_url (create an image?)

# Create a strain ID for each strain.
strain_data['strain_id'] = strain_data['strain_name'].apply(
    create_hash,
    private_key='',
)

# Get strain keywords.
strain_data['keywords'] = strain_data['strain_name'].apply(
    lambda x: str(x).lower().split()
)


def upload_strains(collection: str = 'public/data/strains') -> list:
    """Upload strain data to Firestore."""

    # FIXME: Compile the strain data.
    # strain_data = compile_strain_data()
    strain_data = pd.DataFrame()

    # Initialize Firebase.
    db = firebase.initialize_firebase()

    # Compile the references and documents.
    refs, docs = [], []
    for _, row in strain_data.iterrows():
        doc = row.to_dict()
        _id = str(doc['strain_id'])
        doc['updated_at'] = datetime.now().isoformat()
        ref = f'{collection}/{_id}'
        refs.append(ref)
        docs.append(doc)

    # Upload the data to Firestore.
    firebase.update_documents(refs, docs, database=db)
    return docs


# === Test ===
if __name__ == '__main__':
    
    # Set Firebase credentials.
    try:
        config = dotenv_values('../../.env')
        credentials = config['GOOGLE_APPLICATION_CREDENTIALS']
    except KeyError:
        config = dotenv_values('./.env')
        credentials = config['GOOGLE_APPLICATION_CREDENTIALS']
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials

    # Get any subset specified from the command line.
    import sys
    try:
        subset = sys.argv[1]
        if subset.startswith('--ip'):
            subset = 'all'
    except KeyError:
        subset = 'all'
    
    # Upload Firestore with cannabis license data.
    all_results = upload_strains(subset=subset)
    print('Uploaded strains data to Firestore.')


# === DEV ===

# # Get strains from Firestore.
# strains = firebase.get_collection('public/data/strains')

# # Add keywords to strains.
# refs, updates = [], []
# for strain in strains:
#     # Get the name and convert it to lower case.
#     name = strain['strain_name'].lower()

#     # Tokenize the name into keywords and add the first letter of the name.
#     keywords = name.split() + [name[0]]

#     # Add unique keywords.
#     keywords = list(set(keywords))

#     # Add the document reference and the new keywords to the update lists.
#     doc_id = strain['id']
#     refs.append(f'public/data/strains/{doc_id}')
#     updates.append({'keywords': keywords})

# # Update the strains in Firestore.
# firebase.update_documents(refs, updates)
