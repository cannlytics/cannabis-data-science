"""
Create Strain NFT Art
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 5/29/2022
Updated: 8/10/2022
License: <https://github.com/cannlytics/cannlytics/blob/main/LICENSE>

Description:

    Create 420 NFT art pieces for the top 420 cannabis strains.

Data Source:

    - Aggregated Cannabis Laboratory Test Results
    URL: <https://huggingface.co/datasets/cannlytics/aggregated-cannabis-test-results>

References:

    - Turn Photos into Cartoons Using Python
    URL: <https://towardsdatascience.com/turn-photos-into-cartoons-using-python-bb1a9f578a7e>

"""
# Standard imports.
from ast import literal_eval
import os
import shutil

# External imports.
import matplotlib.pyplot as plt
import pandas as pd
import requests

# Internal imports.
from cannlytics.data.data import create_sample_id
from cannlytics.utils import kebab_case
from flower_art import FlowerArt


# Specify where your data lives.
DATA_DIR = '../../.datasets/lab_results'
IMAGE_DIR = '../../.datasets/strains/images'

# Create directories if they don't already exist.
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


#-------------------------------------------
# NFT Art with PSI Labs Images.
#-------------------------------------------

# Read in the lab results data.
filename = f'{DATA_DIR}/aggregated-cannabis-test-results-2022-08-09.xlsx'
data = pd.read_excel(filename, sheet_name='psi_labs_raw_data')

# Identify flower samples.
flower = data.loc[data['product_type'] == 'Flower']

# Look at the most frequent product name.
top_product = flower['product_name'].value_counts().idxmax()
print('Top product:', top_product)

# Get the top 10 most frequent product names.
top_products = flower['product_name'].value_counts()[:10]
for i, product in enumerate(top_products.index.tolist()):
    print('%i. %s' % (i + 1, product))

# Visualize the top products
top_products.sort_values().plot(kind="barh")
plt.show()

# Get a random image for the strain.
strain_name = 'Death Star'
criterion = (flower['product_name'] == strain_name)
sample = flower.loc[criterion].sample(1, random_state=420).iloc[0]
images = literal_eval(sample['images'])
image_url = images[0]['url']

# Download the model image.
image_file = f'{IMAGE_DIR}/model.jpg'
response = requests.get(image_url, stream=True)
if response.status_code == 200:
    image_loaded = True
    with open(image_file, 'wb') as f:
        response.raw.decode_content = True
        shutil.copyfileobj(response.raw, f)

# Create an art AI client.
# Note: Incorporate randomness here!
art = FlowerArt(
    line_size = 7,
    blur_value = 7,
    number_of_filters = 10, # 5 (fast) to 10 (slow) filters recommended.
    total_colors = 50,
    sigmaColor = 50,
    sigmaSpace = 50,
)

# Create an NFT image for the strain.
key = kebab_case(strain_name)
nft_file = f'{IMAGE_DIR}/nft-{key}.jpg'
nft_image_data = art.cartoonize_image(image_file, nft_file)


#-----------------------------------------------------------------------
# NFT Art with SC Labs Images.
#-----------------------------------------------------------------------

# Read in the lab results data.
filename = f'{DATA_DIR}/aggregated-cannabis-test-results-2022-08-09.xlsx'
data = pd.read_excel(filename, sheet_name='sc_labs_raw_data')

# Identify flower samples.
flower = data.loc[data['product_type'] == 'Flower, Inhalable']

# Look at the most frequent product name.
flower['product_name'].value_counts().idxmax()

# Get the top 10 most frequent product names.
top_products = flower['product_name'].value_counts()[:10]
for i, product in enumerate(top_products.index.tolist()):
    print('%i. %s' % (i + 1, product))

# Visualize the top products
top_products.sort_values().plot(kind="barh")
plt.show()


#-----------------------------------------------------------------------
# NFT Art with MCR Labs Images.
#-----------------------------------------------------------------------

# Read lab results data.
filename = f'{DATA_DIR}/aggregated-cannabis-test-results-2022-08-09.xlsx'
data = pd.read_excel(filename, sheet_name='mcr_labs_raw_data')

# Identify flower samples.
data['date_tested'] = pd.to_datetime(data['date_tested'])
flower = data.loc[
    (data['product_type'] == 'flower') &
    (data['date_tested'] >= pd.to_datetime('2022-01-01'))
]

# Look at the most frequent product name.
flower['product_name'].value_counts().idxmax()

# Get the top 10 most frequent product names.
top_products = flower['product_name'].value_counts()[:10]
for i, product in enumerate(top_products.index.tolist()):
    print('%i. %s' % (i + 1, product))

# Visualize the top products
top_products.sort_values().plot(kind="barh")
plt.show()


#-----------------------------------------------------------------------
# NFT Generation
#-----------------------------------------------------------------------

# Create an art AI client.
art = FlowerArt(
    line_size = 7,
    blur_value = 7,
    number_of_filters = 10, # 5 (fast) to 10 (slow) filters recommended.
    total_colors = 9,
    sigmaColor = 50,
    sigmaSpace = 50,
)

# Mint images for the top products.
nfts = []
for strain_name in top_products[:5]:

    print('Minting:', strain_name)

    # Download a random sample's image.
    image_file = f'{IMAGE_DIR}/model.jpg'

    # Get a random image for the strain.
    criterion = (flower['product_name'] == strain_name)
    sample = flower.loc[criterion].sample(1, random_state=420).iloc[0]
    images = literal_eval(sample['images'])
    image_url = images[-1]['url']
    print('Sample:', sample['sample_id'])
    print('Sample URL:', sample['lab_results_url'])
    print('Sample Image URL:', image_url)

    # Download the model image.
    response = requests.get(image_url)
    open(image_file, 'wb').write(response.content)

    # Create a cartoon modelled after the image.
    key = kebab_case(strain_name)
    nft_file = f'{IMAGE_DIR}/nft-{key}.jpg'
    nft_image_data = art.cartoonize_image(image_file, nft_file)

    # Future work: Mint an NFT for the image!
    nfts.append({
        'file_name': nft_file,
        'id': create_sample_id('cannlytics.eth', nft_image_data),
        'image_url': image_url,
        'lab_results_url': sample['lab_results_url'],
        'name': strain_name,
        'sample_id': sample['sample_id'],
    })


#-----------------------------------------------------------------------
# Future work: Create 420 Strain NFTs for the top 420 strains.
#-----------------------------------------------------------------------


# 1. Stack all SC Labs, MCR Labs, and PSI Labs lab results.
lab_results = pd.DataFrame()
sheets = ['mcr_labs_raw_data', 'psi_labs_raw_data', 'sc_labs_raw_data']
for sheet in sheets:
    data = pd.read_excel(filename, sheet_name=sheet)
    lab_results = pd.concat([lab_results, data], ignore_index=True, axis=0)

# Hot-fix for Gorilla Glue!!!
lab_results['product_name'] = lab_results['product_name'].str.replace('-- ', '')
lab_results['product_name'] = lab_results['product_name'].replace({
    'Gorilla Glue #4': 'Gorilla Glue',
    'GG': 'Gorilla Glue',
    'GG4': 'Gorilla Glue',
    'GG #4': 'Gorilla Glue',
    'GG#4': 'Gorilla Glue',
    'GG 4': 'Gorilla Glue',
    'Gorilla Glue 4': 'Gorilla Glue',
    'Gorilla Glue #4 - 1': 'Gorilla Glue',
    'GSC': 'Girl Scout Cookies',
    'GDP': 'Grand Daddy Purple',
    'Granddaddy Purple': 'Grand Daddy Purple',
    'Chem Dog': 'Chemdawg',
    'Chemdog': 'Chemdawg',
    'SFV': 'SFV OG',
    'Cannatonic #4': 'Cannatonic',
    'Cannatonic #4 - RP': 'Cannatonic',
})

# Hot-fix to exclude lab digit names.
lab_results = lab_results.loc[
    (lab_results['product_name'].str.len() > 1) &
    (lab_results['product_name'] != '#1') &
    (lab_results['product_name'] != '#2') &
    (lab_results['product_name'] != '#3') &
    (lab_results['product_name'] != '10') &
    (lab_results['product_name'] != '15') &
    (lab_results['product_name'] != '18') &
    (lab_results['product_name'] != '22')
]

# 2. Find the top 420 strains.
count = 420
flower_types = [
    'flower',
    'Flower',
    'Flower, Inhalable',
    'Flower, Product Inhalable',
    'Flower, Medical Inhalable',
    'Flower, Hemp Flower',
    'Flower, Hemp',
    'Indica, Flower, Inhalable',
    'Sativa, Flower, Inhalable',
]
flower = lab_results.loc[lab_results['product_type'].isin(flower_types)]
top_products = flower['product_name'].value_counts()[:count]
for i, product in enumerate(top_products.index.tolist()):
    print('%i. %s' % (i + 1, product))

# Visualize the top products
top_products.sort_values()[-15:].plot(kind="barh")
plt.show()
for index, value in top_products.iteritems():
    print(index, value)

# 3. Mint 420 NFT images!


#-----------------------------------------------------------------------
# Future work: Use NLP to identify the top strains.
# https://medium.com/analytics-vidhya/nlp-tutorial-for-text-classification-in-python-8f19cd17b49e
# https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
# https://towardsdatascience.com/named-entity-recognition-ner-using-spacy-nlp-part-4-28da2ece57c6
#-----------------------------------------------------------------------

# Use NLP to identify common "strains".
import spacy
from spacy import displacy
from collections import Counter
from string import punctuation

# Create a NLP client.
nlp = spacy.load('en_core_web_lg')

# Find the similarity between words.
nlp('Gorilla Glue #4').similarity(nlp('Gorilla Glue'))
nlp('Blue Dream 2').similarity(nlp('Blue Dream'))
nlp('Blue Dream 2').similarity(nlp('Uncle Monkey'))
nlp('Blue Dream').similarity(nlp('Gorilla Glue'))
nlp('Gorilla Glue').similarity(nlp('Gorilla Cookies'))
nlp('Gorilla Glue').similarity(nlp('Uncle Monkey'))
