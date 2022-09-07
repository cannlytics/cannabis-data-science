"""
CoA Parsing for Consumer Product Recommendation
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 8/23/2022
Updated: 8/30/2022
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    Parse a producer's corpus of CoAs, create a standardized datafile, then
    use the data, augmented with data about consumer's prior purchases, to
    create product recommendations for each consumer.

Data Source:

    - Raw Garden Lab Results
    URL: <https://rawgarden.farm/lab-results/>

"""
# Standard imports.
import os

# External imports.
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

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
DATA_DIR = '../../.datasets/lab_results/raw_garden'
PDF_DIR = '../../.datasets/lab_results/raw_garden/pdfs'

# Create directories if they don't already exist.
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(PDF_DIR):
    os.makedirs(PDF_DIR)


#-----------------------------------------------------------------------
# Get the data!
# URL: <https://rawgarden.farm/lab-results/>
#-----------------------------------------------------------------------

# Get Raw Garden's lab results page.
base = 'https://rawgarden.farm/lab-results/'
response = requests.get(base, headers=DEFAULT_HEADERS)
soup = BeautifulSoup(response.content, 'html.parser')

# Get all of the PDF URLs.
urls = []
for i, link in enumerate(soup.findAll('a')):
    try:
        href = link.get('href')
        if href.endswith('.pdf'):
            urls.append(href)
    except AttributeError:
        continue

# Download all of the PDFs.
for url in urls[1130:]:
    name = url.split('/')[-1]
    outfile = os.path.join(PDF_DIR, name)
    response = requests.get(url, headers=DEFAULT_HEADERS)
    with open(outfile, 'wb') as pdf:
        pdf.write(response.content)

# Count the number of PDFs downloaded.
print(len([x for x in os.listdir(PDF_DIR)]))


#-----------------------------------------------------------------------
# Parse and standardize the data with CoADoc
#-----------------------------------------------------------------------

# Parse lab results with CoADoc.
# parser = CoADoc()
# data = parser.parse(PDF_DIR)


#-----------------------------------------------------------------------
# Analyze the data.
#-----------------------------------------------------------------------

# Look at beta-caryophyllene to humulene ratio.

# Look at beta-pinene to d-limonene ratio.

# Read in review data (augment).

# Calculate the historic average profile per consumer.

# Recommend the nearest product in similarity of terpene ratios.


