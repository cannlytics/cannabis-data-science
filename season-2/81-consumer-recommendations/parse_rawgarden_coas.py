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

    - Strain Reviews
    https://cannlytics.page.link/reported-effects

"""
# Standard imports.
from datetime import datetime
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
DATA_DIR = '.datasets'
COA_DATA_DIR = f'{DATA_DIR}/lab_results/raw_garden'
COA_PDF_DIR = f'{COA_DATA_DIR}/pdfs'
REVIEW_DATA_DIR = f'{DATA_DIR}/effects'

# Create directories if they don't already exist.
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
if not os.path.exists(COA_DATA_DIR): os.makedirs(COA_DATA_DIR)
if not os.path.exists(COA_PDF_DIR): os.makedirs(COA_PDF_DIR)
if not os.path.exists(REVIEW_DATA_DIR): os.makedirs(REVIEW_DATA_DIR)


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
pause = 0.24 # Pause to respect the server serving the PDFs.
total = len(urls)
print('Downloading PDFs, ETA > %.2fs' % (total * pause))
start = datetime.now()
for i, url in enumerate(urls):
    name = url.split('/')[-1]
    outfile = os.path.join(COA_PDF_DIR, name)
    response = requests.get(url, headers=DEFAULT_HEADERS)
    with open(outfile, 'wb') as pdf:
        pdf.write(response.content)
    print('Downloaded %i / %i' % (i +  1, total))
    sleep(pause)
end = datetime.now()

# Count the number of PDFs downloaded.
files = [x for x in os.listdir(COA_PDF_DIR)]
print('Downloaded %i PDFs.' % len(files), 'Time:', end - start)

# Optional: Organize the PDFs into folder by type.


#-----------------------------------------------------------------------
# Parse and standardize the data with CoADoc
#-----------------------------------------------------------------------

# Parse lab results with CoADoc.
parser = CoADoc()

# Iterate over PDF directory.
all_data = []
recorded = []
unidentified = []
for path, subdirs, files in os.walk(COA_PDF_DIR):
    for name in files:

        # Only parse PDFs.
        if not name.endswith('.pdf'):
            continue

        # Parse CoA PDFs one by one.
        file_name = os.path.join(path, name)
        try:
            coa_data = parser.parse(file_name)
            all_data.extend(coa_data)
            print('Parsed:', name)
            recorded.append(name)
        except:
            print('Error:', name)
            unidentified.append(name)
            pass

# Save the CoA data.
timestamp = datetime.now().isoformat()[:19].replace(':', '-')
outfile = f'{COA_DATA_DIR}/rawgarden-coa-data-{timestamp}.xlsx'
coa_data = parser.save(all_data, outfile)

# Read the CoA data back in.
coa_values = pd.read_excel(outfile, sheet_name='Values')
