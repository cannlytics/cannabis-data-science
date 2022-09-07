"""
CoA Parsing for Consumer Product Recommendation
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 8/23/2022
Updated: 9/7/2022
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
DATA_DIR = '../../.datasets'
COA_DATA_DIR = f'{DATA_DIR}/lab_results/raw_garden'
COA_PDF_DIR = f'{COA_DATA_DIR}/pdfs'

# Create directories if they don't already exist.
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
if not os.path.exists(COA_DATA_DIR): os.makedirs(COA_DATA_DIR)
if not os.path.exists(COA_PDF_DIR): os.makedirs(COA_PDF_DIR)


#-----------------------------------------------------------------------
# Get the data!
# URL: <https://rawgarden.farm/lab-results/>
#-----------------------------------------------------------------------

# Get Raw Garden's lab results page.
base = 'https://rawgarden.farm/lab-results/'
response = requests.get(base, headers=DEFAULT_HEADERS)
soup = BeautifulSoup(response.content, 'html.parser')

# Get all of the product categories.
# Match `product_subtype` to the `coa_pdf` filename.
subtypes = []
categories = soup.find_all('div', attrs={'class': 'category-content'})
for category in categories:
    subtype = category.find('h3').text
    for i, link in enumerate(category.findAll('a')):
        try:
            href = link.get('href')
            if href.endswith('.pdf'):
                subtypes.append({
                    'coa_pdf': href.split('/')[-1],
                    'lab_results_url': href,
                    'product_subtype': subtype
                })
        except AttributeError:
            continue

# Save `product_subtype` to `coa_pdf` match.
timestamp = datetime.now().isoformat()[:19].replace(':', '-')
outfile = f'{COA_DATA_DIR}/rawgarden-coa-subtypes-{timestamp}.xlsx'
pd.DataFrame(subtypes).to_excel(outfile)

# Get all of the PDF URLs.
urls = []
for i, link in enumerate(soup.findAll('a')):
    try:
        href = link.get('href')
        # FIXME: Check if the PDF is already downloaded.
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
temp_path = f'{COA_DATA_DIR}/tmp'
temp_file = f'{COA_DATA_DIR}/ocr-coa.pdf'
for path, subdirs, files in os.walk(COA_PDF_DIR):
    for i, name in enumerate(reversed(files)):
        file_name = os.path.join(path, name)

        # Only parse PDFs.
        if not name.endswith('.pdf'):
            continue

        # Only parse unidentified CoAs.
        if name in recorded:
            continue

        # Optional: See if we can identify each LIMS.
        # lab = parser.identify_lims(file_name)

        # Parse CoA PDFs one by one.
        try:
            coa_data  = parser.parse(file_name, temp_path=temp_path)
            all_data.extend(coa_data)
            print('Parsed:', name)
            recorded.append(name)
        except:
            print('Error:', name)
            unidentified.append({'coa_pdf': name})
            pass

        # Save the CoA data every 100 COAs parsed.
        if (i % 100) == 0:
            timestamp = datetime.now().isoformat()[:19].replace(':', '-')
            outfile = f'{COA_DATA_DIR}/rawgarden-coa-data-{timestamp}.xlsx'
            coa_data = parser.save(all_data, outfile)
            outfile = f'{COA_DATA_DIR}/rawgarden-unidentified-coas-{timestamp}.xlsx'
            pd.DataFrame(unidentified).to_excel(outfile)

# Save the unidentified COAs.
timestamp = datetime.now().isoformat()[:19].replace(':', '-')
outfile = f'{COA_DATA_DIR}/rawgarden-unidentified-coas-{timestamp}.xlsx'
pd.DataFrame(unidentified).to_excel(outfile)

# Save the CoA data.
timestamp = datetime.now().isoformat()[:19].replace(':', '-')
outfile = f'{COA_DATA_DIR}/rawgarden-coa-data-{timestamp}.xlsx'
coa_data = parser.save(all_data, outfile)


#-----------------------------------------------------------------------
# Look at the data.
#-----------------------------------------------------------------------

# Read the CoA data back in.
# coa_values = pd.read_excel(outfile, sheet_name='Values')

# TODO: Look at the terpene ratios.

