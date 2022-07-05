"""
Web Scraping | Cannabis Data Science #73 | 2022-07-06
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: July 4th, 2022
Updated: 7/4/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description:

    Archive all of the PSI Labs test results.

Data Sources:

    - PSI Labs Test Results
    URL: <https://results.psilabs.org/test-results/>

Resources:

    - ChromeDriver
    URL: <https://chromedriver.chromium.org/home>

    - Web Scraping using Selenium and Python
    URL: <https://www.scrapingbee.com/blog/selenium-python/>

    - https://xkcd.com/1319/

"""
# Standard imports.
import os

# External imports.
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import abline_plot


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import ElementNotInteractableException


ANALYSES = {
    'potency': 'cannabinoid',
    'terpene': 'terpene',
    'solvent': 'residual_solvent',
    'pesticide': 'pesticide',
    'microbial': 'microbe',
    'metal': 'heavy-metal',
}
BASE = 'https://results.psilabs.org/test-results/?page={}'
PAGES = 4921
print('Expected treasure:', PAGES * 10, 'observations!')


#------------------------------------------------------------------------------
# Setup.
#------------------------------------------------------------------------------

# Specify the full-path to your chromedriver.
# You can also put your chromedriver in `C:\Python39\Scripts`.
DRIVER_PATH = '../assets/tools/chromedriver_win32/chromedriver'

# Open a webpage.
full_driver_path = os.path.abspath(DRIVER_PATH)
service = Service(DRIVER_PATH)
driver = webdriver.Chrome(service=service)
driver.get(BASE.format('4200'))

# Open a webpage, headless.
options = Options()
options.headless = True
options.add_argument('--window-size=1920,1200')
driver = webdriver.Chrome(options=options, service=service)
driver.get(BASE.format('4200'))
print(driver.page_source)
driver.quit()

# Optional: Pull Michigan state limits? Or else use static limits.


#------------------------------------------------------------------------------
# Looking at the data. For each sample, we want:
# - sample_id (generated)
# ✓ product_name
# - product_type
# - producer
# ✓ lab_results_url
# ✓ analyses
# ✓ images
# - coa_urls
# ✓ QR code
# - date_received
# - date_tested
# - method
# ✓ results
#    * units (parsed)
#    ✓ value
#    ✓ margin_of_error
#    ✓ name
#    * key (augmented)
#    * cas (augmented)
#    * limit (augmented)
#------------------------------------------------------------------------------

# Get all the samples on the page.
cards = driver.find_elements(by=By.TAG_NAME, value='sample-card')
print('Found %i sample cards.' % len(cards))

# Begin getting sample details from the card.
card = cards[0]
details = card.find_element(by=By.TAG_NAME, value='md-card-title')

# Get images.
image_elements = details.find_elements(by=By.TAG_NAME, value='img')
images = []
for image in image_elements:
    src = image.get_attribute('src')
    filename = src.split('/')[-1]
    images.append({'url': src, 'filename': filename})

# Get the product name.
product_name = details.find_element(by=By.CLASS_NAME, value='md-title').text

# TODO: Get the producer and date tested.
headers = details.find_elements(by=By.CLASS_NAME, value='md-subhead')
company = headers[0].text
mm, dd, yy = tuple(headers[1].text.split(': ')[-1].split('/'))
date_tested = f'20{yy}-{mm}-{dd}'

# TODO: Get the product type.


# TODO: Get the totals:
# - total_cannabinoids
# - total_thc
# - total_cbd
# - total_terpenes
# - top cannabinoids

# Create a sample ID.
sample_id

# Get the analyses.
analyses = []
container = details.find_element(by=By.CLASS_NAME, value='layout-row')
chips = container.find_elements(by=By.TAG_NAME, value='md-chip')
for chip in chips:
    hidden = chip.get_attribute('aria-hidden')
    if hidden == 'false':
        analyses.append(chip.text)

# Get the lab results URL.
links = card.find_elements(by=By.TAG_NAME, value='a')
lab_results_url = links[0].get_attribute('href')


#------------------------------------------------------------------------------
# Above will be compiled into a DataFrame before getting each sample details.
# Below will be used after all samples are collected to get sample details.
#------------------------------------------------------------------------------

# Get the sample page. Parse all elements below.
lab_results_url = 'https://results.psilabs.org/test-results/show/8MnYiSGCqoEaKNfsD'
driver.get(lab_results_url)

# FIXME: Get CoA URLs by finding all links with with `report_pdf`.
# items = driver.find_elements(by=By.TAG_NAME, value='md-list-item')
# coa_urls = []
# for item in items:
#     link = item.get_attribute('a')
#     if link:
#         break
#     print(href)
#     if 'report_pdfs' in str(href):
#         coa_urls.append({'filename': link.text, 'url': href})

# Get the QR code.
qr_code_el = driver.find_elements(by=By.CLASS_NAME, value='qrcode-link')
qr_code = qr_code_el[0].get_attribute('href')

# Get results for each analysis.
results = []
values = ['name', 'value', 'margin_of_error']
analysis_cards = driver.find_elements(by=By.TAG_NAME, value='ng-include')
for analysis in analysis_cards:
    try:
        analysis.click()
    except ElementNotInteractableException:
        continue
    src = analysis.get_attribute('src')
    rows = analysis.find_elements(by=By.TAG_NAME, value='tr')
    if rows:
        print('Getting results for:', src)
        for row in rows:
            result = {}
            cells = row.find_elements(by=By.TAG_NAME, value='td')
            for i, cell in enumerate(cells):
                key = values[i]
                result[key] = cell.text
            results.append(result)


# TODO: Get date_received


# TODO: Get method


#-----------------------------------------------------------------------
# Getting ALL the Data
#-----------------------------------------------------------------------


def get_psi_labs_test_results():
    """Get all test results for PSI labs."""


    raise NotImplementedError


def get_psi_labs_test_result_details():
    """Get the test result details for a specific PSI lab result."""


    raise NotImplementedError


# TODO: Get all of the results.
# results = get_psi_labs_test_results()
# details = results.apply(get_psi_labs_test_result_details)


# TODO: Save the results.






#-----------------------------------------------------------------------
# Preprocessing the Data
#-----------------------------------------------------------------------

# TODO: Clean the data:
# - `analyses`
# - `results`


# TODO: Read in the saved results.


# TODO: Augment with key, limit, and CAS.



# TODO: Create a strain_id for each observation.
# TODO: Create a data NFT for the lab results!!!


# TODO: Save the curated data.


#------------------------------------------------------------------------------
# Exploring the data.
#------------------------------------------------------------------------------

# TODO: Count the number of lab results scraped!


# TODO: Count the number of unique data points scraped!


# TODO: Look at cannabinoid concentrations over time.


# TODO: Look at cannabinoid distributions by type.


# TODO: Look at terpene distributions by type!


#-----------------------------------------------------------------------
# Modeling the Data
#-----------------------------------------------------------------------

# TODO: Given a lab result, predict if it's in the Xth percentile.



#-----------------------------------------------------------------------
# Training and Testing the Model?
#-----------------------------------------------------------------------


#-----------------------------------------------------------------------
# Evaluating the Model?
#-----------------------------------------------------------------------
