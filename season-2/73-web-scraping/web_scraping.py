"""
Web Scraping | Cannabis Data Science #73 | 2022-07-06
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: July 4th, 2022
Updated: 7/6/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    Archive all of the PSI Labs test results.

Data Sources:

    - PSI Labs Test Results
    URL: <https://results.psilabs.org/test-results/>

Resources:

    - ChromeDriver
    URL: <https://chromedriver.chromium.org/home>

    - Automation Cartoon
    URL: https://xkcd.com/1319/

    - Efficiency Cartoon
    URL: https://xkcd.com/1445/

    - SHA in Python
    URL: https://www.geeksforgeeks.org/sha-in-python/

    - Split / Explode a column of dictionaries into separate columns with pandas
    URL: https://stackoverflow.com/questions/38231591/split-explode-a-column-of-dictionaries-into-separate-columns-with-pandas

    - Tidyverse: Wide and Long Data Tables
    URL: https://rstudio-education.github.io/tidyverse-cookbook/tidy.html

    - Web Scraping using Selenium and Python
    URL: <https://www.scrapingbee.com/blog/selenium-python/>

"""
# Standard imports.
from datetime import datetime
from hashlib import sha256
import hmac
import os
from time import sleep

# External imports.
from cannlytics.utils.utils import snake_case
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import abline_plot

# Selenium imports.
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import ElementNotInteractableException, TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


BASE = 'https://results.psilabs.org/test-results/?page={}'
PAGES = 4921
print('Expected treasure:', PAGES * 10, 'observations!')


#------------------------------------------------------------------------------
# Setup.
#------------------------------------------------------------------------------

# Specify the full-path to your chromedriver.
# You can also put your chromedriver in `C:\Python39\Scripts`.
# DRIVER_PATH = '../assets/tools/chromedriver_win32/chromedriver'
# full_driver_path = os.path.abspath(DRIVER_PATH)
service = Service()

# Open a webpage.
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
# ✓ sample_id (generated)
# ✓ product_name
# ✓ product_type
# ✓ producer
# ✓ lab_results_url
# ✓ analyses
# ✓ images
# ✓ coa_urls
# ✓ QR code
# ✓ date_received
# ✓ date_tested
# ✓ method
# - totals
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

# Get the producer, date tested, and product type.
headers = details.find_elements(by=By.CLASS_NAME, value='md-subhead')
producer = headers[0].text
mm, dd, yy = tuple(headers[1].text.split(': ')[-1].split('/'))
date_tested = f'20{yy}-{mm}-{dd}'
product_type = headers[2].text.split(' ')[-1]

# Create a sample ID.
private_key = bytes(date_tested, 'UTF-8')
public_key = snake_case(product_name)
salt = snake_case(producer)
sample_id = hmac.new(private_key, (public_key + salt).encode(), sha256).hexdigest()
print('Sample ID:', sample_id)

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
lab_results_url = 'https://results.psilabs.org/test-results/show/wqiGAobDTi8HPvhhu'
driver.get(lab_results_url)

# Wait for elements to load.
delay = 3 # seconds
try:

    # Wait for the QR code to load.
    detect = EC.presence_of_element_located((By.CLASS_NAME, 'qrcode-link'))
    qr_code_link = WebDriverWait(driver, delay).until(detect)
    print('QR code loaded!')

    # Get the QR code.
    qr_code = qr_code_link.get_attribute('href')

    # Get CoA URLs by finding all links with with `analytics-event="PDF View"`.
    actions = driver.find_elements(by=By.TAG_NAME, value='a')
    coa_urls = []
    for action in actions:
        event = action.get_attribute('analytics-event')
        if event == 'PDF View':
            href = action.get_attribute('href')
            coa_urls.append({'filename': action.text, 'url': href})

except TimeoutException:
    print('QR Code not loaded.')

# Get results for each analysis.
results = []
date_received, sample_weight, method = None, None, None
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
            if result:
                results.append(result)

    # Get the last few sample details: method, sample_weight, and received_at
    if analysis == 'potency':
        extra = analysis.find_elements(by=By.TAG_NAME, value='md-card-content')
        headings = extra.find_elements(by=By.TAG_NAME, value='h3')
        mm, dd, yy = tuple(headings[0].text.split('/'))
        date_received = f'20{yy}-{mm}-{dd}'
        sample_weight = headings[1].text
        method = headings[-1].text


#-----------------------------------------------------------------------
# Getting ALL the data.
#-----------------------------------------------------------------------

def get_psi_labs_test_results(driver, max_delay=3) -> list:
    """Get all test results for PSI labs.
    Args:
        driver (WebDriver): A Selenium Chrome WebDiver.
        max_delay (float): The maximum number of seconds to wait for rendering.
    Returns:
        (list): A list of dictionaries of sample data.
    """

    # Get all the samples on the page.
    samples = []
    try:
        detect = EC.presence_of_element_located((By.TAG_NAME, 'sample-card'))
        WebDriverWait(driver, max_delay).until(detect)
    except TimeoutException:
        print('Failed to load page within %i seconds.' % max_delay)
        return samples
    cards = driver.find_elements(by=By.TAG_NAME, value='sample-card')
    for card in cards:

        # Begin getting sample details from the card.
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

        # Get the producer, date tested, and product type.
        headers = details.find_elements(by=By.CLASS_NAME, value='md-subhead')
        producer = headers[0].text
        try:
            mm, dd, yy = tuple(headers[1].text.split(': ')[-1].split('/'))
            date_tested = f'20{yy}-{mm}-{dd}'
        except ValueError:
            date_tested = headers[1].text.split(': ')[-1]
        product_type = headers[2].text.split(' ')[-1]

        # Create a sample ID.
        private_key = bytes(date_tested, 'UTF-8')
        public_key = snake_case(product_name)
        salt = snake_case(producer)
        sample_id = hmac.new(private_key, (public_key + salt).encode(), sha256).hexdigest()

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

        # Aggregate sample data.
        sample = {
            'analyses': analyses,
            'date_tested': date_tested,
            'images': images,
            'lab_results_url': lab_results_url,
            'producer': producer,
            'product_name': product_name,
            'product_type': product_type,
            'sample_id': sample_id,
        }
        samples.append(sample)

    return samples


def get_psi_labs_test_result_details(driver, max_delay=3) -> dict:
    """Get the test result details for a specific PSI lab result.
    Args:
        driver (WebDriver): A Selenium Chrome WebDiver.
        max_delay (float): The maximum number of seconds to wait for rendering.
    Returns:
        (dict): A dictionary of sample details.
    """

    # Wait for elements to load, after a maximum delay of X seconds.
    qr_code, coa_urls = None, []
    try:

        # Wait for the QR code to load.
        detect = EC.presence_of_element_located((By.CLASS_NAME, 'qrcode-link'))
        qr_code_link = WebDriverWait(driver, max_delay).until(detect)

        # Get the QR code.
        qr_code = qr_code_link.get_attribute('href')

        # Get CoA URLs by finding all links with with `analytics-event="PDF View"`.
        actions = driver.find_elements(by=By.TAG_NAME, value='a')
        coa_urls = []
        for action in actions:
            event = action.get_attribute('analytics-event')
            if event == 'PDF View':
                href = action.get_attribute('href')
                coa_urls.append({'filename': action.text, 'url': href})

    except TimeoutException:
        print('QR Code not loaded within %i seconds.' % max_delay)

    # Get results for each analysis.
    results = []
    date_received, sample_weight, method = None, None, None
    values = ['name', 'value', 'margin_of_error']
    analysis_cards = driver.find_elements(by=By.TAG_NAME, value='ng-include')
    for analysis in analysis_cards:
        try:
            analysis.click()
        except ElementNotInteractableException:
            continue
        rows = analysis.find_elements(by=By.TAG_NAME, value='tr')
        if rows:
            for row in rows:
                result = {}
                cells = row.find_elements(by=By.TAG_NAME, value='td')
                for i, cell in enumerate(cells):
                    key = values[i]
                    result[key] = cell.text
                if result:
                    results.append(result)

        # Get the last few sample details: method, sample_weight, and received_at
        if analysis == 'potency':
            extra = analysis.find_element(by=By.TAG_NAME, value='md-card-content')
            headings = extra.find_elements(by=By.TAG_NAME, value='h3')
            mm, dd, yy = tuple(headings[0].text.split('/'))
            date_received = f'20{yy}-{mm}-{dd}'
            sample_weight = headings[1].text
            method = headings[-1].text

    # Aggregate sample details.
    details = {
        'coa_urls': coa_urls,
        'date_received': date_received,
        'method': method,
        'qr_code': qr_code,
        'results': results,
        'sample_weight': sample_weight,
    }
    return details


def get_all_psi_labs_test_results(pages, pause=0.125, verbose=True):
    """Get ALL of PSI Labs test results.
    Args:
        pages (iterable): A range of pages to get lab results from.
        pause (float): A pause between requests to respect PSI Labs' server.
        verbose (bool): Whether or not to print out progress, True by default (optional).
    Returns:
        (list): A list of collected lab results.
    """

    # Create a headless Chrome browser.
    options = Options()
    options.headless = True
    options.add_argument('--window-size=1920,1200')
    driver = webdriver.Chrome(options=options, service=service)

    # Iterate over all of the pages to get all of the samples.
    test_results = []
    for page in pages:
        if verbose:
            print('Getting samples on page:', page)
        driver.get(BASE.format(str(page)))
        results = get_psi_labs_test_results(driver)
        if results:
            test_results += results
        else:
            print('Failed to find samples on page:', page)
        sleep(pause)

    # Get the details for each sample.
    for i, test_result in enumerate(test_results):
        if verbose:
            print('Getting details for:', test_result['product_name'])
        driver.get(test_result['lab_results_url'])
        details = get_psi_labs_test_result_details(driver)
        test_results[i] = {**test_result, **details}
        sleep(pause)

    # Close the browser and return the results.
    driver.quit()
    return test_results


# Get all of the results.
pages = range(1, 11) # PAGES + 1
pause = 0.125
runtime = round((len(pages) * 3 + (10 * len(pages) * 3)) / 60, 2)
print('Collecting results. Max runtime >', runtime, 'minutes.')
start = datetime.now()
all_test_results = get_all_psi_labs_test_results(pages, pause=pause)
data = pd.DataFrame(all_test_results)
end = datetime.now()
print('Runtime took:', end - start)

# Save the results.
timestamp = datetime.now().isoformat()[:19].replace(':', '-')
datafile = f'../../.datasets/michigan/psi-lab-results-{timestamp}.xlsx'
data.to_excel(datafile, index=False)


#-----------------------------------------------------------------------
# Preprocessing the Data
#-----------------------------------------------------------------------

ANALYSES = {
    'cannabinoids': ['potency', 'POT'],
    'terpenes': ['terpene', 'TERP'],
    'residual_solvents': ['solvent', 'RST'],
    'pesticides': ['pesticide', 'PEST'],
    'microbes': ['microbial', 'MICRO'],
    'heavy_metals': ['metal', 'MET'],
}
ANALYTES = {
    # TODO: Define all of the known analytes from the Cannlytics library.
}
DECODINGS = {
    '<LOQ': 0,
    '<LOD': 0,
}

# Read in the saved results.
datafile = f'../../.datasets/michigan/psi-lab-results-sample-2022-07-06.xlsx'
data = pd.read_excel(datafile)

# Optional: Drop rows with no analyses at this point.

# Create both wide and long data for ease of use.
# See: https://rstudio-education.github.io/tidyverse-cookbook/tidy.html
# TODO: Normalize and clean the data. In particular, flatten:
# - `analyses`
# - `results`
# - `images`
# - `coa_urls`
wide_data = pd.DataFrame()
long_data = pd.DataFrame()
for index, row in data.iterrows():
    series = row.copy()
    analyses = series['analyses']
    images = series['images']
    results = series['results']
    series.drop(['analyses', 'images', 'results'], inplace=True)
    if not analyses:
        continue

    # TODO: Iterate over results, cleaning results and adding columns.
    # Future work: Augment results with key, limit, and CAS.


# Optional: Create data / CoA NFTs for the lab results!!!


# TODO: Save the curated data, both wide and long data.



#------------------------------------------------------------------------------
# Exploring the data.
#------------------------------------------------------------------------------

# TODO: Count the number of lab results scraped!


# TODO: Count the number of unique data points scraped!


# TODO: Look at cannabinoid concentrations over time.


# TODO: Look at cannabinoid distributions by type.


# TODO: Look at terpene distributions by type!


#-----------------------------------------------------------------------
# Modeling the data.
#-----------------------------------------------------------------------

# TODO: Given a lab result, predict if it's in the Xth percentile.


# TODO: Use in ARIMA model to approach the question:
# Are terpene or cannabinoid concentrations increasing over time by sample type?
# - total_terpenes
# - D-limonene
# - beta-pinene
# - myrcene
# - caryophyllene
# - linalool
# - cbg
# - thcv
# - total_thc
# - total_cbd
# - total_cannabinoids


#-----------------------------------------------------------------------
# Training and testing the model.
#-----------------------------------------------------------------------

# TODO: Separate results after 2020 as test data.


# TODO: Estimate a large number of ARIMA models on the training data,
# use the models to predict the test data, and measure the accuracies.


# TODO: Pick the model that predicts the test data the best.


#-----------------------------------------------------------------------
# Evaluating the model.
#-----------------------------------------------------------------------

# TODO: Re-estimate the model with the entire dataset.


# TODO: Predict if cannabinoid and terpene concentrations are trending
# up or down and to what degree if so.


# TODO: Take away an insight: Is there statistical evidence that
# cannabis cultivated in Michigan is successfully being bred to, on average,
# have higher levels of cannabinoids or terpenes? If so, which compounds?


# TODO: Forecast: If the trend continues, what would cannabis look like
# in 10 years? What average cannabinoid and terpene concentration can
# we expect to see in Michigan in 2025 and 2030?
