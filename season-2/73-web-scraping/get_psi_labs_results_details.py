"""
Get PSI Labs Test Results
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: July 4th, 2022
Updated: 7/10/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    Archive all of the PSI Labs test results.

Data Sources:

    - PSI Labs Test Results
    URL: <https://results.psilabs.org/test-results/>

Resources:

    - ChromeDriver
    URL: <https://chromedriver.chromium.org/home>

Setup:

    1. Create a folder `../../.datasets/lab_results/raw_data` to store your data.

    2. Download ChromeDriver and put it in your `C:\Python39\Scripts` folder
    or pass the `executable_path` to the `Service`.

    3. Pick the `PAGES` that you want to collect.
"""
# Standard imports.
from datetime import datetime
from hashlib import sha256
import hmac
import os
from time import sleep

# External imports.
from cannlytics.utils.utils import snake_case
import pandas as pd

# Selenium imports.
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import ElementNotInteractableException, TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


# Desired order for output columns.
COLUMNS = [
    'sample_id',
    'date_tested',
    'analyses',
    'producer',
    'product_name',
    'product_type',
    'results',
    'coa_urls',
    'images',
    'lab_results_url',
    'date_received',
    'method',
    'qr_code',
    'sample_weight',
]



def create_sample_id(private_key, public_key, salt='') -> str:
    """Create a hash to be used as a sample ID.
    The standard is to use:
        1. `private_key = producer`
        2. `public_key = product_name`
        3. `salt = date_tested`
    Args:
        private_key (str): A string to be used as the private key.
        public_key (str): A string to be used as the public key.
        salt (str): A string to be used as the salt, '' by default (optional).
    Returns:
        (str): A sample ID hash.
    """
    secret = bytes(private_key, 'UTF-8')
    message = snake_case(public_key) + snake_case(salt)
    sample_id = hmac.new(secret, message.encode(), sha256).hexdigest()
    return sample_id


#------------------------------------------------------------------------------
# Setup.
#------------------------------------------------------------------------------

# Define pages to collect.
BASE = 'https://results.psilabs.org/test-results/?page={}'
TOTAL_PAGES = 4921
PAGES = range(1, 1025) # Collect N pages. (Get pg 4921!)

# Specify your chromedriver. You can:
# 1. Put your chromedriver in `C:\Python39\Scripts` or equivalent.
# 2. Specify the full path to the driver, but this is troublesome.
# DRIVER_PATH = '../assets/tools/chromedriver_win32/chromedriver'
# full_driver_path = os.path.abspath(DRIVER_PATH)
service = Service()


#-----------------------------------------------------------------------
# Getting ALL the data.
#-----------------------------------------------------------------------

def get_psi_labs_test_results(driver, max_delay=5, reverse=True) -> list:
    """Get all test results for PSI labs.
    Args:
        driver (WebDriver): A Selenium Chrome WebDiver.
        max_delay (float): The maximum number of seconds to wait for rendering (optional).
        reverse (bool): Whether to collect in reverse order, True by default (optional).
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
    if reverse:
        cards.reverse()
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


def get_psi_labs_test_result_details(driver, max_delay=5) -> dict:
    """Get the test result details for a specific PSI lab result.
    Args:
        driver (WebDriver): A Selenium Chrome WebDiver.
        max_delay (float): The maximum number of seconds to wait for rendering.
    Returns:
        (dict): A dictionary of sample details.
    """

    # Deemed optional:
    # Wait for elements to load, after a maximum delay of X seconds.
    qr_code, coa_urls = None, []
    # try:

    #     # Wait for the QR code to load.
    #     detect = EC.presence_of_element_located((By.CLASS_NAME, 'qrcode-link'))
    #     qr_code_link = WebDriverWait(driver, max_delay).until(detect)

    #     # Get the QR code.
    #     qr_code = qr_code_link.get_attribute('href')

    #     # Get CoA URLs by finding all links with with `analytics-event="PDF View"`.
    #     actions = driver.find_elements(by=By.TAG_NAME, value='a')
    #     coa_urls = []
    #     for action in actions:
    #         event = action.get_attribute('analytics-event')
    #         if event == 'PDF View':
    #             href = action.get_attribute('href')
    #             coa_urls.append({'filename': action.text, 'url': href})

    # except TimeoutException:
    #     print('QR Code not loaded within %i seconds.' % max_delay)


    # Wait for the results to load.
    try:
        detect = EC.presence_of_element_located((By.TAG_NAME, 'ng-include'))
        WebDriverWait(driver, max_delay).until(detect)
    except TimeoutException:
        print('Results not loaded within %i seconds.' % max_delay)

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


if __name__ == '__main__':

    # Get all of the results.
    start = datetime.now()

    # Create a headless Chrome browser.
    options = Options()
    options.headless = True
    options.add_argument('--window-size=1920,1200')
    driver = webdriver.Chrome(options=options, service=service)

    # Read in the test results.
    filename = 'aggregated-cannabis-test-results.xlsx'
    data_dir = '../../.datasets/lab_results/raw_data'
    sheet_name = 'psi_labs_raw_data'
    datafile = f'{data_dir}/{filename}'
    all_data = pd.read_excel(datafile, sheet_name=sheet_name)

    # Get the details for each sample.
    # NOTE: Restricting subset here: Change to control pages collected.
    subset = all_data.loc[all_data['results'].isnull()][15_000:]
    rows = []
    total = len(subset)
    for index, values in subset.iterrows():
        percent = round(index / total, 4)
        message = 'Collecting (%.4f%%) (%i/%i):' % (percent, index + 1, total)
        print(message, values['product_name'])
        driver.get(values['lab_results_url'])
        details = get_psi_labs_test_result_details(driver)
        rows.append({**values.to_dict(), **details})

    # Save the results.
    data = pd.DataFrame(rows)
    data = pd.concat([all_data, data])
    data.drop_duplicates(subset='sample_id', keep='last', inplace=True)
    data.to_excel(datafile, index=False, sheet_name=sheet_name)
    end = datetime.now()
    print('Runtime took:', end - start)

    # Close the browser.
    driver.quit()
