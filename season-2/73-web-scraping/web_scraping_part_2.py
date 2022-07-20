"""
Web Scraping Part 2 | Cannabis Data Science #74 | 2022-07-13
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 7/8/2022
Updated: 7/8/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    Archive SC Labs test results.

Data Sources:

    - SC Labs Test Results
    URL: <https://client.sclabs.com/>

"""
# Internal imports.
from hashlib import sha256
import hmac

# External imports.
from bs4 import BeautifulSoup
from cannlytics.utils import snake_case
import requests


BASE = 'https://client.sclabs.com'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1667.0 Safari/537.36'}


def create_sample_id(private_key, public_key, salt):
    """Create a hash to be used as a sample ID.
    The standard is to use:
        1. `private_key = producer`
        2. `public_key = product_name`
        3. `salt = date_tested`
    """
    secret = bytes(private_key, 'UTF-8')
    message = snake_case(public_key) + snake_case(salt)
    sample_id = hmac.new(secret, message.encode(), sha256).hexdigest()
    return sample_id


#-----------------------------------------------------------------------
# Get client samples data:
# ✓ producer
# ✓ producer_image_url
# ✓ images
#-----------------------------------------------------------------------

# Request a client page.
url = '/'.join([BASE, 'client', '2821'])
params = {'limit': 100, 'page': 1}
response = requests.get(url, headers=HEADERS, params=params)
soup = BeautifulSoup(response.content, 'html.parser')

# Get producer.
details = soup.find('div', attrs={'id': 'detailQuickView'})
producer = details.find('h2').text

# Get producer image.
producer_image_url = details.find('img')['src'].replace('\n', '').strip()

# Get producer website.
element = details.find('span', attrs={'class': 'pp-social-web'})
producer_url = element.find('a')['href']

# Get all of the sample cards.
cards = soup.find_all('div', attrs={'class': 'grid-item'})

# DEV: Look at the first sample.
card = cards[0]

# Get the lab's internal ID.
lab_id = card['id'].replace('div_', '')

# Get the product name.
product_name = card.find('h3').text

# Get lab results URL.
actions = card.find('div', attrs={'class': 'sample-actions'})
lab_results_url = BASE + card.find('a')['href']

# Get the date tested.
mm, dd, yyyy = card.find('h6').text.split('-')
date_tested = '-'.join([yyyy, mm, dd])

# Get totals.
totals = card.find('div', attrs={'class': 'sample-details'})
values = totals.find_all('div')
total_thc = values[0].text.split(':')[-1].replace('%', '')
total_cbd = values[1].text.split(':')[-1].replace('%', '')
total_terpenes = values[2].text.split(':')[-1].replace('%', '')

# Create a sample ID.
sample_id = create_sample_id(producer, product_name, date_tested)


#-----------------------------------------------------------------------
# Get sample details data:
# ✓ analyses
# ✓ coa_id
# ✓ batch_number
# ✓ batch_size
# ✓ date_collected
# ✓ date_received
# ✓ date_tested
# ✓ images
# ✓ product_type
# - producer_dba
# ✓ license_number
# ✓ city
# - county (augmented)
# ✓ zip_code
# - distributor_name
# - distributor_license_number
# - distributor_city
# - distributor_county (augmented)
# - distributor_zip_code
# ✓ notes
# ✓ status
# ✓ results:
#   ✓ total_cannabinoids
#   ✓ total_cbg, total_thcv, total_cbc, total_cbdv
#   ✓ value, units, margin_of_error, lod, loq
#   - key (augmented)
#   - cas (augmented)
#-----------------------------------------------------------------------

# Request a sample page.
url = '/'.join([BASE, 'sample', '858084'])
response = requests.get(url, headers=HEADERS)
soup = BeautifulSoup(response.content, 'html.parser')

# Get the product type.
product_type = soup.find('p', attrs={'class': 'sdp-producttype'}).text

# Get the image.
image_url = soup.find('a', attrs={'data-popup': 'fancybox'})['href']

# Get the date tested.
element = soup.find('div', attrs={'class': 'sdp-masthead-data'})
mm, dd, yyyy = element.find('p').text.split('/')
date_tested = '-'.join([yyyy, mm, dd])

# Get the overall status: Pass / Fail.
status = soup.find('p', attrs={'class': 'sdp-result-pass'}).text
status = status.replace('\n', '').strip()

# Get the sample details.
sample_details = {}
for detail in details:
    try:
        label = detail.find('span').text
        value = detail.text
        value = value.replace(label, '')
        value = value.replace('\n', '').strip()
        label = label.replace(':', '')
    except AttributeError:
        pass
    sample_details[snake_case(label)] = value

# Get the Metrc IDs
metrc_ids = sample_details['source_metrc_uid'].split(',')

# Format the dates.
mm, dd, yyyy = sample_details['date_collected'].split('/')
date_collected = '-'.join([yyyy, mm, dd])
mm, dd, yyyy = sample_details['date_received'].split('/')
date_received = '-'.join([yyyy, mm, dd])

# Format the address.
address = sample_details['address'].split('*')[-1].strip()
city = address.split(',')[0]
zip_code = address.split(' ')[-1]

# Get the CoA ID.
coa_id = soup.find('p', attrs={'class': 'coa-id'}).text.split(':')[-1]

# TODO: Get the distributor data.

# Get all of the analyses and results.
analyses = []
results = []
cards = soup.find_all('div', attrs={'class': 'analysis-container'})
for element in cards:

    # Get the analysis.
    analysis = element.find('h4').text
    if 'Notes' in analysis:
        div = element.find('div', attrs={'class': 'section-inner'})
        notes = div.find('p').text
    if 'Analysis' not in analysis:
        continue
    analysis = snake_case(analysis.split(' Analysis')[0])
    analyses.append(analysis)

    # Get the method.
    bold = element.find('b')
    method = bold.parent.text.replace('Method: ', '')

    # Get all of the results for the analysis.
    # - value, units, margin_of_error, lod, loq
    table = element.find('table')
    rows = table.find_all('tr')
    for row in rows[1:]:
        cells = row.find_all('td')
        result = {}
        for cell in cells:
            key = cells['class']
            value = cell.text
            result[key] = value
        results.append(result)


#-----------------------------------------------------------------------
# Future work: Processing the raw data.
#-----------------------------------------------------------------------

# TODO: Find the county for the zip_code

# Normalize the `results`, `images`.

# Standardize the `product_type` and `status`.

# Optional: Add CAS to results.
