"""
Computer Vision Application
Cannabis Data Science #75 | 2022-07-20
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 7/16/2022
Updated: 7/20/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    Certificates of analysis (CoAs) are abundant for cultivators,
    processors, retailers, and consumers too,, but the data is often
    locked away. Rich, valuable laboratory data so close, yet so far
    away! Cannlytics puts these vital data points in your hands by
    parsing PDFs and URLs, finding all the data, standardizing the data,
    and cleanly returning the data to you.

Data Sources:

    - Confident Cannabis
    URL: <https://orders.confidentcannabis.com/>
    Data points:

        ✓ analyses
        - {analysis}_method
        ✓ {analysis}_status
        ✓ classification
        ✓ coa_urls
        ✓ date_tested
        - date_received
        ✓ images
        ✓ lab_results_url
        ✓ producer
        ✓ product_name
        ✓ product_type
        ✓ predicted_aromas
        ✓ results
        - sample_weight
        - total_cannabinoids (calculated)
        ✓ total_thc
        ✓ total_cbd
        - total_terpenes (calculated)
        ✓ sample_id (generated)
        ✓ strain_name
        ✓ lab_id
        ✓ lab
        ✓ lab_image_url
        - lab_license_number
        ✓ lab_address
        ✓ lab_city
        - lab_county (augmented)
        ✓ lab_state
        ✓ lab_zipcode
        ✓ lab_phone
        ✓ lab_email
        - lab_latitude (augmented)
        - lab_longitude (augmented)

    - TagLeaf LIMS
    URL: <https://lims.tagleaf.com>
    Data points:

        ✓ analyses
        - {analysis}_method
        ✓ {analysis}_status
        - classification
        - coa_urls
        ✓ date_tested
        - date_received
        ✓ distributor
        ✓ distributor_license_number
        ✓ distributor_license_type
        - distributor_latitude (augmented)
        - distributor_longitude (augmented)
        ✓ images
        ✓ lab_results_url
        ✓ producer
        - producer_latitude (augmented)
        - producer_longitude (augmented)
        ✓ product_name
        ✓ product_type
        ✓ results
        - sample_weight
        ✓ status
        ✓ total_cannabinoids
        ✓ total_thc
        ✓ total_cbd
        - total_terpenes (calculated)
        ✓ sample_id (generated)
        - strain_name (predict later)
        - lab_id
        ✓ lab
        ✓ lab_image_url
        ✓ lab_license_number
        ✓ lab_address
        - lab_city
        - lab_county (augmented)
        - lab_state
        - lab_zipcode
        ✓ lab_phone
        - lab_email
        - lab_latitude (augmented)
        - lab_longitude (augmented)

Resources:

    - https://github.com/huggingface/transformers

"""
# External imports.
import pandas as pd
import pdfplumber
from pyzbar.pyzbar import decode
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import (
    ElementNotInteractableException,
    TimeoutException,
)
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Internal imports.
from cannlytics.data.coas import CoADoc
from cannlytics.data.data import create_sample_id
from cannlytics.utils import (
    strip_whitespace,
    snake_case,
    unzip_files,
)

ANALYSES = {
    'cannabinoids': {'names': ['potency', 'POT']},
    'terpenes': {'names': ['terpene', 'TERP', 'terpenoids']},
    'residual_solvents': {'names': ['solvent', 'RST']},
    'pesticides': {'names': ['pesticide', 'PEST']},
    'microbes': {'names': ['microbial', 'MICRO']},
    'mycotoxins': {'names': ['mycotoxins', 'MYCO']},
    'heavy_metals': {'names': ['metal', 'MET']},
    'foreign_matter': {'names': ['foreign_materials']},
    'moisture_content': {'names': ['moisture']},
    'water_activity': {'names': ['WA']},
}
ANALYTES = {
    'CBC': 'cbc',
    'CBCA': 'cbca',
    'CBD': 'cbd',
    'CBDA': 'cbda',
    'CBDV': 'cbdv',
    'CBDVA': 'cbdva',
    'CBG': 'cbg',
    'CBGA': 'cbga',
    'CBN': 'cbn',
    'Δ8-THC': 'delta_8_thc',
    'Δ9-THC': 'delta_9_thc',
    'THCA': 'thca',
    'THCV': 'thcv',
    'THCVA': 'thcva',
    'Total THC(Total THC = (THCA x 0.877) + THC)': 'total_thc',
    'Total CBD(Total CBD = (CBDA x 0.877) + CBD)': 'total_cbd',
    'Total Terpenes *': 'total_terpenes',
    'Terpinolene': 'terpinolene',
    'β-Caryophyllene': 'beta_caryophyllene',
    'α-Humulene': 'humulene',
    'β-Myrcene': 'beta_myrcene',
    'Linalool': 'linalool',
    'β-Pinene': 'beta_pinene',
    'd-Limonene': 'd_limonene',
    'α-Pinene': 'alpha_pinene',
    'β-Ocimene': 'ocimene',
    'cis-Nerolidol': 'cis_nerolidol',
    'α-Bisabolol': 'alpha_bisabolol',
    '3-Carene': 'carene',
    'Δ3-Carene': 'carene',
    'trans-Nerolidol': 'trans_nerolidol',
    'α-Terpinene': 'alpha_terpinene',
    'γ-Terpinene': 'gamma_terpinene',
    'Terpinen-4-ol': 'terpineol',
    'Caryophyllene Oxide': 'caryophyllene_oxide',
    'Geraniol': 'geraniol',
    'Eucalyptol': 'eucalyptol',
    'Camphene': 'camphene',
    'Guaiol': 'guaiol',
    'Isopulegol': 'isopulegol',
    'p-Cymene': 'p_cymene',
    'α-Ocimene': 'alpha_ocimene',
    '* Beyond scope of accreditation': 'wildcard',
    'Moisture': 'moisture_content',
    'Aspergillus flavus': 'aspergillus_flavus',
    'Aspergillus fumigatus': 'aspergillus_fumigatus',
    'Aspergillus niger': 'aspergillus_niger',
    'Aspergillus terreus': 'aspergillus_terreus',
    'Salmonella spp.': 'salmonella_spp',
    'Shiga toxin-producing E. coli': 'shiga_toxin_producing_e_coli',
    'Aflatoxin B1': 'aflatoxin_b1',
    'Aflatoxin B2': 'aflatoxin_b2',
    'Aflatoxin G1': 'aflatoxin_g1',
    'Aflatoxin G2': 'aflatoxin_g2',
    'Aflatoxins': 'total_aflatoxins',
    'Ochratoxin A': 'ochratoxin_a',
    'Abamectin': 'abamectin',
    'Acephate': 'acephate',
    'Acequinocyl': 'acequinocyl',
    'Acetamiprid': 'acetamiprid',
    'Aldicarb': 'aldicarb',
    'Azoxystrobin': 'azoxystrobin',
    'Bifenazate': 'bifenazate',
    'Bifenthrin': 'bifenthrin',
    'Boscalid': 'boscalid',
    'Captan': 'captan',
    'Carbaryl': 'carbaryl',
    'Carbofuran': 'carbofuran',
    'Chlorantranil-iprole': 'chlorantraniliprole',
    'Chlordane': 'chlordane',
    'Chlorfenapyr': 'chlorfenapyr',
    'Chlorpyrifos': 'chlorpyrifos',
    'Clofentezine': 'clofentezine',
    'Coumaphos': 'coumaphos',
    'Cyfluthrin': 'cyfluthrin',
    'Cypermethrin': 'cypermethrin',
    'Daminozide': 'daminozide',
    'Diazinon': 'diazinon',
    'Dichlorvos': 'dichlorvos',
    'Dimethoate': 'dimethoate',
    'Dimethomorph': 'dimethomorph',
    'Ethoprophos': 'ethoprophos',
    'Etofenprox': 'etofenprox',
    'Etoxazole': 'etoxazole',
    'Fenhexamid': 'fenhexamid',
    'Fenoxycarb': 'fenoxycarb',
    'Fenpyroximate': 'fenpyroximate',
    'Fipronil': 'fipronil',
    'Flonicamid': 'flonicamid',
    'Fludioxonil': 'fludioxonil',
    'Hexythiazox': 'hexythiazox',
    'Imazalil': 'imazalil',
    'Imidacloprid': 'imidacloprid',
    'Kresoxim-methyl': 'kresoxim_methyl',
    'Malathion': 'malathion',
    'Metalaxyl': 'metalaxyl',
    'Methiocarb': 'methiocarb',
    'Methomyl': 'methomyl',
    'Methyl parathion': 'methyl_parathion',
    'Mevinphos': 'mevinphos',
    'Myclobutanil': 'myclobutanil',
    'Naled': 'naled',
    'Oxamyl': 'oxamyl',
    'Paclobutrazol': 'paclobutrazol',
    'Pentachloroni-trobenzene': 'pentachloroni_trobenzene',
    'Permethrin': 'permethrin',
    'Phosmet': 'phosmet',
    'Piperonylbuto-xide': 'piperonyl_butoxide',
    'Prallethrin': 'prallethrin',
    'Propiconazole': 'propiconazole',
    'Propoxur': 'propoxur',
    'Pyrethrins': 'pyrethrins',
    'Pyridaben': 'pyridaben',
    'Spinetoram': 'spinetoram',
    'Spinosad': 'spinosad',
    'Spiromesifen': 'spiromesifen',
    'Spirotetramat': 'spirotetramat',
    'Spiroxamine': 'spiroxamine',
    'Tebuconazole': 'tebuconazole',
    'Thiacloprid': 'thiacloprid',
    'Thiamethoxam': 'thiamethoxam',
    'Trifloxystrob-in': 'trifloxystrobin',
    'Arsenic': 'arsenic',
    'Cadmium': 'cadmium',
    'Lead': 'lead',
    'Mercury': 'mercury',
    'Water Activity': 'water_activity',
    'Imbedded Foreign Material': 'foreign_matter',
    'Insect Fragments, Hair, Mammal Excrement': 'foreign_matter_fragments',
    'Mold': 'mold',
    'Sand, Soil, Cinders, Dirt': 'soil',
}
DECARB = 0.877 # Source: <https://www.conflabs.com/why-0-877/>
DECODINGS = {
    '<LOQ': 0,
    '<LOD': 0,
    'ND': 0,
    'NR': None,
    'N/A': None,
}
KEYS = {
    'Analyte': 'name',
    'Labeled Amount': 'sample_weight',
    'Limit': 'limit',
    'Detected': 'value',
    'LOD': 'lod',
    'LOQ': 'loq',
    'Pass/Fail': 'status',
    'metrc_src_uid': 'source_metrc_uid',
    'matrix': 'product_type',
    'collected_on': 'date_collected',
    'received_on': 'date_received',
    'moisture': 'moisture_content',
    'terpenoids': 'terpenes',
    'foreign_materials': 'foreign_matter',
}
LIMS = {
    'Confident Cannabis': {
        'algorithm': 'parse_cc_url',
        'key': 'Con\x00dent Cannabis',
        'qr_code_index': 3,
        'url': 'https://orders.confidentcannabis.com',
    },
    'TagLeaf LIMS': {
        'algorithm': 'parse_tagleaf_url',
        'key': 'lims.tagleaf',
        'qr_code_index': 2,
        'url': 'https://lims.tagleaf.com',
    },
    # TODO: Implement an algorithm to parse any custom CoA.
    'custom': {
        'algorithm': '',
        'key': 'custom',
        'qr_code_index': -1,
        'url': 'https://cannlytics.com',
    }
}
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1667.0 Safari/537.36'}

#-----------------------------------------------------------------------
# --- Setup ---
#-----------------------------------------------------------------------

# Create a CoA parser.
parser = CoADoc()
sample = {'analyses': [], 'results': []}

# Test Confident Cannabis CoAs.
cc_coa_pdf = f'../../.datasets/coas/Sunbeam.pdf'
cc_coa_url = 'https://share.confidentcannabis.com/samples/public/share/4ee67b54-be74-44e4-bb94-4f44d8294062'


#-----------------------------------------------------------------------
# --- Confident Cannabis Parsing ---
#-----------------------------------------------------------------------

# # Get the PDF.
pdf_file = pdfplumber.open(cc_coa_pdf)
page = pdf_file.pages[0]
for img in page.images:
    try:
        # Decode a QR code.
        y = page.height
        bbox = (img['x0'], y - img['y1'], img['x1'], y - img['y0'])
        crop = page.crop(bbox)
        obj = crop.to_image(resolution=300)
        image_data = decode(obj.original)
        url = image_data[0].data.decode('utf-8')
        if url:
            break
    except:
        continue

# # Get the `lab_results_url` from the QR code on the first page.
# lab_results_url = parser.find_pdf_qr_code_url(cc_coa_pdf)

lab_results_url = cc_coa_url

# Load the lab results with Selenium.
service = Service()
options = Options()
options.headless = False # DEV: In production set to `True`.
options.add_argument('--window-size=1920,1200')
driver = webdriver.Chrome(options=options, service=service)
driver.get(lab_results_url)

# Wait for the results to load.
max_delay = 10
try:
    el = (By.CLASS_NAME, 'product-box-cc')
    detect = EC.presence_of_element_located(el)
    WebDriverWait(driver, max_delay).until(detect)
except TimeoutException:
    print('Failed to load page within %i seconds.' % max_delay)

# Find the sample image.
# try:
#     element = driver.find_element(by=By.CLASS_NAME, value='product-box-cc')
# except:
element = driver.find_element(by=By.CLASS_NAME, value='product-box-cc')
# element = driver.find_element(by=By.CLASS_NAME, value='product-shot')
img = element.find_element(by=By.TAG_NAME, value='img')
image_url = img.get_attribute('src')
filename = image_url.split('/')[-1]
sample['images'] = [{'url': image_url, 'filename': filename}]

# Try to get sample details.
el = driver.find_element(by=By.CLASS_NAME, value='product-desc')
block = el.text.split('\n')
sample['product_name'] = block[0]
sample['lab_id'] = block[1]
sample['classification'] = block[2]
sample['strain_name'], sample['product_type'] = tuple(block[3].split(', '))

# Get the date tested.
el = driver.find_element(by=By.CLASS_NAME, value='report')
span = el.find_element(by=By.TAG_NAME, value='span')
tooltip = span.get_attribute('uib-tooltip')
tested_at = tooltip.split(': ')[-1]
sample['date_tested'] = pd.to_datetime('10/20/21 5:07 PM').isoformat()

# Get the CoA URL.
button = el.find_element(by=By.TAG_NAME, value='button')
href = button.get_attribute('href')
base = lab_results_url.split('/report')[0]
coa_url = base.replace('/#!', '') + href
filename = image_url.split('/')[-1].split('?')[0] + '.pdf'
sample['coa_urls'] = [{'url': coa_url, 'filename': filename}]

# Find the analyses and results.
els = driver.find_elements(by=By.CLASS_NAME, value='ibox')
for i, el in enumerate(els):
    try:
        title = el.find_element(by=By.TAG_NAME, value='h5').text.lower()
    except:
        continue

    # Try to get cannabinoids data.
    if title == 'cannabinoids':
        totals = el.find_elements(by=By.TAG_NAME, value='compound-box')
        for total in totals:
            value = total.find_element(by=By.CLASS_NAME, value='value').text
            units = total.find_element(by=By.CLASS_NAME, value='unit').text
            name = total.find_element(by=By.CLASS_NAME, value='name').text
            key = snake_case(name)
            sample[key] = value
            sample[f'{key}_units'] = units.replace('%', 'percent')

        # Get the cannabinoids totals.
        columns = ['name', 'value', 'mg_g']
        table = el.find_element(by=By.TAG_NAME, value='table')
        rows = table.find_elements(by=By.TAG_NAME, value='tr')
        for row in rows[1:]:
            result = {}
            cells = row.find_elements(by=By.TAG_NAME, value='td')
            for i, cell in enumerate(cells):
                key = columns[i]
                result[key] = cell.get_attribute('textContent').strip()
            sample['results'].append(result)

    # Try to get terpene data.
    if title == 'terpenes':
        columns = ['name', 'value', 'mg_g']
        table = el.find_element(by=By.TAG_NAME, value='table')
        rows = table.find_elements(by=By.TAG_NAME, value='tr')
        for row in rows[1:]:
            result = {}
            cells = row.find_elements(by=By.TAG_NAME, value='td')
            for i, cell in enumerate(cells):
                key = columns[i]
                result[key] = cell.get_attribute('textContent').strip()
            sample['results'].append(result)

        # Try to get predicted aromas.
        container = el.find_element(by=By.CLASS_NAME, value='row')
        aromas = container.text.split('\n')
        sample['predicted_aromas'] = [snake_case(x) for x in aromas]

    # Ty to get screening data.
    if title == 'safety':
        sample['status'] = el.find_element(by=By.CLASS_NAME, value='sample-status').text
        table = el.find_element(by=By.TAG_NAME, value='table')
        rows = table.find_elements(by=By.TAG_NAME, value='tr')
        for row in rows[1:]:
            cells = row.find_elements(by=By.TAG_NAME, value='td')
            status = cells[1].get_attribute('textContent').strip()
            if status == 'Not Tested':
                continue
            analysis = snake_case(cells[0].get_attribute('textContent'))
            sample[f'{analysis}_status'] = status.lower()
            sample['analyses'].append(analysis)

            # Click the row. and get all of the results from the modal!
            columns = ['compound', 'status', 'value', 'limit', 'loq']
            if row.get_attribute('class') == 'clickable-content':
                try:
                    row.click()
                except ElementNotInteractableException:
                    continue
                modal = driver.find_element(by=By.ID, value='safety-modal-table')
                modal_table = modal.find_element(by=By.TAG_NAME, value='tbody')
                modal_rows = modal_table.find_elements(by=By.TAG_NAME, value='tr')
                headers = modal.find_elements(by=By.TAG_NAME, value='th')
                units = headers[-1].text.split('(')[-1].replace(')', '')
                for modal_row in modal_rows:
                    result = {'units': units}
                    modal_cells = modal_row.find_elements(by=By.TAG_NAME, value='td')
                    for i, cell in enumerate(modal_cells):
                        key = columns[i]
                        result[key] = cell.get_attribute('textContent').strip()
                    sample['results'].append(result)
                try:
                    body = driver.find_element(
                        by=By.TAG_NAME,
                        value='body'
                    )
                    body.click()
                except ElementNotInteractableException:
                    continue

    # Try to get lab data.
    if title == 'order info':
        img = el.find_element(by=By.TAG_NAME, value='img')
        block = el.find_element(by=By.TAG_NAME, value='confident-address').text.split('\n')
        street = block[1]
        address = tuple(block[2].split(', '))
        sample['lab'] = block[0]
        sample['lab_address'] = f'{street} {address}'
        sample['lab_image_url'] = img.get_attribute('src')
        sample['lab_street'] = street
        sample['lab_city'] = address[0]
        sample['lab_state'], sample['lab_zipcode'] = tuple(address.split(' '))
        sample['lab_phone'] = block[-2].split(': ')[-1]
        sample['lab_email'] = block[-1]
        sample['producer'] = el.find_element(by=By.CLASS_NAME, value='public-name').text

# At this stage, create a sample ID.
sample['sample_id'] = create_sample_id(
    private_key=sample['producer'],
    public_key=sample['product_name'],
    salt=sample['date_tested'],
)

# Close the Chrome driver once all PDFs have been parsed.
driver.quit()


#-----------------------------------------------------------------------
# --- Bonus: TagLeaf LIMS Parsing ---
#-----------------------------------------------------------------------

# # Test TagLeaf LIMS CoAs.
# tagleaf_coa_pdf = f'{DATA_DIR}/Sunbeam.pdf'
# tagleaf_coa_url = 'https://lims.tagleaf.com/coas/F6LHqs9rk9vsvuILcNuH6je4VWCiFzdhgWlV7kAEanIP24qlHS'
# tagleaf_coa_short_url = 'https://lims.tagleaf.com/coa_/F6LHqs9rk9'

# # Find the QR code to public lab results.
# qr_code_index = LIMS['TagLeaf LIMS']['qr_code_index']
# lab_results_url = parser.find_pdf_qr_code(front_page, image_index=None)

# # Get the HTML!
# response = requests.get(lab_results_url, headers=HEADERS)
# soup = BeautifulSoup(response.content, 'html.parser')

# # Get the date tested.
# el = soup.find('p', attrs={'class': 'produced-statement'})
# date_tested = pd.to_datetime(el.text.split(': ')[-1]).isoformat()
# sample['date_tested'] = date_tested

# # Get lab details.
# el = soup.find('section', attrs={'class': 'header-container'})
# img = el.find('img')
# pars = el.find_all('p')
# details = [strip_whitespace(x) for x in pars[0].text.split('//')]
# address = details[1]
# sample['lab'] = details[0]
# sample['lab_address'] = address
# sample['lab_image_url'] = img.attrs['src']
# sample['lab_phone'] = details[2].replace('PH: ', '')

# # Get data from headings.
# headings = soup.find_all('p', attrs={'class': 'h5'}, limit=2)
# parts = strip_whitespace(headings[0].text.split('//')[0]).split(' (')
# product_name = parts[0]
# sample['product_name'] = product_name
# sample['product_type'] = parts[1].replace(')', '')
# sample['status'] = strip_whitespace(headings[1].text.split(':')[-1]).lower()

# # Get cannabinoid totals.
# el = soup.find('div', attrs={'class': 'cannabinoid-overview'})
# rows = el.find_all('div', attrs={'class': 'row'})
# for row in rows:
#     pars = row.find_all('p')
#     key = snake_case(strip_whitespace(pars[1].text))
#     value = strip_whitespace(pars[0].text)
#     sample[key] = value

# # Get cultivator and distributor details.
# els = soup.find_all('div', attrs={'class': 'license'})
# values = [x.text for x in els[0].find_all('p')]
# producer = values[1]
# sample['producer'] = producer
# sample['license_number'] = values[3]
# sample['license_type'] = values[5]
# values = [x.text for x in els[1].find_all('p')]
# sample['distributor'] = values[1]
# sample['distributor_license_number'] = values[3]
# sample['distributor_license_type'] = values[5]

# # Get the sample image.
# el = soup.find('div', attrs={'class': 'sample-photo'})
# img = el.find('img')
# image_url = img['src']
# filename = image_url.split('/')[-1]
# sample['images'] = [{'url': image_url, 'filename': filename}]

# # Get the sample details
# el = soup.find('div', attrs={'class': 'sample-info'})
# pars = el.find_all('p')
# for par in pars:
#     key = snake_case(par.find('span').text)
#     key = KEYS.get(key, key) # Get preferred key.
#     value = ''.join([x for x in par.contents if type(x) == NavigableString])
#     value = strip_whitespace(value)
#     print(key, value)

# # Get the lab ID and metrc ID.
# sample['lab_id'] = sample['sample_id']
# sample['metrc_ids'] = [sample['source_metrc_uid']]

# # Format `date_collected` and `date_received` dates.
# sample['date_collected'] = pd.to_datetime(sample['date_collected']).isoformat()
# sample['date_received'] = pd.to_datetime(sample['date_received']).isoformat()

# # Get the analyses and `{analysis}_status`.
# analyses = []
# el = soup.find('div', attrs={'class': 'tests-overview'})
# blocks = strip_whitespace(el.text)
# blocks = [x for x in blocks.split('    ') if x]
# for i, value in enumerate(blocks):
#     if i % 2:
#         analysis = analyses[-1]
#         if value != '\xa0':
#             sample[f'{analysis}_status'] = value.lower()
#     else:
#         analysis = snake_case(value)
#         analysis = KEYS.get(analysis, analysis) # Get preferred key.
#         analyses.append(analysis)
# sample['analyses'] = analyses

# # Get `{analysis}_method`s.
# els = soup.find_all('div', attrs={'class': 'table-header'})
# for el in els:
#     analysis = el.attrs['id'].replace('_test', '')
#     analysis = KEYS.get(analysis, analysis) # Get preferred key.
#     heading = el.find('h3')
#     title = ''.join([x for x in heading.contents if type(x) == NavigableString])
#     sample[f'{analysis}_method'] = strip_whitespace(title)

# # Get the `results`.
# tables = soup.find_all('table')
# for table in tables:

#     # Get the columns, noting that `value` is repeated for `mg_g`.
#     headers = table.find_all('th')
#     columns = [KEYS[strip_whitespace(x.text)] for x in headers]
#     rows = table.find_all('tr')[1:]
#     for row in rows:
#         mg_g = False
#         result = {}
#         cells = row.find_all('td')
#         for i, cell in enumerate(cells):
#             key = columns[i]
#             if key == 'value' and mg_g:
#                 key = 'mg_g'
#             if key == 'value':
#                 mg_g = True
#             result[key] = strip_whitespace(cell.text)
#         sample['results'].append(result)

# # At this stage, create a sample ID.
# sample['sample_id'] = create_sample_id(
#     private_key=producer,
#     public_key=product_name,
#     salt=date_tested,
# )
