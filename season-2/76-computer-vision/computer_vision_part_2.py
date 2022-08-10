"""
Computer Vision Application
Cannabis Data Science #76 | 2022-07-27
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 7/27/2022
Updated: 7/27/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    Parse a Green Leaf Labs CoA.

Data Points:

    ✓ analyses
    - {analysis}_method
    ✓ {analysis}_status
    ✓ batch_size
    x coa_urls
    ✓ date_harvested
    ✓ date_tested
    ✓ date_received
    ✓ date_sampled
    ✓ distributor
    ✓ distributor_address
    ✓ distributor_street
    ✓ distributor_city
    ✓ distributor_state
    ✓ distributor_zipcode
    ✓ distributor_license_number
    x images
    ✓ image_data
    x lab_results_url
    ✓ metrc_lab_id
    ✓ metrc_source_id
    ✓ producer
    ✓ producer_address
    ✓ producer_street
    ✓ producer_city
    ✓ producer_state
    ✓ producer_zipcode
    ✓ producer_license_number
    ✓ product_name
    ✓ product_type
    ✓ results
    ✓ sample_weight
    ✓ sample_size
    ✓ sampling_method
    ✓ status
    ✓ total_cannabinoids
    ✓ total_thc
    ✓ total_cbd
    ✓ total_terpenes
    ✓ sample_id (generated)
    - strain_name (augmented)
    ✓ lab_id

Static Data Points:

    ✓ lab
    ✓ lab_image_url
    ✓ lab_license_number (should be dynamic)
    ✓ lab_address (should be dynamic)
    ✓ lab_street (should be dynamic)
    ✓ lab_city (should be dynamic)
    ✓ lab_county (augmented)
    ✓ lab_state (should be dynamic)
    ✓ lab_zipcode (should be dynamic)
    ✓ lab_phone (should be dynamic)
    - lab_email
    ✓ lab_website
    ✓ lab_latitude (augmented) (should be dynamic)
    ✓ lab_longitude (augmented) (should be dynamic)

Setup:

    Install `cannlytics` from the `dev` branch.

    ```
    pip uninstall cannlytics
    git clone -b dev https://github.com/cannlytics/cannlytics.git
    cd cannlytics
    pip install .

"""
# Standard imports.
from ast import literal_eval
import re

# External imports.
import pandas as pd
import pdfplumber

# Internal imports.
from cannlytics.data.data import create_sample_id
from cannlytics.utils.utils import (
    snake_case,
    split_list,
    strip_whitespace,
)

# It is assumed that the lab has the following details.
# Future work: Make this dynamic to handle multiple lab locations.
# E.g. Green Leaf Lab has a California and an Oregon location.
GREEN_LEAF_LAB = {
    'coa_parsing_algorithm': 'parse_green_leaf_lab_pdf',
    'lims': 'Green Leaf Lab',
    'lab': 'Green Leaf Lab',
    'lab_image_url': 'https://cdn-djjmk.nitrocdn.com/MuWSCTBsUZpIUufaWqGQkErSrYFMxIqD/assets/static/optimized/rev-a199899/wp-content/uploads/2018/12/greenleaf-logo.png',
    'lab_license_number': ' C8-0000078-LIC', # <- Make dynamic.
    'lab_address': '251 Lathrop Way Suites D&E Sacramento, CA 95815', # <- Make dynamic.
    'lab_street': '251 Lathrop Way Suites D&E', # <- Make dynamic.
    'lab_city': 'Sacramento', # <- Make dynamic.
    'lab_county': 'Sacramento', # <- Make dynamic.
    'lab_state': 'CA', # <- Make dynamic.
    'lab_zipcode': '95815', # <- Make dynamic.
    'lab_phone': '916-924-5227', # <- Make dynamic.
    'lab_email': '', # <- Make dynamic.
    'lab_website': 'https://greenleaflab.org/',
    'lab_latitude': '38.596060', # <- Make dynamic.
    'lab_longitude': '-121.459870', # <- Make dynamic.
}

# It is assumed that there are the following analyses on each CoA.
GREEN_LEAF_LAB_ANALYSES = {
    'cannabinoids': {
        'name': 'Potency Analysis by HPLC',
        'columns': ['name', 'lod', 'loq', 'value', 'mg_g'],
    },
    'pesticides': {
        'name': 'Pesticide Analysis by GCMS/LCMS',
        'columns': ['name', 'value', 'limit', 'lod', 'loq', 'units'],
        'double_column': True,
    },
    'water_activity': {
        'name': 'Water Activity by Aqua Lab',
        'columns': ['name', 'value', 'units', 'lod', 'loq'],
    },
    'moisture_content': {
        'name': 'Moisture by Moisture Balance',
        'columns': ['name', 'value', 'units'],
    },
    'terpenes': {
        'name': 'Terpene Analysis by GCMS',
        'columns': ['name', 'value', 'mg_g'],
        'double_column': True,
    },
    'heavy_metals': {
        'name': 'Metals Analysis by ICPMS',
        'columns': ['name', 'value', 'limit', 'lod', 'loq', 'units'],
    },
    'mycotoxins': {
        'name': 'Mycotoxins by LCMSMS',
        'columns': ['name', 'value', 'limit', 'loq', 'loq', 'units'],
    },
    'microbials': {
        'name': 'Microbials by PCR',
        'columns': ['name', 'status', 'limit', 'lod', 'loq', 'units', 'value'],
    },
    'foreign_matter': {
        'name': 'Filth and Foreign Material Inspection by Magnification',
        'columns': ['name', 'status', 'limit', 'lod', 'loq', 'units'],
    },
}

# It is assumed that the CoA has the following parameters.
GREEN_LEAF_LAB_COA = {
    'coa_qr_code_index': None,
    'coa_image_index': 2,
    'coa_page_area': '(0, 198, 612, 693)',
    'coa_distributor_area': '(0, 79.2, 244.8, 142.56)',
    'coa_producer_area': '(244.8, 79.2, 612, 142.56)',
    'coa_sample_details_area': '(0, 126.72, 612, 205.92)',
    # Optional: Clean up the fields!
    'coa_fields': {
        'licensenumber': 'lab_license_number',
        'lab_sample_id': 'lab_id',
        'matrix': 'product_type',
        'batch_size': 'batch_size',
        'sample_size': 'sample_size',
        'date_sampled': 'date_sampled',
        'date_received': 'date_received',
        'harvesttoprocessing_date': 'date_harvested',
        'product_density': 'sample_weight',
        'overall_batch': 'status',
        'cannabinoids': 'cannabinoids_status',
        'pesticides': 'pesticides_status',
        'water_activity': 'water_activity_status',
        'moisture_content': 'moisture_content_status',
        'terpene_analysis_add_on': 'terpenes_status',
        'microbials': 'microbials_status',
        'metals': 'heavy_metals_status',
        'foreign_material': 'foreign_matter_status',
        'mycotoxins': 'mycotoxins_status',
        'sampling_method': 'sampling_method',
        'Test RFID': 'metrc_lab_id',
        'Source RFID': 'metrc_source_id',
        'Lab Sample ID': 'lab_id',
        'Sampling Method/SOP': 'sampling_method',
        'Source Batch ID': 'batch_id',
        'Matrix': 'product_type',
        'Batch Size': 'batch_size',
        'Sample Size': 'sample_size',
        'Date Sampled': 'date_sampled',
        'Date Received': 'date_received',
        'Harvest/Processing Date': 'date_harvested',
        'Product Density': 'sample_weight',
    },
    'coa_sample_detail_fields': [
        'Test RFID',
        'Source RFID',
        'Lab Sample ID',
        'Sampling Method/SOP',
        'Source Batch ID',
        'Matrix',
        'Batch Size',
        'Sample Size',
        'Date Sampled',
        'Date Received',
        'Harvest/Processing Date',
        'Product Density',
    ],
    'coa_skip_values': [
        'Date/Time',
        'Analysis Method',
        'Analyte',
        'ND - Compound not detected',
        '<LOQ - Results below the Limit of Quantitation',
        'Results above the Action Level',
        'Sesquiterpenes',
        'Monoterpenes',
    ],
    'coa_replacements': [
        {'text': '< LOQ', 'key': '<LOQ'},
        {'text': '< LOD', 'key': '<LOD'},
        {'text': 'No detection in 1 gram', 'key': 'ND'},
        {'text': 'Ocimene isomer II', 'key': 'beta-ocimene'},
        {'text': 'Ocimene isomer I', 'key': 'alpha-ocimene'},
        {'text': 'p-Mentha-1,5-diene', 'key': 'p-Mentha-1-5-diene'},
        {'text': 'Methyl parathion', 'key': 'Methyl-parathion'},
    ],
}


def augment_analyte_result(result, columns, parts):
    """Quickly augment an analyte result."""
    r = result.copy()
    if len(parts) > len(columns):
        break_point = len(parts) - len(columns) + 1
        name = ' '.join(parts[:break_point])
        r['name'] = name
        r['key'] = snake_case(name)
        for i, part in enumerate(parts[break_point:]):
            r[columns[i + 1]] = part
    else:
        for i, part in enumerate(parts):
            if i == 0:
                r[columns[i]] = part
                r['key'] = snake_case(part)
            else:
                r[columns[i]] = part
    return r


#------------------------------------------------------------------------------
# CoADoc Example
#------------------------------------------------------------------------------

# Test parsing a Green Leaf Lab CoA
from cannlytics.data.coas import CoADoc

# Specify where your test CoA lives.
DATA_DIR = '../../.datasets/coas'
coa_pdf = f'{DATA_DIR}/Raspberry Parfait.pdf'

# [✓] TEST: Detect the lab / LIMS that generated the CoA.
parser = CoADoc()
lab = parser.identify_lims(coa_pdf)
assert lab == 'Green Leaf Lab'

# [✓] TEST: Parse a Green Leaf Lab CoA.
parser = CoADoc()
data = parser.parse_green_leaf_lab_pdf(parser, coa_pdf)
assert data is not None


#------------------------------------------------------------------------------
# Inspect how a CoADoc algorithm functions, line by line.
#------------------------------------------------------------------------------

# Get the lab / LIMS analyses and CoA parameters.
obs = {}
lab = GREEN_LEAF_LAB
lab_analyses = GREEN_LEAF_LAB_ANALYSES
coa_parameters = GREEN_LEAF_LAB_COA

# Get the lab's analyses.
standard_analyses = list(lab_analyses.keys())
analysis_names = [x['name'] for x in lab_analyses.values()]

# Read the PDF.
if isinstance(coa_pdf, str):
    report = pdfplumber.open(coa_pdf)
else:
    report = coa_pdf
front_page = report.pages[0]

# Look at the PDF!
img = front_page.to_image(resolution=150)
img.draw_rects(front_page.rects)

# Look at all the images.
img = front_page.to_image(resolution=150)
img.draw_rects(front_page.images)

# Look at all of the words.
img = front_page.to_image(resolution=150)
img.draw_rects(front_page.extract_words())

# Look at all of the lines.
img = front_page.to_image(resolution=150)
img.draw_rects(front_page.lines)

# Look at all of the characters
img = report.pages[1].to_image(resolution=150)
img.reset().draw_rects(report.pages[1].chars)

# Get the lab-specific CoA page areas.
page_area = literal_eval(coa_parameters['coa_page_area'])
distributor_area = literal_eval(coa_parameters['coa_distributor_area'])
producer_area = literal_eval(coa_parameters['coa_producer_area'])
sample_details_area = literal_eval(coa_parameters['coa_sample_details_area'])

# Look at front page areas.
img = front_page.to_image(resolution=150)
img.draw_rects([
    # distributor_area,
    # producer_area,
    sample_details_area,
])

# Look at result page areas.
img = report.pages[1].to_image(resolution=150)
img.draw_rects([page_area])

# Get lab CoA specific fields.
coa_fields = coa_parameters['coa_fields']
coa_replacements = coa_parameters['coa_replacements']
sample_details_fields = coa_parameters['coa_sample_detail_fields']
skip_values = coa_parameters['coa_skip_values']

# Get all distributor details.
crop = front_page.within_bbox(distributor_area)
details = crop.extract_text().split('\n')
address = details[2]
parts = address.split(',')
street = parts[0]
subparts = parts[-1].strip().split(' ')
city = ' '.join(subparts[:-2])
state, zipcode = subparts[-2], subparts[-1]
address = ','.join([street, details[3]])
obs['distributor'] = details[1]
obs['distributor_address'] = address
obs['distributor_street'] = street
obs['distributor_city'] = city
obs['distributor_state'] = state
obs['distributor_zipcode'] = zipcode
obs['distributor_license_number'] = details[-1]

# Get all producer details.
crop = front_page.within_bbox(producer_area)
details = crop.extract_text().split('\n')
producer = details[1]
street = details[2]
parts = details[3].split(',')
city = parts[0]
state, zipcode = tuple(parts[-1].strip().split(' '))
address = ','.join([street, details[3]])
obs['producer'] = producer
obs['producer_address'] = address
obs['producer_street'] = street
obs['producer_city'] = city
obs['producer_state'] = state
obs['producer_zipcode'] = zipcode
obs['producer_license_number'] = details[-1]

# Optional: Get the image data.
# image_index = coa_parameters['coa_image_index']
# obs['image_data'] = parser.get_pdf_image_data(report.pages[0], image_index)
obs['images'] = []

# Get the sample details.
crop = front_page.within_bbox(sample_details_area)
details = crop.extract_text()
details = re.split('\n|' + '|'.join(sample_details_fields), details)
product_name = details[0]
index = 0
for i, detail in enumerate(details[1:]):
    if detail:
        field = sample_details_fields[index]
        key = coa_fields[field]
        obs[key] = detail.replace(':', '').strip()
        index += 1  

# Get the `analyses` and `{analysis}_status`.
analyses = []
table = report.pages[0].extract_table()
for rows in table:
    for row in rows:
        cells = row.split('\n')
        for cell in cells:
            parts = cell.split(':')
            key = snake_case(parts[0])
            try:
                value = strip_whitespace(parts[1])
            except IndexError:
                continue
            field = coa_fields.get(key, key)
            obs[field] = value.lower()
            if field != 'status':
                analysis = field.replace('_status', '')
                analyses.append(analysis)

# FIXME: Identify `{analysis}_method`s.
# Also find the times for all of the tests and find when all
# tests were completed. Future work: record finish time of analyses.
methods = []
tested_at = []
for page in report.pages[1:]:
    text = page.extract_text()
    parts = text.split('Analysis Method/SOP')
    for part in parts[1:]:
        method = part.split('\n')[0].replace(':', '').strip()
        method = method.split('Date/Time')[0] # <-- Hot-fix.
        methods.append(method)
    try:
        parts = text.split('Analyzed:')
        date = parts[1].split('\n')[0].strip()
        tested_at.append(pd.to_datetime(date))
    except:
        pass
date_tested = max(tested_at).isoformat()

# Get all of the `results` rows.
all_rows = []
for page in report.pages[1:]:
    crop = page.within_bbox(page_area)
    rows = parser.get_page_rows(crop)
    for row in rows:
        if row in all_rows:
            pass
        else:
            all_rows.append(row)

# Iterate over all rows to get the `results` rows
# seeing if row starts with an analysis or analyte.
results = []
current_analysis = None
for row in all_rows:

    # Identify the analysis.
    analysis = current_analysis
    for i, name in enumerate(analysis_names):
        if name in row:
            analysis = standard_analyses[i]
            break
    if analysis != current_analysis:
        current_analysis = analysis
        continue

    # Skip detail rows.
    detail_row = False
    for skip_value in skip_values:
        if row.startswith(skip_value):
            detail_row = True
            break
    if detail_row:
        continue

    # Get the analysis details.
    analysis_details = lab_analyses[current_analysis]
    columns = analysis_details['columns']
    double_column = analysis_details.get('double_column')

    # Get the result!
    values = row
    for replacement in coa_replacements:
        values = values.replace(replacement['text'], replacement['key'])
    values = values.split(' ')
    values = [x for x in values if x]
    result = {'analysis': analysis}

    # FIXME: Skip the analysis title here.
    if snake_case(values[0]) == current_analysis:
        continue

    # FIXME: There is probably a better way to do this.
    # This may exclude results by accident.
    if len(values) < len(columns):
        continue

    # FIXME: Hot-fix for `total_terpenes`.
    if values[0] == 'Total' and values[1] == 'Terpenes':
        values = ['Total Terpenes'] + values[2:]

    # FIXME: Hot-fix for `mycotoxins`.
    if values[0] == 'aflatoxin':
        values.insert(2, 20)

    # Split the row if double_column.
    # Future work: This code could probably be refactored.
    if double_column and len(values) > len(columns):
        multi_part = split_list(values, int(len(values) / 2))
        entry = augment_analyte_result(result, columns, multi_part[0])
        if entry['name'] != 'mgtog' and entry['name'] != 'action':
            results.append(entry)
        entry = augment_analyte_result(result, columns, multi_part[1])
        if entry['name'] != 'mgtog' and entry['name'] != 'action':
            results.append(entry)
    else:
        entry = augment_analyte_result(result, columns, values)
        if entry['name'] != 'mgtog' and entry['name'] != 'action':
            results.append(entry)
    
    # FIXME: Add `units` to results!

# Finish data collection with a freshly minted sample ID.
obs['sample_id'] = create_sample_id(
    private_key=producer,
    public_key=product_name,
    salt=date_tested,
)

# Data aggregation.
obs['analyses'] = analyses
obs['date_tested'] = date_tested
obs['methods'] = methods
obs['product_name'] = product_name
obs['results'] = results

# Turn dates to ISO format.
date_columns = [x for x in obs.keys() if x.startswith('date')]
for date_column in date_columns:
    try:
        obs[date_column] = pd.to_datetime(obs[date_column]).isoformat()
    except:
        pass

# TODO: Lowercase `results` `status`.

# TODO: Standardize `results` `units`.

# Future work: Standardize the `product_type`.

# Future work: Attempt to identify `strain_name` from `product_name`.

# Optional: Calculate THC to CBD ratio.

# Optional: Calculate terpene ratios:
# - beta-pinene to d-limonene ratio
# - humulene to caryophyllene
# - linalool and myrcene? (Research these!)

# Optional: Calculate terpinenes total.
# analytes = ['alpha_terpinene', 'gamma_terpinene', 'terpinolene', 'terpinene']
# compounds = sum_columns(compounds, 'terpinenes', analytes, drop=False)

# Optional: Sum `nerolidol`s and `ocimene`s.

# Optional: Calculate total_cbg, total_thcv, total_cbc, etc.

# Save the observation!
observation = {**lab, **obs}
data = pd.DataFrame([observation])
data.to_excel('coa-data.xlsx')
