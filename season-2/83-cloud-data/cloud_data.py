"""
Cloud Data
Copyright (c) 2022 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
    Candace O'Sullivan-Sutherland <https://github.com/candy-o>
Created: 9/13/2022
Updated: 9/14/2022
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

"""
from cannlytics.data.coas import CoADoc
from datasets import load_dataset # pip install datasets
import pandas as pd


#-----------------------------------------------------------------------
# Get cloud data!
#-----------------------------------------------------------------------

# Download Raw Garden lab result details dataset from Hugging Face.
dataset = load_dataset('cannlytics/cannabis_tests', 'rawgarden')
details = dataset['details']
assert len(details) > 0
print('Downloaded %i observations.' % len(details))

# Initialize CoADoc.
parser = CoADoc()

# Create custom column order.
column_order = ['sample_hash', 'results_hash']
column_order += list(parser.column_order)
index = column_order.index('product_type') + 1
column_order.insert(index, 'product_subtype')

# Save the data locally with "Details", "Results", and "Values" worksheets.
outfile = 'details.xlsx'
parser.save(details.to_pandas(), outfile)

# Read the values.
values = pd.read_excel(outfile, sheet_name='Values')


#-----------------------------------------------------------------------
# Look at the data!
#-----------------------------------------------------------------------

# Define analytes of interest.
cannabinoids = [
    'thca',
    'delta_9_thc',
    'delta_8_thc',
    'thcv',
    'cbda',
    'cbd',
    'cbdv',
    'cbga',
    'cbg',
    'cbn',
    'cbc',
    'thcva',
    'cbdva',
    'cbl',
    'cbca',
]
terpenes = [
    'delta_3_carene',
    'alpha_cedrene',
    'alpha_pinene',
    'alpha_terpineol',
    'beta_myrcene',
    'borneol',
    'camphor',
    'cedrol',
    'cis_nerolidol',
    'fenchol',
    'gamma_terpinene',
    'geranyl_acetate',
    'isoborneol',
    'd_limonene',
    'menthol',
    'ocimene',
    'trans_nerolidol',
    'alpha_bisabolol',
    'alpha_humulene',
    'alpha_terpinene',
    'beta_caryophyllene',
    'beta_pinene',
    'camphene',
    'caryophyllene_oxide',
    'geraniol',
    'eucalyptol',
    'fenchone',
    'gamma_terpineol',
    'guaiol',
    'linalool',
    'p_cymene',
    'pulegone',
    'terpinolene',
    'valencene',
    'isopulegol',
    'beta_ocimene',
    'terpineol',
    'nerolidol',
    'citronellol',
    'alpha_phellandrene',
    'sabinene_hydrate',
]

# Future work: Group by product subtype.
# group = values.groupby('product_subtype').mean()

# TODO: Look at the distribution for the various terpenes.

# Calculate the percent of samples with detects.

# Calculate the average terpene concentration by product subtype.


#-----------------------------------------------------------------------
# Future work: Use the data for analysis.
#-----------------------------------------------------------------------


#-----------------------------------------------------------------------
# TODO: Save the statistics to the cloud!
#-----------------------------------------------------------------------

# from cannlytics.firebase import initialize_firebase, update_document

# # Initialize Firestore.
# database = initialize_firebase(env_file='../../.env')

# # Define the statistics to save.
# summary_stats = {
#     'count': len(values),
# }

# # Save statistics to Firestore.
# ref = 'public/stats/rawgarden'
# update_document(f'{ref}/summary_stats', summary_stats)
