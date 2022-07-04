"""
Combine lab results, licensees, inventories, inventory types, and strains data.
Cannabis Data Science Meetup Group
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 1/11/2022
Updated: 1/19/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script combines relevant fields from the licensees, inventories,
inventory types, and strains datasets with the lab results data.

Data Sources:

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd?page=1
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd?page=2

    - Leaf Data Systems Data Guide:
    https://lcb.wa.gov/sites/default/files/publications/Marijuana/traceability/WALeafDataSystems_UserManual_v1.37.5_AddendumC_LicenseeUser.pdf

"""
# Standard imports.
import gc

# External imports.
import pandas as pd

# Internal imports.
from utils import get_number_of_lines

#------------------------------------------------------------------------------
# Read lab results data.
#------------------------------------------------------------------------------

def read_lab_results(
        columns=None,
        fields=None,
        date_columns=None,
        nrows=None,
        data_dir='../.datasets',
):
    """
    1. Read Leaf lab results.
    2. Sort the data, removing null observations.
    3. Define a lab ID for each observation and remove attested lab results.
    """
    shards = []
    lab_datasets = ['LabResults_0', 'LabResults_1', 'LabResults_2']
    for dataset in lab_datasets:
        lab_data = pd.read_csv(
            f'{data_dir}/{dataset}.csv',
            sep='\t',
            encoding='utf-16',
            usecols=columns,
            dtype=fields,
            parse_dates=date_columns,
            nrows=nrows,
        )
        shards.append(lab_data)
        del lab_data
        gc.collect()
    data = pd.concat(shards)
    del shards
    gc.collect()
    data.dropna(subset=['global_id'], inplace=True)
    # data.set_index('global_id', inplace=True)
    data.sort_index(inplace=True)
    data['lab_id'] = data['global_id'].map(lambda x: x[x.find('WAL'):x.find('.')])
    data = data.loc[data.lab_id != '']
    return data


#------------------------------------------------------------------------------
# Combine lab result data with inventory data.
#------------------------------------------------------------------------------

# Define necessary lab result fields.
lab_result_fields = {
    'global_id' : 'string',
    'global_for_inventory_id': 'string'
}

# Read lab result fields necessary to connect with inventory data.
lab_results = read_lab_results(
    columns=list(lab_result_fields.keys()),
    fields=lab_result_fields,
)

# Save initial enhanced lab results.
lab_results.to_csv('../.datasets/enhanced_lab_results.csv')

# Define inventory fields.
inventory_fields = {
    'global_id' : 'string',
    'inventory_type_id': 'string',
    'strain_id': 'string',
}
inventory_columns = list(inventory_fields.keys())

# Define chunking parameters.
# inventory_type_rows = get_number_of_lines('../.datasets/Inventories_0.csv')
inventory_row_count = 129_920_072
chunk_size = 30_000_000
read_rows = 0
skiprows = None
datatypes = {
    'global_id' : 'string',
    'global_for_inventory_id': 'string',
    'lab_id': 'string',
    'inventory_type_id': 'string',
    'strain_id': 'string',
}

# Read in a chunk at a time, match with lab results, and save the data.
while read_rows < inventory_row_count:

    # Define the chunk size.
    if read_rows:
        skiprows = [i for i in range(1, read_rows)]

    # 1. Open enhanced lab results.
    lab_results = pd.read_csv(
        '../.datasets/lab_results_with_ids.csv',
        # index_col='global_id',
        dtype=datatypes
    )

    # 2. Read chunk of inventories.
    inventories = pd.read_csv(
        '../.datasets/Inventories_0.csv',
        sep='\t',
        encoding='utf-16',
        usecols=inventory_columns,
        dtype=inventory_fields,
        skiprows=skiprows,
        nrows=chunk_size,
    )

    # 3. Merge inventories with enhanced lab results.
    inventories.rename(columns={'global_id': 'inventory_id'}, inplace=True)
    lab_results = pd.merge(
        left=lab_results,
        right=inventories,
        how='left',
        left_on='global_for_inventory_id',
        right_on='inventory_id',
    )

    # Remove overlapping columns
    try:
        new_entries = lab_results[['inventory_type_id_y', 'strain_id_x']]
        lab_results = lab_results.combine_first(new_entries)
        lab_results.rename(columns={
            'inventory_type_id_x': 'inventory_type_id',
            'strain_id_x': 'strain_id',
        }, inplace=True)
    except KeyError:
        pass
    extra_columns = ['inventory_id', 'Unnamed: 0', 'inventory_type_id_y',
                     'strain_id_y']
    lab_results.drop(extra_columns, axis=1, inplace=True, errors='ignore')

    # 4. Save lab results enhanced with IDs.
    lab_results.to_csv('../.datasets/lab_results_with_ids.csv')
    read_rows += chunk_size
    print('Read:', read_rows)

del new_entries
del inventories
gc.collect()


#------------------------------------------------------------------------------
# Combine lab result data with inventory type data.
#------------------------------------------------------------------------------

# Get only the results with
results_with_ids = lab_results[~lab_results['inventory_type_id'].isna()]

# Read in inventory type names.
inventory_type_names = pd.read_csv(
    '../.datasets/inventory_type_names.csv',
    # index_col='global_id',
    dtype={
        'global_id' : 'string',
        'inventory_name': 'string',
    }
)

# Merge enhanced lab results with inventory type names.
results_with_ids = pd.merge(
    left=results_with_ids,
    right=inventory_type_names,
    how='left',
    left_on='inventory_type_id',
    right_on='global_id',
)
results_with_ids.rename(columns={'global_id_x': 'global_id'}, inplace=True)
results_with_ids.drop(['global_id_y'], axis=1, inplace=True, errors='ignore')

# Save the lab results enhanced with inventory names.
results_with_ids.to_csv('../.datasets/lab_results_with_inventory_names.csv')


#------------------------------------------------------------------------------
# Combine lab result data with strain data.
#------------------------------------------------------------------------------

# Define strain fields.
strain_fields = {
    'global_id': 'string',
    'name': 'string',
}
strain_columns = list(strain_fields.keys())

# Read in strain data.
strains = pd.read_csv(
    '../.datasets/Strains_0.csv',
    sep='\t',
    encoding='utf-16',
    dtype=strain_fields,
    usecols=strain_columns,
)

# Merge enhanced lab results with strain data.
strains.rename(columns={
    'global_id': 'strain_id',
    'name': 'strain_name',
}, inplace=True)
results_with_ids = pd.merge(
    left=results_with_ids,
    right=strains,
    how='left',
    left_on='strain_id',
    right_on='strain_id',
)
results_with_ids.rename(columns={'global_id_x': 'global_id'}, inplace=True)
results_with_ids.drop(['global_id_y'], axis=1, inplace=True, errors='ignore')

# Save the extra lab results fields.
results_with_ids.to_csv('../.datasets/lab_results_with_strain_names.csv')

#------------------------------------------------------------------------------
# TODO: Combine lab result data with licensee data.
#------------------------------------------------------------------------------

# Get only the inventory names from the inventory types data.
# from get_inventory_type_names import get_inventory_type_names
# get_inventory_type_names()

# Read lab result fields necessary to connect with inventory data.
# lab_result_fields = {
#     'global_id' : 'string',
#     'mme_id': 'string'
# }
# lab_results = read_lab_results(
#     columns=list(lab_result_fields.keys()),
#     fields=lab_result_fields,
# )

# Add code variable to lab results with IDs.
results_with_ids['code'] = results_with_ids['global_for_inventory_id'].map(
    lambda x: x[x.find('WA'):x.find('.')]
).str.replace('WA', '')

# Specify the licensee fields.
licensee_fields = {
    'global_id' : 'string',
    'code': 'string',
    'name': 'string',
    'type': 'string',
    'address1': 'string',
    'address2': 'string',
    'city': 'string',
    'state_code': 'string',
    'postal_code': 'string',
}
licensee_date_fields = [
    'created_at', # No records if issued before 2018-02-21.
]
licensee_columns = list(licensee_fields.keys()) + licensee_date_fields

# # Read in the licensee data.
licensees = pd.read_csv(
    '../.datasets/Licensees_0.csv',
    sep='\t',
    encoding='utf-16',
    usecols=licensee_columns,
    dtype=licensee_fields,
    parse_dates=licensee_date_fields,
)

# Format the licensees data.
licensees.rename(columns={
    'global_id': 'mme_id',
    'created_at': 'license_created_at',
    'type': 'license_type',
}, inplace=True)

# Combine the data sets.
results_with_ids = pd.merge(
    left=results_with_ids,
    right=licensees,
    how='left',
    left_on='code',
    right_on='code'
)
results_with_ids.rename(columns={'global_id_x': 'global_id'}, inplace=True)
results_with_ids.drop(['global_id_y'], axis=1, inplace=True, errors='ignore')

# Save lab results enhanced with additional fields.
results_with_ids.to_csv('../.datasets/lab_results_with_licensee_data.csv')


#------------------------------------------------------------------------------
# Combine lab result data with enhanced lab results data.
#------------------------------------------------------------------------------

# Read in results with IDs.
results_with_ids = pd.read_csv(
    '../.datasets/lab_results_with_licensee_data.csv',
    dtype = {
        'global_id': 'string',
        'global_for_inventory_id': 'string',
        'lab_result_id': 'string',
        'inventory_type_id': 'string',
        'lab_id': 'string',
        'strain_id': 'string',
        'inventory_name': 'string',
        'strain_name': 'string',
        'code': 'string',
        'mme_id': 'string',
        'license_created_at': 'string',
        'name': 'string',
        'address1': 'string',
        'address2': 'string',
        'city': 'string',
        'state_code': 'string',
        'postal_code': 'string',
        'license_type': 'string',
    },
)

# Read all lab results fields with any valuable data.
lab_result_fields = {
    'global_id' : 'string',
    'intermediate_type' : 'category',
    'status' : 'category',
    'cannabinoid_status' : 'category',
    'cannabinoid_d9_thca_percent': 'float16',
    'cannabinoid_d9_thca_mg_g' : 'float16',
    'cannabinoid_d9_thc_percent' : 'float16',
    'cannabinoid_d9_thc_mg_g' : 'float16',
    'cannabinoid_d8_thc_percent' : 'float16',
    'cannabinoid_d8_thc_mg_g' : 'float16',
    'cannabinoid_cbd_percent' : 'float16',
    'cannabinoid_cbd_mg_g' : 'float16',
    'cannabinoid_cbda_percent' : 'float16',
    'cannabinoid_cbda_mg_g' : 'float16',
    'cannabinoid_cbdv_percent' : 'float16',
    'cannabinoid_cbg_percent' : 'float16',
    'cannabinoid_cbg_mg_g' : 'float16',
    'solvent_status' : 'category',
    'solvent_acetone_ppm' : 'float16',
    'solvent_benzene_ppm' : 'float16',
    'solvent_butanes_ppm' : 'float16',
    'solvent_chloroform_ppm' : 'float16',
    'solvent_cyclohexane_ppm' : 'float16',
    'solvent_dichloromethane_ppm' : 'float16',
    'solvent_ethyl_acetate_ppm' : 'float16',
    'solvent_heptane_ppm' : 'float16',
    'solvent_hexanes_ppm' : 'float16',
    'solvent_isopropanol_ppm' : 'float16',
    'solvent_methanol_ppm' : 'float16',
    'solvent_pentanes_ppm' : 'float16',
    'solvent_propane_ppm' : 'float16',
    'solvent_toluene_ppm' : 'float16',
    'solvent_xylene_ppm' : 'float16',
    'foreign_matter' : 'bool',
    'foreign_matter_stems': 'float16',
    'foreign_matter_seeds': 'float16',
    'microbial_status' : 'category',
    'microbial_bile_tolerant_cfu_g' : 'float16',
    'microbial_pathogenic_e_coli_cfu_g' : 'float16',
    'microbial_salmonella_cfu_g' : 'float16',
    'moisture_content_percent' : 'float16',
    'moisture_content_water_activity_rate' : 'float16',
    'mycotoxin_status' : 'category',
    'mycotoxin_aflatoxins_ppb' : 'float16',
    'mycotoxin_ochratoxin_ppb' : 'float16',
    'thc_percent' : 'float16',
    'notes' : 'float32',
    'testing_status' : 'category',
    'type' : 'category',
    'external_id' : 'string',
}
lab_result_date_columns = ['created_at', 'updated_at', 'received_at',]
lab_result_columns = list(lab_result_fields.keys()) + lab_result_date_columns
complete_lab_results = read_lab_results(
    columns=lab_result_columns,
    fields=lab_result_fields,
    date_columns=None,
)

# Merge lab results with the complete lab results data.
complete_lab_results.rename(columns={
    'global_id': 'lab_result_id',
    'type': 'sample_type',
}, inplace=True)
results_with_ids = pd.merge(
    left=results_with_ids,
    right=complete_lab_results,
    how='left',
    left_on='global_id',
    right_on='lab_result_id',
)
results_with_ids.drop(['global_id'], axis=1, inplace=True, errors='ignore')

# Save the complete lab results data.
results_with_ids.to_csv('../.datasets/lab_results_complete.csv')
