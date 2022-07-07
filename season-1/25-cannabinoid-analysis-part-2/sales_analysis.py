"""
Cannabis Sales Analysis in Washington State 2018 - 2020
Cannabis Data Science

Authors:
    Keegan Skeate <keegan@cannlytics.com>
    Charles Rice
Created: 8/10/2021
Updated: 8/10/2021
License: GPLv3+
Data Source:
    https://lcb.app.box.com/s/fnku9nr22dhx04f6o646xv6ad6fswfy9?page=1
"""
from datetime import datetime
import pandas as pd

#------------------------------------------------------------------------------
# Aggregate variables.
#------------------------------------------------------------------------------

col_dtypes = {
    'global_id' : 'string',
    # '#mme_id' : 'category',
    # 'external_id' : 'string',
    # 'inventory_id' : 'string',
    'status' : 'category',
    #'testing_status' : 'category',
    #'batch_id' : 'string',
    #'parent_lab_result_id' : 'string',
    #'og_parent_lab_result_id' : 'string',
    #'copied_from_lab_id' : 'string',
    'type' : 'category',
    'cannabinoid_d9_thca_percent': 'float16',
    'cannabinoid_d9_thca_mg_g' : 'float16', #if you are not using Dask change this to float16
    'cannabinoid_d9_thc_percent' : 'float16', #if you are not using Dask change this to float16
    'cannabinoid_d9_thc_mg_g' : 'float16', #if you are not using Dask change this to float16
    'cannabinoid_d8_thc_percent' : 'float16', #if you are not using Dask change this to float16
    'cannabinoid_d8_thc_mg_g' : 'float16', #if you are not using Dask change this to float16
    'cannabinoid_cbd_percent' : 'float16', #if you are not using Dask change this to float16
    'cannabinoid_cbd_mg_g' : 'float16', #if you are not using Dask change this to float16
    'cannabinoid_cbda_percent' : 'float16', #if you are not using Dask change this to float16
    'cannabinoid_cbda_mg_g' : 'float16', #if you are not using Dask change this to float16
    'cannabinoid_cbdv_percent' : 'float16', #if you are not using Dask change this to float16
    'cannabinoid_cbg_percent' : 'float16', #if you are not using Dask change this to float16
    'cannabinoid_cbg_mg_g' : 'float16', #if you are not using Dask change this to float16
    'microbial_bile_tolerant_cfu_g' : 'float16', #if you are not using Dask change this to float16
    'microbial_pathogenic_e_coli_cfu_g' : 'float16', #if you are not using Dask change this to float16
    'microbial_salmonella_cfu_g' : 'float16', #if you are not using Dask change this to float16
    'mycotoxin_status' : 'category',
    'mycotoxin_editor' : 'string',
    'mycotoxin_aflatoxins_ppb' : 'float16', #if you are not using Dask change this to float16
    'mycotoxin_ochratoxin_ppb' : 'float16', #if you are not using Dask change this to float16
    'thc_percent' : 'float16', #if you are not using Dask change this to float16
    'intermediate_type' : 'category',
    'moisture_content_water_activity_rate' : 'float16', #if you are not using Dask change this to float16
    'moisture_content_percent' : 'float16', #if you are not using Dask change this to float16
}

date_cols = [
    'created_at',
    #'deleted_at',
    'updated_at',
    #'tested_at',
]
cols = list(col_dtypes.keys()) + date_cols

# Read in the lab result data.
file_name = '../.datasets/LabResults_0.csv'
lab_data = pd.read_csv(
    file_name,
    sep='\t',
    encoding='utf-16',
    # nrows=10000, # FIXME: Read in all the data!
    usecols=cols,
    dtype=col_dtypes,
    parse_dates=date_cols,
    skipinitialspace=True
)

# TODO: Combine sales and inventories data sets.
# file_name = f'{directory}\Licensees_0\wa_licensee_data.xlsx'
# license_data = pd.read_excel(file_name)
# data = pd.merge(
#     left=lab_data,
#     right=license_data,
#     how='left',
#     left_on='for_mme_id',
#     right_on='global_id'
# )

# Restric the timeframe to 2020.
start_date = datetime(2020, 1, 1)
end_date = datetime(2020, 12, 31)
criterion = (lab_data.updated_at >= start_date) & (lab_data.updated_at <= end_date)
data = lab_data.loc[criterion]
del globals()['lab_data']

#------------------------------------------------------------------------------
# Calculate moisture corrected THC concentration for each sample.
#------------------------------------------------------------------------------

# TODO: Isolate flower data.
flower_data = data.loc[(data.intermediate_type == 'flower') |
                       (data.intermediate_type == 'flower_lots')]

# Calculate moisture corrected cannabinoid concentration.
cannabinoids = [
    'cannabinoid_d9_thca_percent',
    'cannabinoid_d9_thc_percent',
]
for cannabinoid in cannabinoids:
    flower_data['moisture_corrected_' + cannabinoid] = flower_data[cannabinoid] / (1 + flower_data['moisture_content_percent'])

# Exclude outliers?
upper_bound = flower_data.moisture_corrected_cannabinoid_d9_thc_percent.quantile(0.95)
flower_data = flower_data.loc[flower_data.moisture_corrected_cannabinoid_d9_thc_percent < upper_bound]

# Plot the data
flower_data.moisture_corrected_cannabinoid_d9_thc_percent.hist()

#------------------------------------------------------------------------------
# TODO: Calculate sales per lab result.
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Estimate a regressions of sales per lab result on
# the lab result's moisture corrected THC concentration.
#------------------------------------------------------------------------------

