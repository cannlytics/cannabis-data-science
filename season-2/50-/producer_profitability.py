"""
Analyzing the Profitability of High-Quality Cannabis in Washington State
Cannabis Data Science Meetup Group
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 1/11/2022
Updated: 1/26/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: Using data on analyses performed by labs in Washington State,
this script analyzes the potential profitability of growing high-quality
cannabis and manufacturing high-concentration processed products.
"""
# Standard imports.
import json
from typing import Any, List, Optional

# External imports.
from dateutil import relativedelta
from dotenv import dotenv_values
# from fredapi import Fred
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

# Internal imports.
from utils import (
    format_billions,
    format_millions,
)

#------------------------------------------------------------------------------
# Perform housekeeping and define useful functions.
#------------------------------------------------------------------------------

# Define format for all plots.
# Pro tip: print(plt.rcParams)
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
})

#------------------------------------------------------------------------------
# Read in the laboratory and licensee data.
#------------------------------------------------------------------------------

datatypes = {
    'lab_result_id' : 'string',
    'intermediate_type' : 'category',
    'status' : 'category',
    'global_for_inventory_id': 'string',
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
    # TODO: Will add geocoordinates to augmented lab results.
    # 'latitude': 'float',
    # 'longitude': 'float',
    'license_type': 'string',
    'cannabinoid_status' : 'category',
    'cannabinoid_d9_thca_percent': 'float',
    'cannabinoid_d9_thca_mg_g' : 'float',
    'cannabinoid_d9_thc_percent' : 'float',
    'cannabinoid_d9_thc_mg_g' : 'float',
    'cannabinoid_d8_thc_percent' : 'float',
    'cannabinoid_d8_thc_mg_g' : 'float',
    'cannabinoid_cbd_percent' : 'float',
    'cannabinoid_cbd_mg_g' : 'float',
    'cannabinoid_cbda_percent' : 'float',
    'cannabinoid_cbda_mg_g' : 'float',
    'cannabinoid_cbdv_percent' : 'float',
    'cannabinoid_cbg_percent' : 'float',
    'cannabinoid_cbg_mg_g' : 'float',
    # 'solvent_status' : 'category',
    # 'solvent_acetone_ppm' : 'float16',
    # 'solvent_benzene_ppm' : 'float16',
    # 'solvent_butanes_ppm' : 'float16',
    # 'solvent_chloroform_ppm' : 'float16',
    # 'solvent_cyclohexane_ppm' : 'float16',
    # 'solvent_dichloromethane_ppm' : 'float16',
    # 'solvent_ethyl_acetate_ppm' : 'float16',
    # 'solvent_heptane_ppm' : 'float16',
    # 'solvent_hexanes_ppm' : 'float16',
    # 'solvent_isopropanol_ppm' : 'float16',
    # 'solvent_methanol_ppm' : 'float16',
    # 'solvent_pentanes_ppm' : 'float16',
    # 'solvent_propane_ppm' : 'float16',
    # 'solvent_toluene_ppm' : 'float16',
    # 'solvent_xylene_ppm' : 'float16',
    # 'foreign_matter' : 'bool',
    # 'foreign_matter_stems': 'float16',
    # 'foreign_matter_seeds': 'float16',
    # 'microbial_status' : 'category',
    # 'microbial_bile_tolerant_cfu_g' : 'float16',
    # 'microbial_pathogenic_e_coli_cfu_g' : 'float16',
    # 'microbial_salmonella_cfu_g' : 'float16',
    # 'moisture_content_percent' : 'float16',
    # 'moisture_content_water_activity_rate' : 'float16',
    # 'mycotoxin_status' : 'category',
    # 'mycotoxin_aflatoxins_ppb' : 'float16',
    # 'mycotoxin_ochratoxin_ppb' : 'float16',
    # 'thc_percent' : 'float16',
    # 'notes' : 'float32',
    # 'testing_status' : 'category',
    # 'type' : 'category',
    # 'external_id' : 'string',
}
date_columns = ['created_at']
columns = list(datatypes.keys()) + date_columns

# Read augmented lab results.
file_name = 'augmented-washington-state-lab-results'
results = pd.read_csv(
    f'D:/leaf-data/augmented/{file_name}.csv',
    index_col='lab_result_id',
    dtype=datatypes,
    usecols=columns,
    parse_dates=date_columns,
)

#------------------------------------------------------------------------------
# Augment with geocoded licensee data.
#------------------------------------------------------------------------------

# Read the licensees data.
licensees_file_name = 'augmented-washington-state-licensees'
licensee_fields = {
    'code': 'string',
    'type': 'string',
    'latitude': 'string',
    'longitude': 'string',
    # 'global_id' : 'string',
    # 'name': 'string',
    # 'address1': 'string',
    # 'address2': 'string',
    # 'city': 'string',
    # 'state_code': 'string',
    # 'postal_code': 'string',
}
licensees = pd.read_csv(
    f'D:/leaf-data/augmented/{licensees_file_name}.csv',
    usecols=list(licensee_fields.keys()),
    dtype=licensee_fields,
)

# Combine the data sets.
geocoded_results = pd.merge(
    left=results,
    right=licensees,
    how='left',
    left_on='code',
    right_on='code'
)
geocoded_results = geocoded_results.loc[
    (~geocoded_results.longitude.isnull()) &
    (~geocoded_results.latitude.isnull())
]

#------------------------------------------------------------------------------
# Create supplementary statistics.
#------------------------------------------------------------------------------

# Add a time column.
results['date'] = pd.to_datetime(results['created_at'])

# Calculate total cannabinoids.
cannabinoids_wa = [
    'cannabinoid_d9_thca_percent',
    'cannabinoid_d9_thc_percent',
    'cannabinoid_d8_thc_percent',
    'cannabinoid_cbd_percent',
    'cannabinoid_cbda_percent',
    'cannabinoid_cbdv_percent',
    'cannabinoid_cbg_percent',
    # FIXME: Missing from augmented lab results.
    # 'cannabinoid_cbc_percent',
    # 'cannabinoid_cbn_percent',
    # 'cannabinoid_cbga_percent',
    # 'cannabinoid_thcv_percent',

]
results['total_cannabinoids'] = results[cannabinoids_wa].sum(axis=1)

# Remove values with nonsensical cannabinoid concentrations.
results = results[results['total_cannabinoids'] <= 100]


#------------------------------------------------------------------------------
# Visualize the data.
#------------------------------------------------------------------------------

license_types = [
    'state',
    'cultivator',
    'production',
    'dispensary',
    'lab',
    'cultivator_production',
    'tribe',
    'co-op',
    'transporter'
]

# Calculate statistics for each licensee, one by one. (Slow!)
concs = []
stds = []
variances = []
total_samples = []
for index, licensee in licensees.iterrows():
    licensee_results = results.loc[results['code'] == licensee['code']]
    avg = licensee_results['total_cannabinoids'].mean(skipna=True)
    std = licensee_results['total_cannabinoids'].std(skipna=True)
    var = licensee_results['total_cannabinoids'].var(skipna=True)
    total = len(licensee_results)
    if avg < 0:
        avg = 0
    concs.append(avg)
    stds.append(std)
    variances.append(var)
    total_samples.append(total)
    print(index, licensee['code'], 'total results:', total)

licensees['avg_concentration'] = pd.Series(concs)
licensees['std_deviation'] = pd.Series(stds)
licensees['variance'] = pd.Series(variances)
licensees['total_samples'] = pd.Series(total_samples)

# Ignore licensees with no results.
producers = licensees.loc[licensees['total_samples'] > 0]

# Box Plot
plt.figure(figsize=(12, 8))
results['year'] = results['date'].dt.year
# results['month'] = results['date'].dt.strftime('%Y-%M')
# results['month'] = results['date'].dt.strftime('%B')
results['month'] = results['date'].dt.month
ax = sns.boxplot(
    x='month',
    y='total_cannabinoids',
    # hue='intermediate_type',
    data=results.loc[
        (results['intermediate_type'] == 'flower_lots')
        # (results['intermediate_type'] == 'ethanol_concentrate')
        # ~results.total_cannabinoids.isnull()
    ],
    palette='Set3',
)
ax = sns.swarmplot(
    x='month',
    y='total_cannabinoids',
    data=results.loc[
        (results['intermediate_type'] == 'flower_lots')
    ],
    color='.25',
)
plt.title('Total Cannabinoids by Month of the Year in\nWashington State Cannabis Flower')
plt.ylabel('Total Cannabinoids (%)')
plt.xlabel('Month of the Year')
plt.savefig(
    'total-cannabinoids-by-month-wa.png',
    format='png',
    dpi=300,
    facecolor='white'
)
plt.show()

# Bubble Plot
plt.figure(figsize=(12, 8))
ax = sns.scatterplot(
    producers['avg_concentration'],
    producers['variance'],
    hue=producers['total_samples'],
    color='darkblue',
    alpha=0.5,
    s=0.25 * 2000,
)
plt.xlabel('Avg Concentration (%)')
plt.ylabel('Total Samples')
plt.show()

# Scatter Plot
# FIXME: Scatter Plot of latitude and longitude of licensees
# sns.scatterplot(
#     data=licensees,
#     x='longitude',
#     y='latitude',
#     legend=False,
#     # size='',
#     # sizes=(20, 2000)
# )
# plt.show()

# sns.scatterplot(
#     x='avg_concentration',
#     y='total_samples',
#     hue='std_deviation',
#     data=licensees.loc[
#         (licensees.type == 'cultivator') |
#         (licensees.type == 'cultivator_production')
#     ]
#     # legend=True,
#     # size='',
#     # sizes=(20, 2000)
# )
# plt.show()


#------------------------------------------------------------------------------
# TODO: Perform analysis.
#------------------------------------------------------------------------------

# Plot the density of total cannabinoids in flower for the growers testing the
# most samples.


# Future work:

# Estimate the quantity sold based on quality control requirements and assuming
# that all products that pass quality control testing are sold.

# Regress estimated monthly quantity sold against the monthly average total
# cannabinoids (in flower and in concentrates).


# Plot monthly estimated quantity sold against the average total cannabinoids.


#------------------------------------------------------------------------------
# TODO: Pull data from the MA CCC Open Data for Comparison
# https://masscannabiscontrol.com/open-data/data-catalog/
#------------------------------------------------------------------------------



