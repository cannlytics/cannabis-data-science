"""
Analyzing the Profitability of Processing Cannabis in Washington State
Cannabis Data Science Meetup Group
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 1/11/2022
Updated: 1/19/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: Using data on analyses performed by labs in Washington State,
this script analyzes the potential profitability of manufacturing processed
cannabis products.

Data Sources:

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd?page=1
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd?page=2

    - Leaf Data Systems Data Guide:
    https://lcb.wa.gov/sites/default/files/publications/Marijuana/traceability/WALeafDataSystems_UserManual_v1.37.5_AddendumC_LicenseeUser.pdf

Resources:

    - Introduction to ARCH Models
    https://arch.readthedocs.io/en/latest/univariate/introduction.html


    - How to Model Volatility with ARCH and GARCH for Time Series Forecasting in Python
    https://machinelearningmastery.com/develop-arch-and-garch-models-for-time-series-forecasting-in-python/

"""

# External imports.
import matplotlib.pyplot as plt
# from matplotlib.ticker import FuncFormatter
import pandas as pd
from pandas.plotting import table
import seaborn as sns
# import statsmodels.api as sm

# Internal imports.
# from combine_data import read_lab_results
# from utils import (
#     end_of_month,
#     end_of_year,
#     format_billions,
#     format_millions,
#     format_thousands,
#     sorted_nicely,
# )

#------------------------------------------------------------------------------
# Perform housekeeping and define useful functions.
#------------------------------------------------------------------------------

# Define format for all plots.
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
    # 'thc_percent' : 'float16',
    # 'notes' : 'float32',
    # 'testing_status' : 'category',
    # 'type' : 'category',
    # 'external_id' : 'string',
}
date_columns = ['created_at']
columns = list(datatypes.keys()) + date_columns

# Read enhanced lab results.
results = pd.read_csv(
    '../.datasets/lab_results_completed.csv',
    index_col='lab_result_id',
    dtype=datatypes,
    usecols=columns,
    parse_dates=date_columns,
)


#------------------------------------------------------------------------------
# Create supplementary statistics.
#------------------------------------------------------------------------------

# Identify processors as tests with residual solvent screening
# or solventless products.
products = results[
    (~results['solvent_status'].isna()) |
    (results['intermediate_type'] == 'food_grade_solvent_concentrate') |
    (results['intermediate_type'] == 'non-solvent_based_concentrate')
]
products = products[
    (~products['intermediate_type'].isna()) &
    (products['intermediate_type'] != 'flower_lots')
]

# Create a time column at the time when the product was tested.
products['date'] = pd.to_datetime(products['created_at'])

# Remove values with nonsensical THCA.
products = products[products['cannabinoid_d9_thca_percent'] <= 100]

# Calculate total cannabinoids.
cannabinoids_wa = [
    'cannabinoid_d9_thca_percent',
    'cannabinoid_d9_thc_percent',
    'cannabinoid_d8_thc_percent',
    # 'cannabinoid_thcv_percent',
    'cannabinoid_cbd_percent',
    'cannabinoid_cbda_percent',
    'cannabinoid_cbdv_percent',
    'cannabinoid_cbg_percent',
    # 'cannabinoid_cbga_percent',
    # 'cannabinoid_cbc_percent',
    # 'cannabinoid_cbn_percent',
]
products['total_cannabinoids'] = products[cannabinoids_wa].sum(axis=1)

# Identify the types of samples each processor produces.

# Look at the breakdown of extraction techniques over time.
plt.figure(figsize=(16,8))
ax1 = plt.subplot(121, aspect='equal')
plot_data = products[['intermediate_type']]
# products.groupby(['intermediate_type']).count()
products.intermediate_type.value_counts().plot(
    kind='pie',
    ax=ax1,
    autopct='%1.1f%%', 
    startangle=90,
    shadow=False,
    legend = False,
    fontsize=14,
    colors = sns.color_palette('tab20'),
    title='Proportion of Analyses by Product Type'
    # y='intermediate_type',
    # labels=df['officer_name'],
)
plt.show()
# ax2 = plt.subplot(122)
# plt.axis('off')
# tbl = table(ax2, df, loc='center')
# tbl.auto_set_font_size(False)
# tbl.set_fontsize(14)
# plt.show()

# Look at the average concentration and standard deviation of total cannabinoids
# for each product type.
types = list(products.intermediate_type.unique())
product_groups = products.groupby(['intermediate_type'])
print('Average concentration:', product_groups.total_cannabinoids.mean())
print('Standard deviation:', product_groups.total_cannabinoids.std())

# Look at the concentrations by cannabinoid, seeing which concentrates
# have activated THC.
for cannabinoid in cannabinoids_wa:
    print('Average concentration:', product_groups[cannabinoid].mean())
    print('Standard deviation:', product_groups[cannabinoid].std())

# Look at the average concentration and standard deviation
# for the 10 processors that test the most samples.
top_10 = products.groupby(['mme_id']).count()
top_10.sort_values(by='mme_id', ascending=False, inplace=True)
top_10 = top_10[:10]
for index, row in top_10.iterrows():
    processor_products = products.loc[products.mme_id == row.id]
    avg_concentration = processor_products.total_cannabinoids.mean()
    std_deviation = processor_products.total_cannabinoids.std()
    print(row.mme_id, 'Avg:', avg_concentration, 'Std. Dev.:', std_deviation)


#------------------------------------------------------------------------------
# Future work: Perform analysis.
#------------------------------------------------------------------------------

# Try to calculate the weight of concentrates sold.

# Estimate the quantity sold based on quality control requirements and assuming
# that all products that pass quality control testing are sold.


# Regress the estimated monthly quantity sold against the monthly average total
# cannabinoids and standard deviation in cannabinoids.


# Plot monthly estimated quantity sold against the average total cannabinoids.


#------------------------------------------------------------------------------
# On Saturday we will:
#    - Forecast the total number of samples each processor will test.
#    - Forecast the average concentration of each processor's products.
#    - Forecast the variance in concentration of products by processors
#      in 2022 and see who has the highest and lowest variances.
#------------------------------------------------------------------------------
