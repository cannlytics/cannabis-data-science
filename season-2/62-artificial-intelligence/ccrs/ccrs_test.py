"""
CCRS Test
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 4/10/2022
Updated: 4/20/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description: This script tests Washington State CCRS data management tools.

Data sources:

    - CCRS PRR All Data Up To 3-12-2022
    https://lcb.app.box.com/s/7pi3wqrmkuo3bh5186s5pqa6o5fv8gbs

Setup:

    1. pip install cannlytics

"""
# Standard imports.
import os

# External imports. Using the Cannlytics package!
from cannlytics.utils.utils import snake_case
from dotenv import dotenv_values
import pandas as pd
import matplotlib.pyplot as plt

# Internal imports. Using the CCRS interface!
from ccrs import CCRS
# from constants import analytes, datasets


# Create a place for your data to live.
DATA_DIR = 'D:\\data\\washington\\CCRS PRR All Data Up To 3-12-2022'
START = '2021-12-01'
END = '2021-03-01'


# Initialize a CCRS client.
config = dotenv_values('../../.env')
os.environ['CANNLYTICS_API_KEY'] = config['CANNLYTICS_API_KEY']
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config['GOOGLE_APPLICATION_CREDENTIALS']
ccrs = CCRS(data_dir=DATA_DIR)


#------------------------------------------------------------------------------
# Manage general data.
#------------------------------------------------------------------------------

# Read licensee data.
licensees = ccrs.read_licensees()

# Read areas data.
areas = ccrs.read_areas()


#------------------------------------------------------------------------------
# Manage inventory data.
#------------------------------------------------------------------------------

# Read inventory data.
inventory = ccrs.read_inventory(limit=100_000)

# Wishlist: Augment with licensee data with licensee_id

# Wishlist: Augment with strain data with strain_id

# Wishlist Augment product data with product_id

# Optional: Explore interesting fields:
# - quantity_on_hand
# - total_cost
# - created_date

# Optional: Count inventory items by date for each licensee?

# Estimate Cost of Goods Sold (CoGS) (Poor data for this metric). 
cogs = (inventory.initial_quantity - inventory.quantity_on_hand) * inventory.total_cost

# Read inventory adjustment data.
adjustments = ccrs.read_inventory_adjustments()

# Wishlist: Merge inventory details
# inventory_adjustments = pd.merge()

# Highlight imperfect system.
lost = adjustments.loc[adjustments.inventory_adjustment_reason == 'Lost']
theft = adjustments.loc[adjustments.inventory_adjustment_reason == 'Theft']
seized = adjustments.loc[adjustments.inventory_adjustment_reason == 'Seizure']
other = adjustments.loc[adjustments.inventory_adjustment_reason == 'Other']

not_found = lost.loc[lost['adjustment_detail'].astype(str).str.contains('not found', case=False)]

#------------------------------------------------------------------------------
# Manage cultivation data.
# Question: How much does it cost to produce a pound in your facility?
#------------------------------------------------------------------------------

# Read plant data.
plants = ccrs.read_plants()

# Wishlist: Augment with strain data.
# StrainId is missing from strain data! And all plant StrainIds are 1...
strains = ccrs.read_strains()

# Wishlist: Augment with area data.
# Area data is missing AreaId.

# Wishlist: Augment with licensee data.
# Licensee data is missing LicenseeId

# TODO: Calculate number of plants by type by day, week, month, year
# for each licensee.
# This may have to be done by looking at created_date and harvest_date.


# TODO: Estimate wholesale sales by licensee_id


# Estimate growing period.
final_states = ['Harvested', 'Drying', 'Sold']
harvested = plants.loc[plants.plant_state.isin(final_states)]
grow_days = (harvested.harvest_date - harvested.created_date).dt.days
grow_days = grow_days.loc[(grow_days > 30) & (grow_days < 365)]
grow_days.describe()
grow_days.hist(bins=100)
plt.show()


# TODO: Estimate a production function (yield per plant).


# # Optional: See who is transfering plants to who.
# # InventoryPlantTransfer_0
# # FromLicenseeId, ToLicenseeId, FromInventoryId, ToInventoryId, TransferDate


# Read plant destruction data.
destructions = ccrs.read_plant_destructions()

# Look at the reasons for destruction.
destructions['destruction_reason'].value_counts().plot(kind='pie')

# Look at contaminants
mites = destructions.loc[destructions.destruction_reason == 'Mites']
contaminated = destructions.loc[destructions.destruction_reason == 'Contamination']

# Plot plants destroyed by mites per day.
mites_by_day = mites.groupby('destruction_date')['plant_id'].count()
mites_by_day.plot()
plt.title('Number of Plants Destroyed by Mites in Washington')
plt.show()

# Plot plants destroyed by contamination per day.
contaminated_by_day = contaminated.groupby('destruction_date')['plant_id'].count()
contaminated_by_day.plot()
plt.title('Number of Contaminated Plants in Washington')
plt.show()

# # TODO: Calculate daily risk of plant death.
# destructions_by_day = destructions.groupby('destruction_date')['plant_id'].count()
# # plants_by_day = 
# # plant_risk = 

# Saturday Morning Statistics teaser:
# Capital asset pricing model (CAPM) or...
# Plant liability asset net total model (PLANTM) ;)


#------------------------------------------------------------------------------
# Manage product data.
#------------------------------------------------------------------------------

# Read product data.
products = ccrs.read_products(limit=100_000)

# Look at products by day by licensee.
products_by_day = products.groupby(['licensee_id', 'created_date'])['name'].count()

# Wishlist: There is a reference to InventoryTypeId but not inventory type data.

# Wishlist: Match with licensee data with licensee_id


#------------------------------------------------------------------------------
# Manage sales data.
#------------------------------------------------------------------------------

# Read sale header data.
sale_headers = ccrs.read_sale_headers()

# Read sale detail data.
sale_details = ccrs.read_sale_details()

# Calculate total price and total tax.
sale_details['total_tax'] = sale_details['sales_tax'] + sale_details['other_tax']
sale_details['total_price'] = sale_details['unit_price'] - abs(sale_details['discount']) + sale_details['total_tax']

sale_details = pd.merge(
    sale_details,
    sale_headers,
    left_on='sale_header_id',
    right_on='sale_header_id',
    how='left',
    validate='m:1',
    suffixes=(None, '_header'),
)

# Calculate total transactions, average transaction, and total sales by retailer.
transactions = sale_details.groupby(['sale_header_id', 'licensee_id'], as_index=False)
transaction_amount = transactions['total_price'].sum()
avg_transaction_amount = transaction_amount.groupby('licensee_id')['total_price'].mean()

# Calculate transactions and sales by day.
daily = sale_details.groupby(['sale_date', 'licensee_id'], as_index=False)
daily_sales = daily['total_price'].sum()
daily_transactions = daily['total_price'].count()
group = ['sale_date', 'licensee_id', 'sale_header_id']
daily_avg_transaction_amount = sale_details.groupby(group, as_index=False)['total_price'].mean()

# TODO: Aggregate statistics by daily and licensee.

# TODO: Calculate year-to-date statistics for each licensee.

# FIXME: Figure out how to connect sale_headers.licensee_id with licensees.license_number?

# TODO: Break down by sale type:
# 'RecreationalRetail', 'RecreationalMedical', 'Wholesale'

# TODO: Try to match sale_items.inventory_id to other details?


#------------------------------------------------------------------------------
# Manage transfer data.
#------------------------------------------------------------------------------

# Read transfer data.
transfers = ccrs.read_transfers()

# TODO: Get list of license numbers / addresses from transers.

# Future work: Look at number of items, etc. for each transfer.


#------------------------------------------------------------------------------
# Manage lab result data.
#------------------------------------------------------------------------------

# Read lab results.
lab_results = ccrs.read_lab_results()

# Note: Sometimes "Not Tested" is a `test_value`.
lab_results['test_value'] = pd.to_numeric(lab_results['test_value'], errors='coerce')

# Remove lab results with `created_date` in the past.
lab_results = lab_results.loc[lab_results['created_date'] >= pd.to_datetime(START)]

# Identify all of the labs.
lab_ids = list(lab_results['lab_licensee_id'].unique())

# Trend analytes by day by lab.
group = [pd.Grouper(key='created_date', freq='M'), 'test_name', 'lab_licensee_id']
trending = lab_results.groupby(group, as_index=True)['test_value'].mean()

# Visualize all analytes!!!
tested_analytes = list(trending.index.get_level_values(1).unique())
for analyte in tested_analytes:
    fig, ax = plt.subplots(figsize=(8, 5))
    idx = pd.IndexSlice
    for lab_id in lab_ids:
        try:
            lab_samples = trending.loc[idx[:, analyte, lab_id]]
            if len(lab_samples) > 0:
                lab_samples.plot(
                    ax=ax,
                    label=lab_id,
                )
        except KeyError:
            pass
    plt.legend(title='Lab ID', loc='upper right')
    plt.title(f'Average {analyte} by Lab in Washington')
    plt.show()

# TODO: Save trending!

# Calculate failure rate by lab.

# TODO: Calculate failure rate by licensee.
# fail = lab_results.loc[lab_results['LabTestStatus'] == 'Fail']

# Get lab prices.

# Estimate laboratory revenue.

# Estimate laboratory market share.

# TODO: Estimate amount spent on lab testing by licensee.


#------------------------------------------------------------------------------
# Future work: Augment the data.
#------------------------------------------------------------------------------

# Get Fed FRED data pertinent to geographic area.

# Get Census data pertinent to geographic area.


#------------------------------------------------------------------------------
# Future work: Estimate ARIMAX for every variable.
#------------------------------------------------------------------------------

# Estimate each variable by licensee in 2022 by day, month, week, and year-end:
# - total sales
# - number of transactions (Poisson model)
# - average transaction amount
# - Number of failures (Poisson model)


#------------------------------------------------------------------------------
# Save the data and statistics, making the data available for future use.
#------------------------------------------------------------------------------

# Save all the statistics and forecasts to local data archive.
ccrs.save(lab_results, 'D:\\data\\washington\\stats\\daily_sales.xlsx')

# Upload all the statistics and forecasts to make available through the API.
# through the Cannlytics API and Cannlytics Website.
ccrs.upload(lab_results, 'lab_results', id_field='lab_result_id')

# Get all data and statistics from the API!
base = 'http://127.0.0.1:8000/api'
ccrs.get('lab_results', limit=100, base=base)
