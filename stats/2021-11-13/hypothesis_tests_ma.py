"""
Analysis of Variance (ANOVA) for Massachusetts Cannabis Data
Copyright (c) 2021 Cannlytics and the Cannabis Data Science Meetup Group

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 11/13/2021
Updated: 11/13/2021
License: MIT License <https://opensource.org/licenses/MIT>

References:
    
    - Tests for structural breaks in time-series data
    https://www.stata.com/features/overview/structural-breaks/
    
    - Difference-in-Difference Estimation
    https://www.publichealth.columbia.edu/research/population-health-methods/difference-difference-estimation
    
Data Sources:

    MA Cannabis Control Commission
    - Retail Sales by Date and Product Type: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/xwf2-j7g9
    - Approved Massachusetts Licensees: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/hmwt-yiqy
    - Average Monthly Price per Ounce for Adult-Use Cannabis: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/rqtv-uenj
    - Plant Activity and Volume: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/j3q7-3usu
    - Weekly sales by product type: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/87rp-xn9v

    Fed Fred
    - MA Population: https://fred.stlouisfed.org/series/MAPOP
    - MA Median Income: https://fred.stlouisfed.org/series/MEHOINUSMAA646N
    - MA Income per Capita: https://fred.stlouisfed.org/series/MAPCPI

Objective:
    
    1) Perform analysis of variance of various Massachusetts data points.

"""

# External imports.
from dotenv import dotenv_values
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# Internal imports.
from utils import reverse_dataframe

BREAK = pd.to_datetime('2020-03-26')
RESUME = pd.to_datetime('2020-05-24')

#--------------------------------------------------------------------------
# Get all MA public cannabis data.
#--------------------------------------------------------------------------

# Setup Socrata API, get the App Token, and define the headers.
config = dotenv_values('../.env')
app_token = config.get('APP_TOKEN', None)
headers = {'X-App-Token': app_token}
base = 'https://opendata.mass-cannabis-control.com/resource'

# Get production stats (total employees, total plants, etc.) j3q7-3usu
url = f'{base}/j3q7-3usu.json'
params = {'$limit': 2000, '$order': 'activitysummarydate DESC'}
response = requests.get(url,  headers=headers, params=params)
production = pd.DataFrame(response.json(), dtype=float)
production = reverse_dataframe(production)

# Calculate sales difference.
production['sales'] = production['salestotal'].diff()

# FIX: Fix outlier that appears to have an extra 0.
outlier = production.loc[production.sales >= 10000000]
production.at[outlier.index, 'sales'] = 0

# FIX: Remove negative values.
negatives = production.loc[production.sales < 0]
production.at[negatives.index, 'sales'] = 0

# Aggregate daily production data into monthly and quarterly averages.
production['date'] = pd.to_datetime(production['activitysummarydate'])
production.set_index('date', inplace=True)
monthly_avg_production = production.resample('M').mean()
quarterly_avg_production = production.resample('Q').mean()
monthly_total_production = production.resample('M').sum()
quarterly_total_production = production.resample('Q').sum()

# Get licensees data.
url = f'{base}/hmwt-yiqy.json'
params = {'$limit': 10000,  '$order': 'app_create_date DESC'}
response = requests.get(url,  headers=headers, params=params)
licensees = pd.DataFrame(response.json(), dtype=float)

# Get the monthly average price per ounce.
url = f'{base}/rqtv-uenj.json'
params = {'$limit': 10000, '$order': 'date DESC'}
response = requests.get(url,  headers=headers, params=params)
prices = pd.DataFrame(response.json(), dtype=float)
prices = reverse_dataframe(prices)
prices.set_index('date', inplace=True)

# Calculate the average price per specific quantity.
price_per_gram = prices.avg_1oz.astype(float).divide(28).round(2)
price_per_teenth = prices.avg_1oz.astype(float).divide(16).round(2)
price_per_eighth = prices.avg_1oz.astype(float).divide(8).round(2)
price_per_quarter = prices.avg_1oz.astype(float).divide(4).round(2)

# Get the products.
url = f'{base}/xwf2-j7g9.json'
params = {'$limit': 10000, '$order': 'saledate DESC'}
response = requests.get(url,  headers=headers, params=params)
products = pd.DataFrame(response.json(), dtype=float)
products = reverse_dataframe(products)
products.set_index('saledate', inplace=True)
products.index = pd.to_datetime(products.index)
product_types = list(products.productcategoryname.unique())

#--------------------------------------------------------------------------
# Calculate novel data series.
#--------------------------------------------------------------------------

# Limit time frame.
production = production.loc[production.index >= pd.to_datetime('2019-01-01')]
products = products.loc[products.index >= pd.to_datetime('2019-01-01')]

# Calculate change in sales.
change_in_sales = production['sales'].pct_change()
change_in_sales = change_in_sales.loc[change_in_sales <= 10]
change_in_sales.plot()
plt.show()

# Calculate change in plants.
change_in_plants = production['total_plantfloweringcount'].pct_change()
change_in_plants.plot()
plt.show()

# Calculate change in inventory.
change_in_inventory = production['total_active_packagecount'].pct_change()
change_in_inventory.plot()
plt.show()

# Calculate weekly series
weekly_sales = production.resample('W-SUN')['sales'].sum()
weekly_plants = production.resample('W-SUN')['total_plantfloweringcount'].mean()
weekly_inventory = production.resample('W-SUN')['total_active_packagecount'].mean()

# Calculate change in weekly series.
change_weekly_sales = weekly_sales.pct_change()
change_weekly_plants = weekly_plants.pct_change()
change_weekly_inventory = weekly_inventory.pct_change()

# Plot changes in sales pre-pandemic and post-pandemic.
change_weekly_sales.loc[change_weekly_sales.index < BREAK].plot()
change_weekly_sales.loc[change_weekly_sales.index > RESUME].plot()
plt.show()

# Plot changes in flowering plants.
change_weekly_plants.loc[change_weekly_plants.index < BREAK].plot()
change_weekly_plants.loc[change_weekly_plants.index > RESUME].plot()
plt.show()

# Plot change in inventory.
change_weekly_inventory.loc[change_weekly_inventory.index < BREAK].plot()
change_weekly_inventory.loc[change_weekly_inventory.index > RESUME].plot()
plt.show()

# Define flower sales.
flower_goods = [
    'Buds',
    'Shake/Trim (by strain)',
    'Raw Pre-Rolls',
    'Shake/Trim',
]
flower_products = products.loc[products.productcategoryname.isin(flower_goods)]
flower_sales = flower_products.dollartotal
weekly_flower_sales = flower_sales.resample('W-SUN').sum()
weekly_flower_sales.pct_change().plot()
plt.show()

# Define processed-goods sales.
processed_goods = [
    'Infused (edible)',
    'Infused (non-edible)',
    'Concentrate (Each)',
    'Concentrate',
    'Infused Pre-Rolls',
    'Vape Product',
    'Suppository',
    'Infused Beverage',
    'Concentrate (Bulk)',
    'Infused Beverages (Bulk)'
]
concentrate_sales = products.loc[
    (products.productcategoryname == 'Concentrate') |
    (products.productcategoryname == 'Concentrate (Each')
].dollartotal
concentrate_products = products.loc[products.productcategoryname.isin(processed_goods)]
concentrate_sales = concentrate_products.dollartotal
weekly_concentrate_sales = concentrate_sales.resample('W-SUN').sum()
weekly_concentrate_sales.pct_change().plot()
plt.show()

#--------------------------------------------------------------------------
# Plot histograms for various data points.
#--------------------------------------------------------------------------

# Flower sales percent change pre-pandemic versus
# flower sales percent change post-pandemic.
pre_change_flower_sales = weekly_flower_sales.loc[
    weekly_flower_sales.index < pd.to_datetime(BREAK)
].pct_change()
post_change_flower_sales = weekly_flower_sales.loc[
    weekly_flower_sales.index > pd.to_datetime(RESUME)
].pct_change()

# Plot time series.
pre_change_flower_sales.plot()
post_change_flower_sales.plot()
plt.show()

# Plot histogram.
pre_change_flower_sales.hist(bins=20)
post_change_flower_sales.hist(bins=20)
plt.show()

# Concentrate sales percent change pre-pandemic versus
# concentrate sales percent change post-pandemic.
pre_change_concentrate_sales = weekly_concentrate_sales.loc[
    weekly_concentrate_sales.index < pd.to_datetime(BREAK)
].pct_change()
post_change_concentrate_sales = weekly_concentrate_sales.loc[
    weekly_concentrate_sales.index > pd.to_datetime(RESUME)
].pct_change()

# Plot time series.
pre_change_concentrate_sales.plot()
post_change_concentrate_sales.plot()
plt.show()

# Plot histogram.
pre_change_concentrate_sales.hist(bins=20)
post_change_concentrate_sales.hist(bins=20, alpha=0.6)
plt.show()

#--------------------------------------------------------------------------
# Calculate Pearson correlation coefficients for various data points.
#--------------------------------------------------------------------------

# Calculate sample correlation coefficient between pre-closure change
# in flower and concentrate sales.
np.corrcoef([pre_change_flower_sales[1:], pre_change_concentrate_sales[1:]])

# Calculate sample correlation coefficient between post-closure change
# in flower and concentrate sales.
np.corrcoef([post_change_flower_sales[1:], post_change_concentrate_sales[1:]])

#--------------------------------------------------------------------------
# Perform ANOVA.
#--------------------------------------------------------------------------

# Look at square footage as a whole.
print(licensees.square_footage_establishment.describe())

# Look at square foot usage betwen facility types.
license_types = [
    'Marijuana Transporter with Other Existing ME License',
    'Marijuana Product Manufacturer',
    'Marijuana Cultivator',
    'Marijuana Retailer',
    'Marijuana Microbusiness',
    'Independent Testing Laboratory',
    'Third Party Marijuana Transporter',
    # 'Craft Marijuana Cooperative'
]
for license_type in license_types:
    subset_licensees = licensees.loc[licensees.license_type == license_type]
    print(f'-------------\n{license_type}')
    print('Avg. square feet:', round(subset_licensees.square_footage_establishment.mean(), 0))
    print('Std. deviation in square feet:', round(subset_licensees.square_footage_establishment.std(), 0))
    # print(subset_licensees.square_footage_establishment.describe())

# TODO: Performa ANOVA.
# See: https://www.statsmodels.org/stable/examples/notebooks/generated/interactions_anova.html
# And: https://www.statsmodels.org/stable/anova.html
# table = sm.stats.anova_lm(moore_lm, typ=2) # Type 2 ANOVA DataFrame
