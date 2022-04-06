"""
Consumer Choice
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 4/5/2022
Updated: 4/5/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script begins to analyze consumer choice in Massachusetts.

Data sources:

    - Massachusetts Cannabis Control Commission Data Catalog
    https://masscannabiscontrol.com/open-data/data-catalog/
"""
# External imports.
from dotenv import dotenv_values
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Internal imports.
from opendata import CCC

#--------------------------------------------------------------------------
# Wrangle Massachussetts Open Data.
#--------------------------------------------------------------------------

# Initialize a CCC Data Catalog client.
ccc = CCC()

# Get licensees
licensees = ccc.get_licensees()
licensees_approved = ccc.get_licensees('approved')
licensees_pending = ccc.get_licensees('pending')
demographics = ccc.get_licensees('demographics')
under_review_stats = ccc.get_licensees('under-review-stats')
application_stats = ccc.get_licensees('application-stats')

# # Get retail stats.
sales_stats = ccc.get_retail('sales-stats')
sales_weekly = ccc.get_retail('sales-weekly')
prices = ccc.get_retail('price-per-ounce')

# Get agent stats.
gender_stats = ccc.get_agents('gender-stats')
ethnicity_stats = ccc.get_agents('ethnicity-stats')

# Get medical stats.
medical = ccc.get_medical()

# Get cultivation data.
plants = ccc.get_plants()

# # Get sales data.
sales = ccc.get_sales()


#--------------------------------------------------------------------------
# Look at the data!
#--------------------------------------------------------------------------

# Quick look at indoor vs. outdoor (for fun!)
cultivators = licensees_approved.loc[licensees_approved['license_type'] == 'Marijuana Cultivator']
indoor = cultivators.loc[cultivators['cultivation_environment'] == 'Indoor']
outdoor = cultivators.loc[cultivators['cultivation_environment'] == 'Outdoor']
percent_indoor = len(indoor)/ len(cultivators)
percent_outdoor = len(outdoor)/ len(cultivators)

# Plot indoor vs outdoor
colors = sns.color_palette('Set2', n_colors=2)
fig, ax = plt.subplots(figsize=(12, 8))
plt.pie(
    [percent_indoor, percent_outdoor],
    labels=['Indoor', 'Outdoor'],
    colors=colors,
    autopct='%.0f%%'
)
plt.title('Breakdown of Indoor vs. Outdoor Cultivators in Massachussetts')
plt.show()


#--------------------------------------------------------------------------
# Curate the data.
#--------------------------------------------------------------------------

# Identify sales by product type.
flower_sales = sales.loc[
    (sales['productcategoryname'] == 'Buds') &
    (sales['unitofmeasurename'] == 'Grams')
]
oil_sales = sales.loc[
    (sales['productcategoryname'] == 'Concentrate') &
    (sales['unitofmeasurename'] == 'Grams')
]
vape_sales = sales.loc[
    (sales['productcategoryname'] == 'Vape Product')
]
beverage_sales = sales.loc[
    (sales['productcategoryname'] == 'Infused Beverage')
]
edible_sales = sales.loc[
    (sales['productcategoryname'] == 'Infused (edible)')
]
preroll_sales = sales.loc[
    (sales['productcategoryname'] == 'Raw Pre-Rolls') |
    (sales['productcategoryname'] == 'Infused Pre-Rolls')
]

# Format prices.
price_per_gram_flower = flower_sales['totalprice'].div(flower_sales['quantity'])
price_per_gram_flower.index = flower_sales['saledate']

price_per_gram_oil = oil_sales['totalprice'].div(oil_sales['quantity'])
price_per_gram_oil.index = oil_sales['saledate']

price_per_vape = vape_sales['totalprice'].div(vape_sales['quantity'])
price_per_vape.index = vape_sales['saledate']

price_per_beverage = beverage_sales['totalprice'].div(beverage_sales['quantity'])
price_per_beverage.index = beverage_sales['saledate']

price_per_edible = edible_sales['totalprice'].div(edible_sales['quantity'])
price_per_edible.index = edible_sales['saledate']

price_per_preroll = preroll_sales['totalprice'].div(preroll_sales['quantity'])


def calculate_inflation_rate(series, period='M'):
    """Calculate the inflation rate for a series."""
    # timeseries = series.resample(period).mean()
    lag = series.shift(1)
    return (series - lag) / lag


# Calculate inflation.
flower_inflation_rate = calculate_inflation_rate(price_per_gram_flower.resample('M').mean())
oil_inflation_rate = calculate_inflation_rate(price_per_gram_oil.resample('M').mean())
vape_inflation_rate = calculate_inflation_rate(price_per_vape.resample('M').mean())
beverage_inflation_rate = calculate_inflation_rate(price_per_beverage.resample('M').mean())
edible_inflation_rate = calculate_inflation_rate(price_per_edible.resample('M').mean())
# preroll_inflation_rate = calculate_inflation_rate(price_per_preroll.resample('M').mean())


# TODO: Format quantity sold series.

# # Calculate sales difference.
# production['sales'] = production['salestotal'].diff()

# # FIX: Fix outlier that appears to have an extra 0.
# outlier = production.loc[production.sales >= 10000000]
# production.at[outlier.index, 'sales'] = 0

# # FIX: Remove negative values.
# negatives = production.loc[production.sales < 0]
# production.at[negatives.index, 'sales'] = 0

# # Aggregate daily production data into monthly and quarterly averages.
# production['date'] = pd.to_datetime(production['activitysummarydate'])
# production.set_index('date', inplace=True)
# monthly_avg_production = production.resample('M').mean()
# quarterly_avg_production = production.resample('Q').mean()
# monthly_total_production = production.resample('M').sum()
# quarterly_total_production = production.resample('Q').sum()


#--------------------------------------------------------------------------
# Look at actual vs. forecasted sales
#--------------------------------------------------------------------------

# forecasts = pd.read_excel('../.datasets/forecasts/ma-forecasts-2021-10-27.xlsx')


#--------------------------------------------------------------------------
# Augment with data from FRED (Federal Reserve Economic Data).
#--------------------------------------------------------------------------

# from fredapi import Fred

# # Initialize Fred client.
# config = dotenv_values('../.env')
# fred = Fred(api_key=config.get('FRED_API_KEY'))

# Find the observation time start.
# observation_start = prices.index.min()

# # Get the Federal funds interest rate.
# interest_rate = fred.get_series('FEDFUNDS', observation_start=observation_start)
# # interest_rate = end_of_period_timeseries(interest_rate)
# interest_rate.index = interest_rate.index.to_period('M').to_timestamp('M')

# TODO: Get CPI data.


#--------------------------------------------------------------------------
# Augment with Census data. Desired Census data points to explore:
#
# Supply side:
# - Crime per retail county
#
# Demand side:
# - Cannabis use rates
# - Median income per county
# - Number of tourists by county
# - Weather
# - Birth rates
# - Education levels
#
# Attribution: This product uses the Census Bureau Data API but
# is not endorsed or certified by the Census Bureau.
#
# Get a Census API Key: https://api.census.gov/data/key_signup.html
#--------------------------------------------------------------------------

# # TODO: Get Census Data
# from dotenv import dotenv_values
# from census import Census # pip install census
# from us import states # pip install us
# import requests

# # Create a request session.
# session = requests.session()
# session.headers.update({'User-Agent': 'census-demo/0.0'})

# # Read your Census API key.
# config = dotenv_values('../.env')
# census_api_key = config['CENSUS_API_KEY']

# FIXME: Get Census data fields.
# # Make requests to the Census API.
# client = Census(census_api_key, session=session)
# census_data = client.acs5.get(
#     ('NAME', 'B07411_001E'),
#     {'for': 'state:{}'.format(states.WA.fips)}
# )

# Optional: Calculate avg. monthly consumption per capita.


# TODO: Augment the product data with census data.


#--------------------------------------------------------------------------
# Analyze the data.
#--------------------------------------------------------------------------

# Question: How does sales correlate to state-level factors?


# Question: How does market penetration relate to county-level factors?


#--------------------------------------------------------------------------
# Optional: Augment with Washington State data and estimate price model.
# Inverse-demand equation: P = a - bQ
#--------------------------------------------------------------------------

