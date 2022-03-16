"""
Exploratory Data Analysis
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 3/15/2022
Updated: 3/15/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script explores the augmented Washington State
sales items.

Data sources:

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd

    - Random Sample of WA Sales Items from 2022-03-16
    https://cannlytics.page.link/cds57

"""
# Standard imports.
from datetime import timedelta

# External imports.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
})


#--------------------------------------------------------------------------
# Read the sample sales data.
# Random sample of sales: https://cannlytics.page.link/cds57
#--------------------------------------------------------------------------

# Read in the data from where your data lives.
DATA_DIR = '../.datasets'
DATA_FILE = f'{DATA_DIR}/random-sales-items-2022-03-16.csv'
data = pd.read_csv(DATA_FILE, low_memory=False)

# Minor cleaning.
data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.strftime('%Y-%m')

# Remove sale outliers.
data = data.loc[data['price_total'] >= 0.25]

# Add shelf-time variable.
data['shelf_time'] = pd.to_datetime(data['created_at']) - pd.to_datetime(data['tested_at'])
data.loc[data['shelf_time'] < timedelta(days=0), 'shelf_time'] = None
data.loc[data['shelf_time'] > data['shelf_time'].quantile(0.95), 'shelf_time'] = None
data['shelf_days'] = data['shelf_time'].dt.days

# Restrict the time period.
data = data.loc[
    (data['date'] >= pd.to_datetime('2020-01-01')) &
    (data['date'] <= pd.to_datetime('2021-10-31'))
]

# Identify specific product type data for analysis.
flower_data = data.loc[
    (data['intermediate_type'] == 'usable_marijuana') &
    (data['parsed_uom'] != 'ea')
]
concentrate_data = data.loc[
    (data['intermediate_type'] == 'concentrate_for_inhalation') &
    (data['parsed_uom'] != 'ea')
]
beverage_data = data.loc[data['intermediate_type'] == 'liquid_edible']


#--------------------------------------------------------------------------
# Look at the distribution of weight and prices (histograms).
# TODO: Visualize standard deviation, mean, etc.
#--------------------------------------------------------------------------

# Plot the distribution of weight of flower sold.
weight = flower_data.loc[
    (flower_data['weight'] > 0) &
    (flower_data['weight'] <= flower_data['weight'].quantile(0.99))
]['weight']
sns.histplot(weight, stat='density', kde=True, bins=100)
plt.show()

# Plot the distribution of weight of concentrates sold.
weight = concentrate_data.loc[
    (concentrate_data['weight'] > 0) &
    (concentrate_data['weight'] <= concentrate_data['weight'].quantile(0.99))
]['weight']
sns.histplot(weight, stat='density', kde=True, bins=100)
plt.show()

# Plot the distribution of beverage prices.
sns.histplot(beverage_data['price_total'], stat='density', kde=True)
plt.show()


#--------------------------------------------------------------------------
# Correlation and regression
#--------------------------------------------------------------------------

# Create a correlation heatmap between flower price and shelf days.
corr = flower_data[['price_total', 'shelf_days']].corr()
sns.heatmap(corr, annot=True, square=True)
plt.yticks(rotation=0)
plt.show()

# Create a correlation heatmap between concentrate price and shelf days.
corr = concentrate_data[['price_total', 'shelf_days']].corr()
sns.heatmap(corr, annot=True, square=True)
plt.yticks(rotation=0)
plt.show()


#--------------------------------------------------------------------------
# Bivariate analysis
#--------------------------------------------------------------------------

# Compare Eastern and Western Washington counties.
comparison = concentrate_data.loc[
    (concentrate_data['county'] == 'Spokane') |
    (concentrate_data['county'] == 'King')
]
sns.relplot(
    x='shelf_days',
    y='price_total',
    hue='county',
    data=comparison
)
plt.show()

# Plot a comparison of median shelf days of flower.
comparison.groupby('county')[['shelf_days']].median().T.plot(kind='barh', figsize=(10,10))

# Plot a comparison of median prices of flower.
comparison.groupby('county')[['price_total']].median().T.plot(kind='barh', figsize=(10,10))

# Plot a comparison of median shelf days for various types.
comparison = data.loc[
    (data['intermediate_type'] == 'usable_marijuana') |
    (data['intermediate_type'] == 'concentrate_for_inhalation') |
    (data['intermediate_type'] == 'liquid_edible')
]
comparison.groupby('intermediate_type')[['shelf_days']].median().T.plot(kind='barh', figsize=(10,10))


# TODO: Try to estimate velocity?


# TODO: Try to plot cumulative density functions (CDFs)?


#--------------------------------------------------------------------------
# Modelling of beverage brands.
#--------------------------------------------------------------------------

# Andrew S. C. Ehrenberg
# negative binomial distribution to the numbers of purchases of
# a brand of consumer goods.


# NBD-Dirichlet model of brand choice successfully modelled the
# repeated category and brand purchases within a wide variety of markets.


# Estimate market share (percent) for the top brands in each category.


# Estimate penetration rate (percent).
# The percentage of the relevant population that has purchased
# a given brand.
# or category at least once in the time period under study.


# Estimate average purchase frequency.


# Are there any brands that are only sold at 1 retail location?


#--------------------------------------------------------------------------
# TODO: Seasonality statistics.
#--------------------------------------------------------------------------


#--------------------------------------------------------------------------
# TODO: Correlate with Census Data
# API Key: https://api.census.gov/data/key_signup.html
# Attribution: This product uses the Census Bureau Data API but is not
# endorsed or certified by the Census Bureau.
# Desired Census data points to explore:
# - Crime per retail county
# - Median income per county
# - Number of tourists / travelers by county?
# - Weather
# - Birth rates
# - Education levels
#--------------------------------------------------------------------------

# from dotenv import dotenv_values
# from census import Census # pip install census
# from us import states # pip install us
# import requests

# # Create a request session.
# session = requests.session()
# session.headers.update({'User-Agent': 'census-demo/0.0'})

# # Read your Census API key.
# config = dotenv_values('../.env')
# census_api_key = api_key=config['CENSUS_API_KEY']

# # Make requests to the Census API.
# client = Census(census_api_key, session=session)
# census_data = client.acs5.get(
#     ('NAME', 'B25034_010E'),
#     {'for': 'state:{}'.format(states.MD.fips)}
# )

# # Examples:
# # c.acs5.get('B01001_004E', {'for': 'state:*'})
# # c.acs5.state('B01001_004E', Census.ALL)
# # c.sf1.state_county_tract('NAME', states.AK.fips, '170', Census.ALL)


# TODO: Use a Chi square test to determine the better model.

# TODO: Visualize the regression results.


# Calculate avg. monthly consumption per capita.
