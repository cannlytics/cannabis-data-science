"""
Poisson Model for Beverage Consumption in Washington State
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 3/4/2022
Updated: 3/5/2022
License: MIT License <https://opensource.org/licenses/MIT>

Data sources:

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd

References:

    Poisson Regression
    - https://jbhender.github.io/Stats506/F17/Projects/Poisson_Regression.html

    - Census Data API User Guide
    https://www.census.gov/content/dam/Census/data/developers/api-user-guide/api-guide.pdf

"""

# External imports.
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

# Internal imports.
from utils import format_millions

# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
})
plt.rc('text', usetex=True)

# Specify where your data lives.
DATA_DIR = 'D:\\leaf-data'


#--------------------------------------------------------------------------
# Read the sample sales data.
# Random sample of sales: https://cannlytics.page.link/cds53
#--------------------------------------------------------------------------

# Read in the data.
DATA_FILE = f'{DATA_DIR}/samples/random-sales-items-2022-03-05.csv'
data = pd.read_csv(DATA_FILE)


#------------------------------------------------------------------------------
# Get the retail city and county.
# Licensees data: https://cannlytics.com/data/market/augmented-washington-state-licensees
# Augment yourself: https://github.com/cannlytics/cannabis-data-science/blob/main/2022-01-26/geocode_licensees.py
#------------------------------------------------------------------------------

# Augment fields from the retailer licensees data.
licensee_fields = {
    'global_id': 'string',
    'city': 'string',
    'county': 'string',
}
licensees = pd.read_csv(
    f'{DATA_DIR}/augmented/augmented-washington-state-licensees.csv',
    usecols=list(licensee_fields.keys()),
    dtype=licensee_fields,
)
data = pd.merge(
    left=data,
    right=licensees,
    how='left',
    left_on='mme_id',
    right_on='global_id',
)
data['retailer_city'] = data['city'].str.title()
data['retailer_county'] = data['county'].str.title().str.replace(' County', '')
data.drop(['global_id', 'city', 'county'], axis=1, inplace=True, errors='ignore')
print('Augmented the sales data with retailer data.')

# Augment fields from the producer licensees data.
data = pd.merge(
    left=data,
    right=licensees,
    how='left',
    left_on='producer_mme_id',
    right_on='global_id',
)
data['producer_city'] = data['city'].str.title()
data['producer_county'] = data['county'].str.title().str.replace(' County', '')
data.drop(['global_id', 'city', 'county'], axis=1, inplace=True, errors='ignore')
print('Augmented the sales data with producer data.')


#--------------------------------------------------------------------------
# Clean the data.
#--------------------------------------------------------------------------

# Determine wholesale vs retail transactions.
# Note: It is necessary to find out if there are any medical sales and
# exclude any medical sales from taxes.
data = data.loc[data['sale_type'] != 'wholesale']

# Drop observations with negative prices and prices in the upper quantile.
data = data.loc[data.price_total > 0]
data = data[data.price_total < data.price_total.quantile(.95)]

# Add a date fields.
data['date'] = pd.to_datetime(data['created_at'])
data['day'] = data['date'].dt.date
data['month'] = data['date'].dt.date
data['quarter'] = data.date.dt.to_period('Q')

# TODO: Calculate price per total cannabinoids ($/mg).
# Need to parse mg from product_name.

# Identify the time period for analysis.
start = '2021-01-01'
end = '2021-10-31'
data = data.loc[
    (data['date'] >= pd.to_datetime(start)) &
    (data['date'] <= pd.to_datetime(end))
]
print('Data cleaned and limited to the period of analysis.')
print('Sample size: {:,} observations.'.format(len(data)))


#--------------------------------------------------------------------------
# Question: Does age affect people's preferences for liquid edibles?
#--------------------------------------------------------------------------
# Analysis: Estimate statistics knowing that we have sampled approximately 0.001
# of the market (so we can estimate population totals as x 1000 our sample totals).
#--------------------------------------------------------------------------
# Model: Fit a Poisson model on the number of beverages sold by month by county
# on various factors, such as
# ✓ average price total in the county
# - average age in the retail county
# - median income in the retail county
# - months since recreational cannabis was permitted
# - month of the year dummy variable
# - anything else you that can think of that has a causal relationship on beverages sold!
#--------------------------------------------------------------------------

import statsmodels.api as sm
from statsmodels.formula.api import poisson

# Identify all beverage sales.
sample_type = 'liquid_edible'
sample_data = data.loc[data.intermediate_type == sample_type]

# TODO: Handle any beverages that may have been sold in packs!

# Estimate the number of beverages sold by day, month, year.
daily_sales = sample_data.groupby('day').count()['price_total'] * 1000
daily_sales.index = pd.to_datetime(daily_sales.index)
monthly_sales = daily_sales.groupby(pd.Grouper(freq='M')).sum()
annual_sales = monthly_sales.groupby(pd.Grouper(freq='Y')).sum()
print(format_millions(annual_sales), 'estimated beverages sold in first 10 months of 2021.')

# Estimate the number of beverage brands.
number_of_brands = len(sample_data['producer_mme_id'].unique())
print(number_of_brands, 'beverage producers in WA in 2021.')

# TODO: Estimate the market share of the top 5 beverage producers.

# Create county panel data.
sample_data.index = sample_data.date
panel = sample_data.groupby([pd.Grouper(freq='M'), 'retailer_county'])
panel_data = panel.count()[['qty']]
panel_data['avg_price'] = panel.mean()['price_total']
panel_data['retailer_county'] = panel_data.index.get_level_values('retailer_county')

# Estimate a Poisson regression.
formula = 'qty ~ avg_price'
model = poisson(formula, data=panel_data).fit(method='newton')
print(model.summary())


#--------------------------------------------------------------------------
# Get Census Data
# API Key: https://api.census.gov/data/key_signup.html
# Attribution: This product uses the Census Bureau Data API but is not
# endorsed or certified by the Census Bureau.
#--------------------------------------------------------------------------

from dotenv import dotenv_values
from census import Census # pip install census
from us import states # pip install us
import requests

# Create a request session.
session = requests.session()
session.headers.update({'User-Agent': 'census-demo/0.0'})

# Read your Census API key.
config = dotenv_values('../.env')
census_api_key = api_key=config['CENSUS_API_KEY']

# Make requests to the Census API.
client = Census(census_api_key, session=session)
census_data = client.acs5.get(
    ('NAME', 'B25034_010E'),
    {'for': 'state:{}'.format(states.MD.fips)}
)

# Examples:
# c.acs5.get('B01001_004E', {'for': 'state:*'})
# c.acs5.state('B01001_004E', Census.ALL)
# c.sf1.state_county_tract('NAME', states.AK.fips, '170', Census.ALL)

# Example request to the Census API.
# baseAPI = "https://api.census.gov/data/2017/acs/acs5?key=%s&get=B01003_001E&for=zip%%20code%%20tabulation%%20area:%s" 
# calledAPI = baseAPI % (apiKey, laZips)
# response = requests.get(calledAPI)
# formattedResponse = json.loads(response.text)[1:]
# formattedResponse = [item[::-1] for item in formattedResponse]
# laZipPopulations = pd.DataFrame(columns=['zipcode', 'population'], data=formattedResponse)
# laZipPopulations.to_csv('laZipPopulations.csv', index=False)

#--------------------------------------------------------------------------
# TODO: Estimate a Poisson regression using county-specific data points.
# For example, you can find a myriad of data points for King County, WA:
# https://www.census.gov/quickfacts/kingcountywashington
#--------------------------------------------------------------------------




# TODO: Use a Chi square test to determine the better model.

# TODO: Visualize the regression results.
