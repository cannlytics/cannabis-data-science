"""
Hemp Analysis in Regards to Prohibition
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 5/10/2022
Updated: 5/11/2022
License: MIT License <https://opensource.org/licenses/MIT>

This product uses the NASS API but is not endorsed or certified by NASS.

Data Sources:

    - The National Agricultural Statistics Service (NASS) Quick Stats API
    https://quickstats.nass.usda.gov/

    - PRISM USDA Plant Hardiness Zone GIS Datasets
    https://prism.oregonstate.edu/projects/plant_hardiness_zones.php

"""
# External imports.
from cannlytics.utils.utils import snake_case
from dotenv import dotenv_values
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})
colors = sns.color_palette('Set2', n_colors=10)
primary_colors = sns.color_palette('husl', n_colors=10)


#------------------------------------------------------------------------------
# Explore the NASS API.
# Get yours here: https://quickstats.nass.usda.gov/api
#------------------------------------------------------------------------------

# Read NASS API key.
config = dotenv_values('../.env')
api_key = config['NASS_API_KEY']
base = 'http://quickstats.nass.usda.gov/api/'

# Definite the commodity.
commodity = 'HEMP'

# Get parameters.
endpoint = 'get_param_values'
params = {
    'key': api_key,
    'param': 'short_desc',
    'commodity_desc': commodity,
}
url = base + endpoint
response = requests.get(url, params=params)
hemp_datasets = response.json()['short_desc']

# Get counts.
endpoint = 'get_counts'
params = {
    'key': api_key,
    'source_desc': 'SURVEY',
    'commodity_desc': commodity,
    'year__GE': 2021,
    'state_alpha': 'NC',
    
}
url = base + endpoint
response = requests.get(url, params=params)
data = response.json()['count']
print(data)

# Get data. (Formats: json | csv | xml)
endpoint = f'api_GET/'
params = {
    'key': api_key,
    'source_desc': 'SURVEY',
    'commodity_desc': commodity,
    'year__GE': 2021,
    'state_alpha': 'NC',
    'format': 'json',
}
url = base + endpoint
response = requests.get(url, params=params)
data = response.json()['data']


#------------------------------------------------------------------------------
# Just for fun: Get fertilizer prices.
# Note: Fertilizer prices are in real 2011 dollars.
# $1 in January of 2011 has the same buying power as
# $1.28 in January 2022.
# See https://www.bls.gov/data/inflation_calculator.htm
#------------------------------------------------------------------------------


def parse_nass_data(response, freq='M'):
    """Parse a NASS data request response into a DataFrame with
    a well-formatted timeseries index."""
    data = pd.DataFrame(response.json()['data'])
    if freq == 'M':
        data['month'] = data['reference_period_desc'].str.title()
        data = data[data.month != 'Year']
        month_year = data['month'] + data['year'].astype(str)
        date_format = '%b%Y'
    else:
        month_year = data['year'].astype(str) + '-12-31'
        date_format = '%Y-%m-%d'
    data.index = pd.to_datetime(month_year, format=date_format)
    data['date'] = data.index
    data.sort_index(inplace=True)
    data.columns = [snake_case(x) for x in data.columns]
    data['value'] = data['value'].apply(
        lambda x: x.replace(',', '').replace('(D)', '').strip()
    )
    data.loc[data.value == '', 'value'] = 0
    data['value'] = pd.to_numeric(data['value'])
    return data


# Define default parameters.
url = base + 'api_GET'
params = {
    'key': api_key,
    'year__GE': 2021,
}

# Get potassium and phosphorous price.
params['short_desc'] = 'POTASH & PHOSPHATE - INDEX FOR PRICE PAID, 2011'
response = requests.get(url, params=params)
potash = parse_nass_data(response)

# Get nitrogen price.
params['short_desc'] = 'NITROGEN - INDEX FOR PRICE PAID, 2011'
response = requests.get(url, params=params)
nitrogen = parse_nass_data(response)

# Visualize fertilizer prices.
sns.lineplot(
    x='date',
    y='value',
    data=nitrogen,
    label='Nitrogen (N)',
    color=colors[0],
)
sns.lineplot(
    x='date',
    y='value',
    data=potash,
    label='Potash (K) and Phosphate (P)',
    color=colors[1],
)
plt.gcf().set_size_inches(18.5, 10.5)
plt.xlabel('')
plt.ylabel('Price Paid Relative to 2011 Dollars')
plt.title('Fertilizer 2011 Based Price Index')
plt.show()


#------------------------------------------------------------------------------
# Get hemp data from the NASS API.
#------------------------------------------------------------------------------

# Define default parameters.
url = base + 'api_GET'
params = {
    'key': api_key,
    'year__GE': 2021,
    'source_desc': 'SURVEY',
    'commodity_desc': commodity,
}

# Get the total canopy of hemp.
params['short_desc'] = 'HEMP, INDUSTRIAL, IN THE OPEN - ACRES HARVESTED'
response = requests.get(url, params=params)
harvested = parse_nass_data(response, freq='Y')
harvested['state'] = harvested['location_desc'].str.title()
harvested.sort_values(
    by='value',
    inplace=True,
    ascending=False,
)

# Optional: Hue by plant hardiness zone average?

# Visualize harvested canopy.
fig, ax = plt.subplots(figsize=(18, 11.5))
sns.barplot(
    x='state',
    y='value',
    data=harvested.loc[~harvested['state_alpha'].isin(['US', 'OT'])],
    dodge=False,
)
plt.xticks(rotation=90)
plt.xlabel('')
plt.ylabel('Acres')
plt.title('Acres of Harvested Hemp in the US by State in 2021')
plt.show()


#------------------------------------------------------------------------------

# # Get the total floral yield (lbs.).
params['short_desc'] = 'HEMP, INDUSTRIAL, IN THE OPEN, FLORAL - PRODUCTION, MEASURED IN LB'
response = requests.get(url, params=params)
floral_yield = parse_nass_data(response, freq='Y')
floral_yield['state'] = floral_yield['location_desc'].str.title()
floral_yield.sort_values(
    by='value',
    inplace=True,
    ascending=False,
)
floral_yield['tons'] = floral_yield['value'] / 2000

# Visualize floral yields.
fig, ax = plt.subplots(figsize=(18, 11.5))
sns.barplot(
    x='state',
    y='tons',

    data=floral_yield.loc[~floral_yield['state_alpha'].isin(['US', 'OT'])],
    dodge=False,
)
plt.xticks(rotation=90)
plt.xlabel('')
plt.ylabel('Tons')
plt.title('Tons of Floral Hemp Produced in the US by State in 2021')
plt.show()

#------------------------------------------------------------------------------

# Get the median floral yield (lbs. per acre).
params['short_desc'] = 'HEMP, INDUSTRIAL, IN THE OPEN, FLORAL - YIELD, MEASURED IN LBS / ACRE'
response = requests.get(url, params=params)
pounds_per_acre = parse_nass_data(response, freq='Y')
pounds_per_acre['state'] = pounds_per_acre['location_desc'].str.title()
pounds_per_acre.sort_values(
    by='value',
    inplace=True,
    ascending=False,
)

# Visualize median floral yield.
fig, ax = plt.subplots(figsize=(18, 11.5))
sns.barplot(
    x='state',
    y='value',

    data=pounds_per_acre.loc[~pounds_per_acre['state_alpha'].isin(['US', 'OT'])],
    dodge=False,
)
plt.xticks(rotation=90)
plt.xlabel('')
plt.ylabel('Tons')
plt.title('Median Pounds per Acre of Floral Hemp Produced in the US by State in 2021')
plt.show()


#------------------------------------------------------------------------------
# Augment with (average?) plant hardiness zones.
#------------------------------------------------------------------------------

import zipcodes # pip install zipcodes


def match_zip_to_state(value):
    """Match zipcode to state abbreviation."""
    state = None
    match = zipcodes.matching(str(value))
    if match:
        state = match[0]['state']
    return state


def calc_mean_range(value):
    """Calculate the mean of a given range 'x to y'."""
    parts = value.split(' to ')
    minimum = parts[0]
    maximum = parts[0]
    return (float(minimum) + float(maximum)) / 2


# Commented for brevity.
# # Read plant hardiness zone (PHZ) data.
# datafile = '../.datasets/public/phm_us_zipcode.csv'
# phz = pd.read_csv(datafile)
# phz['state'] = phz['zipcode'].apply(match_zip_to_state)

# # Save plant hardiness zone with the states.
# phz.to_excel('../.datasets/public/phz.xlsx')

# Read the plant hardiness zone data back in.
phz = pd.read_excel('../.datasets/public/phz.xlsx', index_col=0)

# Calculate average annual extreme minimum temperature
# (1976-2005) mean temperature (F).
phz['mean_min_temp'] = phz['trange'].apply(calc_mean_range)
state_min_temp = phz.groupby('state', as_index=False)['mean_min_temp'].mean()
state_min_temp.sort_values('mean_min_temp', inplace=True)

# Visualize historic extreme minimum temperature by state.
fig, ax = plt.subplots(figsize=(18, 11.5))
sns.barplot(
    x='state',
    y='mean_min_temp',
    data=state_min_temp,
    dodge=False,
)
plt.xticks(rotation=90)
plt.xlabel('')
plt.ylabel('Mean Extreme Minimum Temperature')
plt.title('Mean Extreme Minimum Temperature by State')
plt.show()


#------------------------------------------------------------------------------
# Is there any correlation / pattern between plant hardiness zones and
# hemp production?
#------------------------------------------------------------------------------

# Correlate avg. plant hardiness zone with
sample = pd.merge(
    left=harvested,
    right=state_min_temp,
    left_on='state_alpha',
    right_on='state',
    how='left',
)
fig, ax = plt.subplots(figsize=(18, 11.5))
ax = sns.regplot(
    x='mean_min_temp',
    y='value',
    data=sample,
    color='green',
)
plt.xticks(rotation=90)
plt.xlabel('')
plt.ylabel('Mean Extreme Minimum Temperature')
plt.title('Correlation between Plant Hardiness Zones and Acres of Hemp')
plt.show()

#------------------------------------------------------------------------------

# Correlate avg. plant hardiness zone with
# total floral yield (lbs.).
sample = pd.merge(
    left=floral_yield,
    right=state_min_temp,
    left_on='state_alpha',
    right_on='state',
    how='left',
)
fig, ax = plt.subplots(figsize=(18, 11.5))
ax = sns.regplot(
    x='mean_min_temp',
    y='value',
    data=sample,
    color='green',
)
plt.xticks(rotation=90)
plt.xlabel('')
plt.ylabel('Mean Extreme Minimum Temperature')
plt.title('Correlation between Plant Hardiness Zones and Floral Yield')
plt.show()

#------------------------------------------------------------------------------

# Correlate avg. plant hardiness zone with
# median floral yield (lbs. per acre).
sample = pd.merge(
    left=pounds_per_acre,
    right=state_min_temp,
    left_on='state_alpha',
    right_on='state',
    how='left',
)
fig, ax = plt.subplots(figsize=(18, 11.5))
ax = sns.regplot(
    x='mean_min_temp',
    y='value',
    data=sample,
    color='green',
)
plt.xticks(rotation=90)
plt.xlabel('')
plt.ylabel('Mean Extreme Minimum Temperature')
plt.title('Correlation between Plant Hardiness Zones and Pounds per Acre')
plt.show()


#------------------------------------------------------------------------------
# Coming up in Saturday Morning Statistics
# How NOT to lie with statistics,
# measuring the probability of Type 1 and type 2 errors.
#------------------------------------------------------------------------------
