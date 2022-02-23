"""
The Effect of Taxes on Prices in Washington State
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 2/23/2022
Updated: 2/23/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script analyzes the relationship between price and
sales tax by county in Washington State.

Data sources:

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd

    - WA State Sales Taxes
    https://dor.wa.gov/get-form-or-publication/publications-subject/local-sales-use-tax-rates-excel-file-format#2021

    - WA Sate Cannabis Tax
    https://dor.wa.gov/about/statistics-reports/recreational-and-medical-marijuana-taxes

"""
# Standard imports.
import glob

# External imports.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # pip install basemap
import pandas as pd
import requests
import seaborn as sns


# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
})

# Specify where your data lives.
DATA_DIR = 'D:\\leaf-data'


#--------------------------------------------------------------------------
# Read the sample data.
#--------------------------------------------------------------------------

# Specify where the data lives.
DATA_DIR = 'D:\\leaf-data'
DATA_FILE = f'{DATA_DIR}/samples/random-sales-items-2022-02-16.csv'

# Read in the data.
data = pd.read_csv(DATA_FILE)


#------------------------------------------------------------------------------
# Get the retail city and county.
# Licensees data: https://cannlytics.com/data/market/augmented-washington-state-licensees
#------------------------------------------------------------------------------

# Read licensees data.
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
data.drop(['global_id'], axis=1, inplace=True, errors='ignore')


#--------------------------------------------------------------------------
# Clean the data.
#--------------------------------------------------------------------------

# Determine wholesale vs retail transactions.
data = data.loc[data['sale_type'] != 'wholesale']

# Drop observations with negative prices and prices in the upper quantile.
data = data.loc[data.price_total > 0]
data = data[data.price_total < data.price_total.quantile(.95)]

# Add a date column.
data['date'] = pd.to_datetime(data['created_at'])
data['day'] = data['date'].dt.date

# Add quarter (YYYYQ1, ...) field.
data['quarter'] = data.date.dt.to_period('Q')

# Clean the city name and county.
data['city'] = data['city'].str.title()
data['county'] = data['county'].str.title().str.replace(' County', '')

# TODO: Calculate price per total cannabinoids ($/mg)


# Get the average price by Zip code (optional: use county instead?).
# zip_code_prices = sample_type_data.groupby('postal_code').mean()['price_total']


#--------------------------------------------------------------------------
# TODO: Get the average tax by county.
#--------------------------------------------------------------------------

# Initialize tax rate for each observation.
excise_tax = 0.37
data['tax_rate']

# Read in the taxes.
tax_datafiles = f'{DATA_DIR}/taxes/*.xlsx'
files = glob.glob(tax_datafiles)   
for file_name in files:

    # Identify the quarter.
    year = int(f'20{file_name[-10:-8]}')
    quarter = f'{year}{file_name[-7:-5]}'
    print(quarter)

    # Read in local sales.
    sales_tax = pd.read_excel(file_name, skiprows=2)
    sales_tax = sales_tax[['Location', 'County', 'Combined Sales Tax']]

    # Match with county given time period.


# Augment the data with sales tax by city first, then county if city not found.


# TODO: Calculate the amount paid in taxes.


# TODO: Calculate the price paid by the consumer.


#--------------------------------------------------------------------------
# TODO: Add sales tax with required cannabis sales tax.
#--------------------------------------------------------------------------


#--------------------------------------------------------------------------
# Analysis: Regress the average price of cannabis on tax rate by
# county by month for the time period of January to
# November of 2021.
#--------------------------------------------------------------------------

# Identify flower sales.
# sample_type = 'usable_marijuana'
# sample_type_data = data.loc[data.intermediate_type == sample_type]

# TODO: Create panel data (county i, time t)


# TODO: Regress the average price of flower p_it on the tax rate t_it.
# Optionally add county and month fixed effects.
# Optionally control for:
# - population
# - Number of sunny days?
# - rainfall?
# - avg. electricity price?
# - avg. temperature
# - median income
# - proportion of economy that is agricultural
# - avg. education in number of years
# - number of tourists
# - sales tax


#--------------------------------------------------------------------------
# Interpretation:
# What would be the effect on prices paid by
# consumers and prices received by producers
# from a marginal increase (or decrease) in
# cannabis sales tax in Washington State?
#--------------------------------------------------------------------------


# TODO: Interpret the estimated parameters.

# TODO: Estimate the prices paid by consumers.


# TODO: Visualize the regression results (avg. price to tax rate) (expected decreasing).
# Optional: Plot price paid by consumers to tax rate (expected to be increasing).


