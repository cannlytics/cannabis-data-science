"""
Cannabis Yields in Washington State
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 3/2/2022
Updated: 3/2/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script analyzes yields in Washington State.

Data sources:

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd

    - Random Sample of WA Sales Items from 2022-02-16
    https://cannlytics.page.link/cds53

"""
# Standard imports.
import glob
from datetime import datetime, timedelta

# External imports.
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns

# Internal imports.
from utils import format_millions

# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
})

# Specify where your data lives.
DATA_DIR = 'D:\\leaf-data'


#--------------------------------------------------------------------------
# Read the sample sales data.
# Random sample of sales: https://cannlytics.page.link/cds53
#--------------------------------------------------------------------------

# Read in the data.
DATA_FILE = f'{DATA_DIR}/samples/random-sales-items-2022-02-16.csv'
data = pd.read_csv(DATA_FILE)


#------------------------------------------------------------------------------
# Get the retail city and county.
# Licensees data: https://cannlytics.com/data/market/augmented-washington-state-licensees
# Augment yourself: https://github.com/cannlytics/cannabis-data-science/blob/main/2022-01-26/geocode_licensees.py
#------------------------------------------------------------------------------

# Augment fields from the licensees data.
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
print('Augmented the sales data with city and county.')


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

# Add a date column.
data['date'] = pd.to_datetime(data['created_at'])
data['day'] = data['date'].dt.date

# Add quarter (e.g. 2022Q1) field.
data['quarter'] = data.date.dt.to_period('Q')

# Clean the city name and county.
data['city'] = data['city'].str.title()
data['county'] = data['county'].str.title().str.replace(' County', '')

# TODO: Calculate price per total cannabinoids ($/mg)


# Identify the time period for analysis.
start = '2021-01-01'
end = '2021-10-31'
data = data.loc[
    (data['date'] >= pd.to_datetime(start)) &
    (data['date'] <= pd.to_datetime(end))
]
print('Data cleaned and limited to the period of analysis.')
print('{:,} observations.'.format(len(data)))

# ARCH and GARCH.
sample_type = 'concentrate_for_inhalation'
sample_data = data.loc[data.intermediate_type == sample_type]
daily_data = sample_data.groupby('day')

avg_price = daily_data.mean()['price_total']
avg_price.index = pd.to_datetime(avg_price.index)

# Estimate the total tax paid by month.
monthly_avg_price = avg_price.groupby(pd.Grouper(freq='M')).mean()
monthly_avg_price.plot()

# Estimate the total tax paid by month.
std_price = daily_data.std()['price_total']
std_price.index = pd.to_datetime(std_price.index)
monthly_std_price = std_price.groupby(pd.Grouper(freq='M')).mean()
monthly_std_price.plot()
