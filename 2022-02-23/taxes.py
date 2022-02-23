"""
The Effect of Taxes on Prices in Washington State
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 2/22/2022
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
import pandas as pd
import statsmodels.api as sm
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
# Read the sample sales data.
#--------------------------------------------------------------------------

# Read in the data.
DATA_FILE = f'{DATA_DIR}/samples/random-sales-items-2022-02-16.csv'
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
print('Augmented the sales data with city and county.')


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


# Identify the time period for analysis.
start = '2021-01-01'
end = '2021-10-31'
data = data.loc[
    (data['date'] >= pd.to_datetime(start)) &
    (data['date'] <= pd.to_datetime(end))
]
print('Data cleaned and limited to period of analysis.')
print(len(data), 'observations.')


#--------------------------------------------------------------------------
# Get the average tax by county.
#--------------------------------------------------------------------------

# Initialize tax rate for each observation.
excise_tax = 0.37
state_tax = 0.065
data['tax_rate'] = excise_tax + state_tax

# Assign the location code for each sales item.
data['location_code'] = data['postal_code'].astype(str).str[-4:]

# Read in the taxes, file by file for each quarter.
tax_datafiles = f'{DATA_DIR}/taxes/*.xlsx'
files = glob.glob(tax_datafiles)
files = [x for x in files if not '~' in x]
for file_name in files:

    # Identify the quarter.
    year = int(f'20{file_name[-10:-8]}')
    quarter = f'{year}{file_name[-7:-5]}'
    print('Matching tax rates by city for', quarter)

    # Read in city tax rates for the quarter.
    sales_tax = pd.read_excel(file_name, skiprows=2)
    sales_tax = sales_tax[[
        'Location',
        'Location Code',
        'Combined Sales Tax'
    ]]

    # Match sales items for each city in the quarter.
    # Note: This is slow. Can this be optimized?
    for index, values in sales_tax.iterrows():
        # code = values['Location Code'].astype(str).replace('.0', '')
        data['match'] = data['city'].apply(lambda x: x in values['Location'].strip())
        data.loc[
            # (data['location_code'] == code) &
            (data['match']) &
            (data['quarter'] == quarter),
            'tax_rate'
        ] = values['Combined Sales Tax'] + excise_tax

# Identify the samples with a local tax rate.
has_local_tax = data.loc[data['tax_rate'] > excise_tax + state_tax]
print('Proportion with local tax:', round(len(has_local_tax) / len(data), 2))

# Calculate the amount paid in tax.
data['tax'] = data['tax_rate'] * data['price_total']

# Calculate the price paid by the consumer.
data['consumer_price'] = data['price_total'] + data['tax']

# Save the data.
SAVE_FILE = f'{DATA_DIR}/samples/random-sales-items-with-taxes.xlsx'
data.to_excel(SAVE_FILE, index=False)


#--------------------------------------------------------------------------
# Estimate the amount paid in taxes,
# assuming that our sample is 1/1000 of the population.
#--------------------------------------------------------------------------

# Estimate the average price by sample type by day.
daily_tax = data.groupby('day').sum()['tax'] * 1000

# Estimate the average price by sample type by month.
monthly_tax = daily_tax.groupby(pd.Grouper(freq='M')).sum()

# Estimate the average price by sample type by year.
annual_tax = daily_tax.groupby(pd.Grouper(freq='M')).sum()


#--------------------------------------------------------------------------
# Analysis: Regress the average price of cannabis on tax rate by
# county by month for the time period of January to
# November of 2021.
#--------------------------------------------------------------------------

# Identify flower sales.
sample_type = 'usable_marijuana'
sample_data = data.loc[data.intermediate_type == sample_type]

# Regress price on tax rate for all observations.
Y = sample_data['price_total']
X = sample_data['tax_rate']
X = sm.add_constant(X)
regression = sm.OLS(Y, X).fit()
print(regression.summary())

# TODO: Create panel data (county i, time t)
t = pd.date_range(start, end, periods='M')

panel = sample_data.groupby([
    'county',
    pd.Grouper(key='date', freq='M')
]).mean()

# panel['price_total']
# panel['consumer_price']
# panel['tax_rate']

# Unused: Get conditional average price.
# zip_code_prices = sample_data.groupby('postal_code').mean()['price_total']


# TODO: Regress the average price of flower p_it on the tax rate t_it.
# Optionally add county and month fixed effects.
# Optionally control for:
# - population
# - Number of sunny days (of producer)?
# - rainfall (of producer)?
# - avg. electricity price (of producer)?
# - avg. temperature (of producer)?
# - median income
# - proportion of economy that is agricultural (of producer)?
# - avg. education in number of years
# - number of tourists
# * sales tax


#--------------------------------------------------------------------------
# Interpretation:
# What would be the effect on prices paid by
# consumers and prices received by producers
# from a marginal increase (or decrease) in
# cannabis sales tax in Washington State?
#--------------------------------------------------------------------------


# TODO: Interpret the estimated parameters.


# Format the plot notes.
notes = """Data: A random sample of {:,} “{}” sale items.
Data Source: Washington State Traceability Data from {} through {}.
Notes: The top {}% of sale item observations by price were excluded as outliers.""".format(
    len(sample_data),
    sample_type.replace('_', ' '),
    'January 2021',
    'October 2021',
    '5'
)

# Visualize the regression results (avg. price to tax rate) (expected decreasing).
# Optional: Plot price paid by consumers to tax rate (expected to be increasing).
fig, ax = plt.subplots(figsize=(15, 7))
sns.regplot(
    x='price_total',
    y='tax_rate',
    data=sample_data,
    fit_reg=True,
    ci=True,
    ax=ax,
    label='Price Received',
    # color='b',
)
sns.regplot(
    x='consumer_price',
    y='tax_rate',
    data=sample_data,
    fit_reg=True,
    ci=True,
    ax=ax,
    label='Price Paid',
    # color='g',
)
ax.set_title(
    'Price Paid and Received to Sales Tax\n for Cannabis Flower in Washington State in 2021',
    fontsize=42,
    pad=24,
)
plt.text(0, -0.0575, notes, fontsize=32)
fig.savefig(
    f'{DATA_DIR}/figures/regression_price_on_tax_2021.png',
    format='png',
    dpi=300,
    bbox_inches='tight',
)
plt.show()
