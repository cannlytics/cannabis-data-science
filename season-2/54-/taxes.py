"""
The Effect of Taxes on Prices in Washington State
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 2/22/2022
Updated: 2/24/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script analyzes the relationship between price and
sales tax by county in Washington State.

Resources:

    - Notes on Partial Equilibrium Analysis
    https://cannlytics.page.link/partial-equilibrium-notes

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
files = [x for x in files if '~' not in x]
for file_name in files:

    # Identify the quarter.
    year = int(f'20{file_name[-10:-8]}')
    quarter = f'{year}{file_name[-7:-5]}'
    print('Matching tax rates by city for', quarter)
    iter_start = datetime.now()

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
        code = str(values['Location Code']).replace('.0', '')
        data['match'] = data['city'].apply(lambda x: x in values['Location'].strip())
        data.loc[
            (data['location_code'] == code) |
            (data['match']) &
            (data['quarter'] == quarter),
            'tax_rate'
        ] = values['Combined Sales Tax'] + excise_tax

    iter_end = datetime.now()
    print('Matched %i cities in ', iter_end - iter_start)

# Identify the samples with a local tax rate.
has_local_tax = data.loc[data['tax_rate'] > excise_tax + state_tax]
print('Proportion of sample with local tax:', round(len(has_local_tax) / len(data), 2))

# Calculate the amount paid in tax.
data['tax'] = data['tax_rate'] * data['price_total']

# Calculate the price paid by the consumers.
data['consumer_price'] = data['price_total'] + data['tax']

# Save the data.
SAVE_FILE = f'{DATA_DIR}/samples/random-sales-items-with-taxes.xlsx'
data.to_excel(SAVE_FILE, index=False)


#--------------------------------------------------------------------------
# Estimate the amount of sales,
# assuming that our sample is 1/1000 of the population.
#--------------------------------------------------------------------------

# Estimate the total tax paid by day.
daily_sales = data.groupby('day').sum()['price_total'] * 1000
daily_sales.index = pd.to_datetime(daily_sales.index)

# Estimate the total tax paid by month.
monthly_sales = daily_sales.groupby(pd.Grouper(freq='M')).sum()

# Estimate the total tax paid by year.
annual_sales = monthly_sales.groupby(pd.Grouper(freq='Y')).sum()
print('Estimated sales in first 10 months of 2021:', format_millions(annual_sales))


#--------------------------------------------------------------------------
# Estimate the amount paid in taxes,
# assuming that our sample is 1/1000 of the population.
#--------------------------------------------------------------------------

# Estimate the total tax paid by day.
daily_tax = data.groupby('day').sum()['tax'] * 1000
daily_tax.index = pd.to_datetime(daily_tax.index)

# Estimate the total tax paid by month.
monthly_tax = daily_tax.groupby(pd.Grouper(freq='M')).sum()

# Estimate the total tax paid by year.
annual_tax = daily_tax.groupby(pd.Grouper(freq='Y')).sum()
print('Estimated tax in first 10 months of 2021:', format_millions(annual_tax))


#--------------------------------------------------------------------------
# Analysis: Regress the average price of cannabis on tax rate by
# county by month for the time period of January to November of 2021.
# Optionally control for:
# ✓ county fixed effects
# ✓ time fixied effects
# ✓ total cannabinoids
# - number of retailers in the county
# - population
# - number of tourists
# - median income
# - avg. education in number of years
# - Number of sunny days (of producer)?
# - rainfall (of producer)?
# - avg. electricity price (of producer)?
# - avg. temperature (of producer)?
# - proportion of economy that is agricultural (of producer)?
#--------------------------------------------------------------------------

# Identify flower sales.
sample_type = 'usable_marijuana'
sample_data = data.loc[data.intermediate_type == sample_type]

# Add trend variable before creating the panel.
sample_data['t'] = sample_data.date.dt.month
sample_data['tax_percent'] = sample_data['tax_rate'] * 100
sample_data['ln_price'] = np.log(sample_data['price_total'])
sample_data['ln_consumer_price'] = np.log(sample_data['consumer_price'])

# Create panel data (county i, time t).
group = sample_data.groupby(['county', pd.Grouper(key='date', freq='M')])
panel = group.mean()

# Make the county and date variables.
panel['county'] = panel.index.get_level_values('county')
panel['date'] = panel.index.get_level_values('date')

# Regress the average price, p_it, on tax rate, t_it, for the various markets
# with time and county fixed effects.
formula = 'price_total ~ tax_percent + total_cannabinoid_percent + C(date) + C(county)'
regression = smf.ols(formula, data=panel).fit()
print(regression.summary())

# Regress the log of average price by county on the tax rate.
formula = 'ln_price ~ tax_percent + total_cannabinoid_percent + C(date) + C(county)'
regression = smf.ols(formula, data=panel).fit()
print(regression.summary())

# Regress the average consumer price, p_it + t_it, on tax rate, t_it, for the various markets
# with time and county fixed effects.
formula = 'consumer_price ~ tax_percent + total_cannabinoid_percent + C(date) + C(county)'
regression = smf.ols(formula, data=panel).fit()
print(regression.summary())

# Regress the log of average price by county on the tax rate.
formula = 'ln_consumer_price ~ tax_percent + total_cannabinoid_percent + C(date) + C(county)'
regression = smf.ols(formula, data=panel).fit()
print(regression.summary())


#--------------------------------------------------------------------------
# Interpretation:
# What would be the effect on prices paid by
# consumers and prices received by producers
# from a marginal increase (or decrease) in
# cannabis sales tax in Washington State?
#--------------------------------------------------------------------------

# TODO: Interpret the estimated parameters.


#--------------------------------------------------------------------------
# Visualize the regression results (avg. price to tax rate) (expected decreasing)
# and price paid by consumers to tax rate (expected to be increasing).
#--------------------------------------------------------------------------

# TODO: Create a 4x4 visualization!

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(60, 40))

# 1. Duel-regression plot.
# fig, ax = plt.subplots(figsize=(19.8, 12))
ax1 = axs[0, 0]
colors = sns.color_palette('Set1', n_colors=10)
sns.regplot(
    x='tax_percent',
    y='consumer_price',
    data=panel,
    ax=ax1,
    label='Average Price Paid',
    color=sns.color_palette('Set2', n_colors=10)[0],
)
sns.regplot(
    x='tax_percent',
    y='price_total',
    data=panel,
    ax=ax1,
    label='Average Price Received',
    color=sns.color_palette('Set1', n_colors=10)[3],
)
# plt.ylabel('Average Price ($)', fontsize=32)
# plt.xlabel('Tax Rate (%)',  fontsize=32)
# plt.legend(loc='upper right', fontsize=32)
plt.setp(ax1.get_yticklabels()[0], visible=False) # Hide first label.
ax1.set_title(
    'Average Price Paid and Received for Cannabis Flower by County by Month \nAgainst Tax Rate in Washington State Counties in 2021',
    fontsize=44,
    pad=24,
)
# plt.figtext(0.05, -0.175,
#     """Notes: Washington State levies a {} cannabis excise tax, a {} state sales tax,
# and each locale may levy a sales tax, ranging from 0.5% to 4%.
# Data: A random sample of {:,} “{}” sale items.
# The top {} of sale item observations by price were excluded as outliers.
# Data Source: Washington State Traceability Data from {} through {}.""".format(
#     '37%',
#     '6.5%',
#     len(sample_data),
#     sample_type.replace('_', ' '),
#     '5%',
#     'January 2021',
#     'October 2021',
# ),
#     fontsize=28,
#     ha='left',
# )
plt.ylim(0, 35)

# 2. TODO: Distribution plot.
# fig, ax = plt.subplots(figsize=(19.8, 12))
# colors = sns.color_palette('Set2', n_colors=10)
# data['price_total'].hist(
#     bins=250,
#     density=1,
#     alpha=0.4,
#     zorder=9,
#     color=sns.color_palette('Set1', n_colors=10)[1],
#     label='Price Total',
# )
# data['tax'].hist(
#     bins=250,
#     density=1,
#     alpha=0.4,
#     zorder=99,
#     color=colors[1],
#     label='Tax',
# )
# data['consumer_price'].hist(
#     bins=250,
#     density=1,
#     alpha=1,
#     color=colors[0],
#     label='Consumer Price',
# )
# plt.legend(loc='upper right')
# plt.xlim(0, 50)
# # plt.ylim(0, 35)
# plt.show()


# 3. TODO: Timeseries plot of monthly taxes, sales, and expenditure.
# TODO: Plot actual taxes as a line plot with dots on top of the bars.
ax3 = axs[1, 0]
width = 6
colors = sns.color_palette('Set2', n_colors=10)
total_expenditure = monthly_sales + monthly_tax
ax3.bar(
    monthly_tax.index - timedelta(days=6) - timedelta(days=30),
    monthly_tax,
    color=colors[1],
    width=width,
    edgecolor='black',
    label='Total Taxes',
)
ax3.bar(
    monthly_sales.index - timedelta(days=30),
    monthly_sales,
    color=colors[2],
    width=width,
    edgecolor='black',
    label='Total Sales',
)
ax3.bar(
    total_expenditure.index + timedelta(days=6) - timedelta(days=30),
    total_expenditure,
    color=colors[0],
    width=width,
    edgecolor='black',
    label='Total Expenditure',
)
ax3.set_title(
    'Estimated Cannabis Sales and Taxes in Washington State in 2021',
    fontsize=44,
    pad=24,
)
# plt.xlabel('Month', fontsize=32)
# plt.ylabel('Dollars ($)',  fontsize=32)
# plt.legend(loc='upper right', fontsize=32)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b. %Y'))
ax3.yaxis.set_major_formatter(FuncFormatter(format_millions))
plt.setp(ax3.get_yticklabels()[0], visible=False) # Hide first label.
# plt.ylim(0, 220_000_000)
plt.show()

# 4. TODO: Data block? Put all notes here?
# format_millions(annual_tax)
# format_millions(annual_sales)
# format_millions(total_expenditure.sum())
# average prices?

# notes = """Notes: Washington State levies a {} cannabis excise tax, a {} state sales tax,
# # and each locale may levy a sales tax, ranging from 0.5% to 4%.
# # Data: A random sample of {:,} “{}” sale items.
# # The top {} of sale item observations by price were excluded as outliers.
# # Data Source: Washington State Traceability Data from {} through {}.""".format(
#     '37%',
#     '6.5%',
#     len(sample_data),
#     sample_type.replace('_', ' '),
#     '5%',
#     'January 2021',
#     'October 2021',
# )
# plt.margins(1, 1)
# fig.savefig(
#     f'{DATA_DIR}/figures/regression_price_on_tax_2021.png',
#     format='png',
#     dpi=300,
#     bbox_inches='tight',
#     pad_inches=0.75,
#     transparent=False,
# )
