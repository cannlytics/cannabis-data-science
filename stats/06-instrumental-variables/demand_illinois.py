"""
Estimate the Cannabis Demand Curve in Illinois
Cannabis Data Science Meetup Group
Saturday Morning Statistics
Copyright (c) 2021 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 12/04/2021
Updated: 12/04/2021
License: MIT License <https://opensource.org/licenses/MIT>

Data Sources:
    
    - Licensed Adult Use Cannabis Dispensaries
    <https://www.idfpr.com/LicenseLookup/AdultUseDispensaries.pdf>

    - Illinois adult use cannabis monthly sales figures
    <https://www.idfpr.com/Forms/AUC/2021%2011%2002%20IDFPR%20monthly%20adult%20use%20cannabis%20sales.pdf>

Objective:
    
    Retrieve Illinois cannabis data, locked in public PDFs,
    to attempt to estimate a cannabis demand curve in Illinois.

    You will also need to install various Python dependencies,
    including pdfplumber and linearmodels.
    
    `pip install pdfplumber linearmodels`
    
    A `data` folder is expected.

References:
    
    The Tariff on Animal and Vegetable Oils
    <https://scholar.harvard.edu/files/stock/files/tariff_appendixb.pdf>
    
    Instrumental Variables - Causal Inference for the Brave and True
    <https://matheusfacure.github.io/python-causality-handbook/08-Instrumental-Variables.html>

"""
# Standard imports.
from datetime import datetime

# External imports.
from linearmodels import IV2SLS
import matplotlib.pyplot as plt
import pandas as pd
import pdfplumber
import requests
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import abline_plot

# Internal imports.
from utils import end_of_period_timeseries

#-----------------------------------------------------------------------------
# Download and parse the retailer licensee data.
#-----------------------------------------------------------------------------

# Download the licensees PDF.
licensees_url = 'https://www.idfpr.com/LicenseLookup/AdultUseDispensaries.pdf'
filename = './data/illinois_retailers.pdf'
response = requests.get(licensees_url)
with open(filename, 'wb') as f:
    f.write(response.content)

# Read the licensees PDF.
pdf = pdfplumber.open(filename)

# Get all of the table data.
table_data = []
for page in pdf.pages:
    table = page.extract_table()
    table_data += table
    
# Remove the header.
table_data = table_data[1:]

# Create a DataFrame from the table data.
licensee_columns = [
    'organization',
    'trade_name',
    'address',
    'medical',
    'license_issue_date',
    'license_number', 
]
licensees = pd.DataFrame(table_data, columns=licensee_columns)

# Clean the organization names.
licensees['organization'] = licensees['organization'].str.replace('\n', '')

# Separate address into 'street', 'city', 'state', 'zip_code', 'phone_number'.
# FIXME: This could probably be done more elegantly and it's not perfect.
streets, cities, states, zip_codes, phone_numbers = [], [], [], [], []
for index, row in licensees.iterrows():
    parts = row.address.split(' \n')
    streets.append(parts[0])
    phone_numbers.append(parts[-1])
    locales = parts[1]
    city_locales = locales.split(', ')
    state_locales = city_locales[-1].split(' ')
    cities.append(city_locales[0])
    states.append(state_locales[0])
    zip_codes.append(state_locales[-1])
licensees['street'] = pd.Series(streets)
licensees['city'] = pd.Series(cities)
licensees['state'] = pd.Series(states)
licensees['zip_code'] = pd.Series(zip_codes)
licensees['phone_number'] = pd.Series(phone_numbers)

# Save the licensees data.
licensees.to_excel('./data/licensees_data_il.xlsx', sheet_name='Data')

#-----------------------------------------------------------------------------
# Download and parse the sales data.
#-----------------------------------------------------------------------------

# Download the sales data PDF.
sales_url = 'https://www.idfpr.com/Forms/AUC/2021%2011%2002%20IDFPR%20monthly%20adult%20use%20cannabis%20sales.pdf'
filename = './data/illinois_sales.pdf'
response = requests.get(sales_url)
with open(filename, 'wb') as f:
    f.write(response.content)

# Read the sales data PDF.
pdf = pdfplumber.open(filename)

# Get all of the table data.
table_data = []
for page in pdf.pages:
    
    # Get all of the tables on the page.
    tables = page.find_tables()
    for table in tables:
        data = table.extract()
        table_data += data
    
# Add the year to each observation, assuming reverse chronological order
# starting at the beginning year, 2020, and adding a year at each beginning
# of year.
year = 2020
for row in reversed(table_data):
    row.append(year)
    if row[0] == 'January':
        year += 1

# Create a DataFrame from the table data.
sales_columns = [
    'month',
    'items_sold',
    'in_state_sales',
    'out_of_state_sales',
    'total_sales',
    'year',
]
sales_data = pd.DataFrame(table_data, columns=sales_columns)

# Create a time index and only keep rows that start with a month name.
def month_year_to_date(x):
    try:
        return datetime.strptime(x.replace('.0', ''), '%B %Y')
    except:
        return pd.NaT

# Set the time index.
dates = sales_data.month.map(str) + ' ' + sales_data.year.map(str)
dates = dates.apply(month_year_to_date)
sales_data.index = dates
sales_data = sales_data.loc[sales_data.index.notnull()]
sales_data.sort_index(inplace=True)

# Convert string columns to numeric, handling dollar signs.
sales_data[sales_data.columns[1:]] = sales_data[sales_data.columns[1:]] \
    .replace('[\$,]', '', regex=True).astype(float)

# Set the index as the end of the month.
sales_data = end_of_period_timeseries(sales_data)

# Save the sales data.
sales_data.to_excel('./data/sales_data_il.xlsx', sheet_name='Data')

#-----------------------------------------------------------------------------
# Calculate Illinois retailer statistics.
#-----------------------------------------------------------------------------

# Format the `license_issue_date`.
licensees['issue_date'] = pd.to_datetime(licensees['license_issue_date'])

# Create total retailers by month series.
total_retailers = []
for index, _ in sales_data.iterrows():
    licensed_retailers = licensees.loc[licensees['issue_date'] <= index]
    count = len(licensed_retailers)
    total_retailers.append(count)
sales_data['total_retailers'] = pd.Series(total_retailers, index=sales_data.index)

# Create a stats DataFrame.
stats = pd.concat([sales_data['total_retailers']], axis=1)

#--------------------------------------------------------------------------
# Look at average price per item in Illinois over time.
#--------------------------------------------------------------------------

# Calculate the average price per item.
avg_sales_per_item = sales_data.total_sales / sales_data.items_sold
avg_sales_per_item.plot(title='Average Price per Item')
plt.show()

#--------------------------------------------------------------------------
# Attempt to identify the demand curve without an instrument.
#--------------------------------------------------------------------------

# Run a regression of "price" on "quantity".
Y = avg_sales_per_item
x = sales_data.items_sold
X = sm.add_constant(x)
regression = sm.OLS(Y, X).fit()
print(regression.summary())

# Interpret the relationship.
b_0 = beta = regression.params.values[0]
beta = regression.params.values[1]
change_q = .01
change_p = (b_0 + beta * x * (1 + change_q)) - (b_0 + beta * x)
elasticity_of_demand = change_q / (change_p / Y)

# Visualize the regression.
stats['Price'] = Y
stats['Quantity'] = x
ax = stats.plot(
    x='Quantity',
    y='Price',
    kind='scatter'
)
abline_plot(
    model_results=regression,
    ax=ax
)
plt.show()

#--------------------------------------------------------------------------
# Attempt to identify the demand curve with an instrument.
#--------------------------------------------------------------------------

# Specifiy instrument.
stats['total_retailers'] = sales_data['total_retailers']
# stats['change_in_retailers'] = sales_data['total_retailers'].diff()

# Run the first-stage regression.
X_fs = sm.add_constant(stats['total_retailers'])
first_stage_regression = sm.OLS(stats['Quantity'], X_fs).fit()
print(first_stage_regression.summary())

# Fit the first stage regression.
model_ols = IV2SLS.from_formula('Price ~ 1 + Quantity + total_retailers', stats)
ols1 = model_ols.fit() # robust HCV
ols2 = model_ols.fit(cov_type='unadjusted')

# Fit IV regressions.
model_iv = IV2SLS.from_formula('Price ~ 1 + [Quantity ~ total_retailers]', stats)
iv1 = model_iv.fit() # With robust HCV.
print(iv1.summary)
iv2 = model_iv.fit(cov_type='kernel') # With HACV.
print(iv2.summary)
iv3 = model_iv.fit(cov_type='unadjusted') # With Homoscedastic CV.
print(iv3.summary)

# Compare model results based on linearmodels.
from linearmodels.iv.results import compare
res = {
       'OLS': ols2,
       'IV': iv3,
       'OLS-hcv': ols1,
       'IV-hcv': iv1,
       'IV-hacv': iv2,
}
print(compare(res))

#--------------------------------------------------------------------------
# Perform tests for proper IV specification.
# TODO: Research interpretation of each test.
#--------------------------------------------------------------------------

# (1) Durbin's test of exogeneity.
print(iv1.durbin())

# (2) Wu-Hausman test of exogeneity.
print(iv1.wu_hausman())

# Wooldridge's regression test of exogeneity.
print(iv1.wooldridge_regression)

# Wooldridge's score test of exogeneity.
print(iv1.wooldridge_score)

# Wooldridge's score test of overidentification.
print(iv1.wooldridge_overid)

# Sargan's test of overidentification.
print(iv1.sargan)

# First Stage Diagnostics
print(iv1.first_stage)
