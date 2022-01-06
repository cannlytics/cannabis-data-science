"""
Get Retailer Statistics for Illinois
Cannabis Data Science Meetup Group
Saturday Morning Statistics
Copyright (c) 2021 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 11/17/2021
Updated: 11/27/2021
License: MIT License <https://opensource.org/licenses/MIT>

Data Sources:
    
    - Illinois Licensed Adult Use Cannabis Dispensaries
    <https://www.idfpr.com/LicenseLookup/AdultUseDispensaries.pdf>

    - Illinois adult use cannabis monthly sales figures
    <https://www.idfpr.com/Forms/AUC/2021%2011%2002%20IDFPR%20monthly%20adult%20use%20cannabis%20sales.pdf>

Resources:
    
    - Fed Fred API Keys
    <https://fred.stlouisfed.org/docs/api/api_key.html>
    
Objective:
    
    Retrieve Illinois cannabis data, locked in public PDFs,
    to save the data and calculate interesting statistics,
    such as retailers per 100,000 people and sales per retailer.
    
    You will need a Fed Fred API Key saved in a .env file
    as a FRED_API_KEY variable. A `data` and `figure` folders
    are also expected.

    You will also need to install various Python dependencies,
    including fredapi and pdfplumber.
    
    `pip install fredapi pdfplumber`

"""
# Standard imports.
from datetime import datetime

# External imports.
from dotenv import dotenv_values
from fredapi import Fred
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import pdfplumber
import requests
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import abline_plot

# Internal imports.
from utils import (
    end_of_period_timeseries,
    format_thousands,
)

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

# Get the Illinois population data.
config = dotenv_values('../.env')
fred_api_key = config.get('FRED_API_KEY')
fred = Fred(api_key=fred_api_key)
observation_start = sales_data.index.min().isoformat()
population = fred.get_series('ILPOP', observation_start=observation_start)
population = population.multiply(1000) # thousands of people

# Conjecture that the population remains constant in 2021.
new_row = pd.DataFrame([population[-1]], index=[pd.to_datetime('2021-12-31')])
population = pd.concat([population, pd.DataFrame(new_row)], ignore_index=False)

# Project monthly population.
monthly_population = population.resample('M').mean().pad()
monthly_population = monthly_population.loc[monthly_population.index <= sales_data.index.max()]

# Calculate retailers per capita.
capita = monthly_population / 100_000
retailers_per_capita = sales_data['total_retailers'] / capita[0]
retailers_per_capita.plot(title='Retailers per 100,000 People')
plt.show()

# Calculate sales per retailer.
sales_per_retailer = sales_data['total_sales'] / sales_data['total_retailers']
sales_per_retailer.plot(title='Sales per Retailer')
plt.show()

# Save the retail statistics.
stats = pd.concat([retailers_per_capita, sales_per_retailer], axis=1)
stats.columns = ['retailers_per_capita', 'sales_per_retailer']
stats.to_excel('./data/retail_stats_il.xlsx')

#-----------------------------------------------------------------------------
# Wahoo! We have all of the data, let's calculate even more stats.
#-----------------------------------------------------------------------------

# Calculate average retailers per capita in 2020.
avg_2020_retailers_per_capita = retailers_per_capita.loc[
    (retailers_per_capita.index >= pd.to_datetime('2020-01-01')) &
    (retailers_per_capita.index < pd.to_datetime('2021-01-01'))
].mean()
print('Retailres per capita in IL in 2020: %.2f' % avg_2020_retailers_per_capita)

# Calculate average revenue per retailer in 2020.
avg_2020_sales = sales_per_retailer.loc[
    (sales_per_retailer.index >= pd.to_datetime('2020-01-01')) &
    (sales_per_retailer.index < pd.to_datetime('2021-01-01'))
].sum()
print('Sales per retailer in IL in 2020: %.2fM' % (avg_2020_sales / 1_000_000))

# Calculate average retailers per capita in 2021.
avg_2021_retailers_per_capita = retailers_per_capita.loc[
    (retailers_per_capita.index >= pd.to_datetime('2021-01-01')) &
    (retailers_per_capita.index < pd.to_datetime('2022-01-01'))
].mean()
print('Retailres per capita in IL in 2021: %.2f' % avg_2021_retailers_per_capita)

# Calculate average revenue per retailer in 2021.
avg_2021_sales = sales_per_retailer.loc[
    (sales_per_retailer.index >= pd.to_datetime('2021-01-01')) &
    (sales_per_retailer.index < pd.to_datetime('2022-01-01'))
].sum()
print('Sales per retailer in IL in 2021: %.2fM' % (avg_2021_sales / 1_000_000))


#--------------------------------------------------------------------------
# Estimate the relationship between dispensaries per capita and
# sales per dispensary.
#--------------------------------------------------------------------------

# Run a regression of sales per retailer on retailers per 100,000 people.
Y = sales_per_retailer
X = retailers_per_capita
X = sm.add_constant(X)
regression = sm.OLS(Y, X).fit()
print(regression.summary())

# Interpret the relationship.
beta = regression.params.values[1]
statement = """If retailers per 100,000 people increases by 1,
then everything else held constant one would expect
revenue per retailer to change by {}.
""".format(format_thousands(beta))
print(statement)

# Visualize the regression.
ax = stats.plot(
    x='retailers_per_capita',
    y='sales_per_retailer',
    kind='scatter'
)
abline_plot(
    model_results=regression,
    ax=ax
)
plt.show()

#--------------------------------------------------------------------------
# Create a beautiful visualization.
#--------------------------------------------------------------------------

# # Set chart defaults.
# plt.style.use('fivethirtyeight')
# plt.rcParams['font.family'] = 'Times New Roman'

# # Create the figure. 
# fig, ax = plt.subplots(figsize=(15, 5))

# # Write the text.
# title = """The Relationship Between
# Dispensaries per Capita and
# Sales per Dispensary
# in Illinois"""
# notes = """Data: Monthly number of retailers and total sales in Illinois from
# {start} through {stop}. Annual population to 2020 is used and projected forward.
# Data Source: Illinois Department of Financial and Professional Regulation."""
# notes = notes.format(
#     start=sales_data.iloc[0]['month'] + ' ' + str(int(sales_data.iloc[0]['year'])),
#     stop=sales_data.iloc[-1]['month'] + ' ' + str(int(sales_data.iloc[-1]['year'])),
# )

# # Plot the points.
# stats.plot(
#     x='retailers_per_capita',
#     y='sales_per_retailer',
#     kind='scatter',
#     ax=ax
# )

# # Annotate each point.
# for index, row in stats.iterrows():
#     point = (
#         row['retailers_per_capita'],
#         row['sales_per_retailer']
#     )
#     text = index.strftime('%#m/%y')
#     ax.annotate(text, point, fontsize=11)
    
# # Plot the regression line.
# abline_plot(model_results=regression, ax=ax)

# # Format the Y-axis.
# yaxis_format = FuncFormatter(format_thousands)
# ax.yaxis.set_major_formatter(yaxis_format)

# # Plot the title, labels, and notes.
# plt.ylabel('Monthly Revenue per Dispensary ($)', fontsize=16)
# plt.xlabel('Retailers per 100,000 People', fontsize=16)
# plt.title(title, fontsize=21, pad=10)
# plt.figtext(0.05, -0.15, notes, ha='left', fontsize=12)

# # Format the plot by removing unnecessary ink.
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(True)
# ax.spines['left'].set_visible(True)

# # Save and show the figure.
# plt.margins(1, 1)
# plt.savefig(
#     'figures/revenue_per_retailer_to_retailers_per_100_000_il.png',
#     dpi=300,
#     bbox_inches='tight',
#     pad_inches=0.75,
#     transparent=False,
# )
# plt.show()


#--------------------------------------------------------------------------
# Look at average price per item in Illinois over time.
#--------------------------------------------------------------------------

avg_sales_per_item = sales_data.total_sales / sales_data.items_sold
avg_sales_per_item.plot(title='Average Price per Item')
plt.show()

#--------------------------------------------------------------------------
# Future work: Attempt to identify the demand curve.
#--------------------------------------------------------------------------

# # Run a regression of "price" on "quantity".
# Y = avg_sales_per_item
# x = sales_data.items_sold
# X = sm.add_constant(x)
# regression = sm.OLS(Y, X).fit()
# print(regression.summary())

# # Interpret the relationship.
# b_0 = beta = regression.params.values[0]
# beta = regression.params.values[1]
# change_q = .01
# change_p = (b_0 + beta * x * (1 + change_q)) - (b_0 + beta * x)
# elasticity_of_demand = change_q / (change_p / Y)

# # Visualize the regression.
# stats['Price'] = Y
# stats['Quantity'] = x
# ax = stats.plot(
#     x='Quantity',
#     y='Price',
#     kind='scatter'
# )
# abline_plot(
#     model_results=regression,
#     ax=ax
# )
# plt.show()

# instrument = sales_data['total_retailers'].diff()

#--------------------------------------------------------------------------
# Save the data.
#--------------------------------------------------------------------------
stats = pd.concat([
    sales_data['total_sales'],
    sales_data['total_retailers'],
    retailers_per_capita,
    sales_per_retailer,
    monthly_population,
    avg_sales_per_item,
], axis=1)
stats.columns = [
    'total_sales',
    'total_retailers',
    'retailers_per_capita',
    'sales_per_retailer',
    'population',
    'price',
]
stats.to_excel('./data/il_retail_stats.xlsx')
