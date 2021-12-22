"""
Calculate Washington State Cannabis Retail Statistics
Cannabis Data Science Meetup Group
Copyright (c) 2021 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 12/15/2021
Updated: 12/15/2021
License: MIT License <https://opensource.org/licenses/MIT>

Objective:
    
    Calculate various retail statistics using Washington State cannabis data.
    Measure the market concentration with the HHI and CR5 over time.
    
Data Sources:
    
    WA State Cannabis Data
    FOIA request by the Cannabis Observer <https://cannabis.observer/>
    
    - Sales by retailer by month: https://lcb.wa.gov/sites/default/files/publications/Marijuana/sales_activity/2021-12-06-MJ-Sales-Activity-by-License-Number-Traceability-Contingency-Reporting-Retail.xlsx
    - Traceability Data Through November 2021: https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd?page=1
    
    U.S. Bureau of Labor Statistics

    - Resident Population in Washington: https://fred.stlouisfed.org/series/WAPOP

"""
# External imports.
from dotenv import dotenv_values
from fredapi import Fred
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

# Internal imports
from utils import (
    end_of_period_timeseries,
    format_thousands,
    format_millions,
)

#--------------------------------------------------------------------------
# Get WA public cannabis data.
# Currently requires you to download the following dataset to a `data` folder.
# https://lcb.wa.gov/sites/default/files/publications/Marijuana/sales_activity/2021-12-06-MJ-Sales-Activity-by-License-Number-Traceability-Contingency-Reporting-Retail.xlsx
#--------------------------------------------------------------------------

# Read data if already downloaded.
filename = './data/2021-12-06-MJ-Sales-Activity-by-License-Number-Traceability-Contingency-Reporting-Retail.xlsx'
data = pd.read_excel(filename, skiprows=3)

# Remove null values.
data = data.loc[data['License Number'].notnull()]

# Create a date column.
data['date'] = pd.to_datetime(data['Reporting Period'])

# Get the Washington State population.
config = dotenv_values('../.env')
fred_api_key = config.get('FRED_API_KEY')
fred = Fred(api_key=fred_api_key)
observation_start = data['date'].min().isoformat()
population = fred.get_series('WAPOP', observation_start=observation_start)
population = end_of_period_timeseries(population, 'Y')
population = population.multiply(1000) # thousands of people
new_row = pd.DataFrame([population[-1]], index=[pd.to_datetime('2021-12-31')])
population = pd.concat([population, pd.DataFrame(new_row)], ignore_index=False)
monthly_population = population[0].resample('M').mean().pad()

#--------------------------------------------------------------------------
# Look at monthly sales data over time.
#--------------------------------------------------------------------------

# Plot monthly sales.
monthly_production = data.groupby('Reporting Period').sum()
ax = monthly_production['Total Sales'].plot()
yaxis_format = FuncFormatter(format_millions)
ax.yaxis.set_major_formatter(yaxis_format)
plt.show()

# Count total_retailers over time.
monthly_count = data.groupby('Reporting Period').count()
monthly_count['License Number'].plot()
plt.show()

# Calculate average sales per retailer.
sales_per_retailer = monthly_production['Total Sales'] / \
                     monthly_count['License Number']
sales_per_retailer.index = pd.to_datetime(sales_per_retailer.index)
sales_per_retailer = end_of_period_timeseries(sales_per_retailer)
ax = sales_per_retailer.plot()
yaxis_format = FuncFormatter(format_thousands)
ax.yaxis.set_major_formatter(yaxis_format)
plt.show()

# Calculate retailers per 100,000 people over time.
monthly_count.index = pd.to_datetime(monthly_count.index)
monthly_count = end_of_period_timeseries(monthly_count)
retailers_per_capita = monthly_count['License Number'] / \
                       (monthly_population / 100_000)
retailers_per_capita.plot()
plt.show()

#--------------------------------------------------------------------------
# Look at 2020 aggregates.
#--------------------------------------------------------------------------

# Calculate average retailers per capita in 2020.
avg_2020_retailers_per_capita = retailers_per_capita.loc[
    (retailers_per_capita.index >= pd.to_datetime('2020-01-01')) &
    (retailers_per_capita.index < pd.to_datetime('2021-01-01'))
].mean()
print('Retailres per capita in WA in 2020: %.2f' % avg_2020_retailers_per_capita)

# Calculate average revenue per retailer in 2020.
avg_2020_sales = sales_per_retailer.loc[
    (sales_per_retailer.index >= pd.to_datetime('2020-01-01')) &
    (sales_per_retailer.index < pd.to_datetime('2021-01-01'))
].sum()
print('Sales per retailer in WA in 2020: %.2fM' % (avg_2020_sales / 1_000_000))

#--------------------------------------------------------------------------
# Look at market concentration over time.
#--------------------------------------------------------------------------

# Calculate a list of market shares over time.
market_shares = {}
for index, row in data.iterrows():
    period = row['Reporting Period']
    total_sales = monthly_production.loc[monthly_production.index == period]
    market_share = row['Total Sales'] / total_sales['Total Sales']
    shares = market_shares.get(period, [])
    shares.append(market_share[period])
    market_shares[period] = shares

# Calculate the HHI over time.
hhis = []
for key, shares in market_shares.items():
    share_square = [(s * 100) ** 2 for s in shares]
    hhi = sum(share_square)
    hhis.append(hhi)

# Plot the HHI.
hhi_data = pd.Series(hhis)
hhi_data.index = monthly_production.index
monthly_production['hhi'] = hhi_data
monthly_production['hhi'].plot()
plt.show()

# Calculate the concentration ratio, CR5, over time.
cr5s = []
for key, shares in market_shares.items():
    shares.sort()
    top_5 = shares[-5:]
    cr5 = sum(top_5)
    cr5s.append(cr5)

# Plot the CR5.
cr5_data = pd.Series(cr5s) * 100
cr5_data.plot()
plt.show()

#--------------------------------------------------------------------------
# TODO: Create a beautiful visualization.
#--------------------------------------------------------------------------


#--------------------------------------------------------------------------
# Save the data.
#--------------------------------------------------------------------------
total_sales = monthly_production['Total Sales']
total_sales.index = pd.to_datetime(total_sales.index)
total_sales = end_of_period_timeseries(total_sales)
stats = pd.concat([
    total_sales,
    monthly_count['License Number'],
    retailers_per_capita,
    sales_per_retailer,
    monthly_population,
], axis=1)
stats.columns = [
    'total_sales',
    'total_retailers',
    'retailers_per_capita',
    'sales_per_retailer',
    'population',
]
stats.to_excel('./data/wa_retail_stats.xlsx')
