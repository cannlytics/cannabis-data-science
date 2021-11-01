"""
Market Analysis for Massachusetts Data | Cannabis Data Science

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 10/6/2021
Updated: 10/13/2021
License: MIT License <https://opensource.org/licenses/MIT>

Data Sources:

    MA Cannabis Control Commission
    - Retail Sales by Date and Product Type: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/xwf2-j7g9
    - Approved Massachusetts Licensees: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/hmwt-yiqy
    - Average Monthly Price per Ounce for Adult-Use Cannabis: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/rqtv-uenj
    - Plant Activity and Volume: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/j3q7-3usu
    - Weekly sales by product type: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/87rp-xn9v

    Fed Fred
    - MA Gross Domestic Product: https://fred.stlouisfed.org/series/MANQGSP
    - MA Civilian Labor Force: https://fred.stlouisfed.org/series/MALF
    - MA All Employees: https://fred.stlouisfed.org/series/MANA
    - MA Avg. Weekly Wage: https://fred.stlouisfed.org/series/LES1252881600Q
    - MA Minimum Wage: https://fred.stlouisfed.org/series/STTMINWGMA
    - MA Population: https://fred.stlouisfed.org/series/MAPOP
"""
from dotenv import dotenv_values
from fredapi import Fred
import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table

# Internal imports
from utils import (
    end_of_period_timeseries,
    reverse_dataframe,
)


#--------------------------------------------------------------------------
# Get MA public cannabis data.
#--------------------------------------------------------------------------

# Setup Socrata API, get the App Token, and define the headers.
config = dotenv_values('../.env')
app_token = config.get('APP_TOKEN', None)
headers = {'X-App-Token': app_token}
base = 'https://opendata.mass-cannabis-control.com/resource'

# Get production stats (total employees, total plants, etc.) j3q7-3usu
url = f'{base}/j3q7-3usu.json'
params = {'$limit': 2000, '$order': 'activitysummarydate DESC'}
response = requests.get(url,  headers=headers, params=params)
production = pd.DataFrame(response.json(), dtype=float)
production = reverse_dataframe(production)

# Calculate sales difference.
production['sales'] = production['salestotal'].diff()

# FIX: Fix outlier that appears to have an extra 0.
outlier = production.loc[production.sales >= 10000000]
production.at[outlier.index, 'sales'] = 0

# FIX: Remove negative values.
negatives = production.loc[production.sales < 0]
production.at[negatives.index, 'sales'] = 0

# Aggregate daily production data into monthly and quarterly averages.
production['date'] = pd.to_datetime(production['activitysummarydate'])
production.set_index('date', inplace=True)
monthly_avg_production = production.resample('M').mean()
quarterly_avg_production = production.resample('Q').mean()
monthly_total_production = production.resample('M').sum()
quarterly_total_production = production.resample('Q').sum()

# Get licensees data.
url = f'{base}/hmwt-yiqy.json'
params = {'$limit': 10000,  '$order': 'app_create_date DESC'}
response = requests.get(url,  headers=headers, params=params)
licensees = pd.DataFrame(response.json(), dtype=float)

# Get the monthly average price per ounce.
url = f'{base}/rqtv-uenj.json'
params = {'$limit': 10000, '$order': 'date DESC'}
response = requests.get(url,  headers=headers, params=params)
prices = pd.DataFrame(response.json(), dtype=float)
prices = reverse_dataframe(prices)
prices.set_index('date', inplace=True)

# Calculate the average price per specific quantity.
price_per_gram = prices.avg_1oz.astype(float).divide(28).round(2)
price_per_teenth = prices.avg_1oz.astype(float).divide(16).round(2)
price_per_eighth = prices.avg_1oz.astype(float).divide(8).round(2)
price_per_quarter = prices.avg_1oz.astype(float).divide(4).round(2)

#--------------------------------------------------------------------------
# Get supplemental data from FRED (Federal Reserve Economic Data).
#--------------------------------------------------------------------------

# Initialize Fred client.
config = dotenv_values('../.env')
fred = Fred(api_key=config.get('FRED_API_KEY'))

# Find the observation time start.
start = production.activitysummarydate.min()
observation_start = start.split('T')[0]

# Get the civilian labor force in Massachusetts.
labor_force = fred.get_series('MALF', observation_start=observation_start)
labor_force.index = labor_force.index.to_period('M').to_timestamp('M')

# Get total employees in MA.
total_ma_employees = fred.get_series('MANA', observation_start=observation_start)
total_ma_employees = end_of_period_timeseries(total_ma_employees)
total_ma_employees = total_ma_employees.multiply(1000) # Thousands of people

# Get MA population (conjecturing that population remains constant in 2021).
population = fred.get_series('MAPOP', observation_start=observation_start)
population = end_of_period_timeseries(population, 'Y')
population = population.multiply(1000) # Thousands of people
new_row = pd.DataFrame([population[-1]], index=[pd.to_datetime('2021-12-31')])
population = pd.concat([population, pd.DataFrame(new_row)], ignore_index=False)

# Get Average Weekly Earnings of All Employees: Total Private in Massachusetts.
avg_weekly_wage = fred.get_series('SMU25000000500000011SA', observation_start=observation_start)
avg_weekly_wage = end_of_period_timeseries(avg_weekly_wage)

# Get average weekly hours worked in MA.
avg_weekly_hours = fred.get_series('SMU25000000500000002SA', observation_start=observation_start)
avg_weekly_hours = end_of_period_timeseries(avg_weekly_hours)
avg_monthly_hours = avg_weekly_hours.resample('M').sum()

#--------------------------------------------------------------------------
# TODO: Estimate sales, plants, employees in 2021 and 2022,
#--------------------------------------------------------------------------



#--------------------------------------------------------------------------
# TODO: Estimate sales per retialer
# Estimate plants per cultivator
# Estimate employees per licensee
#--------------------------------------------------------------------------

# Use app_create_date to determine when licensees entered
# Note: Doesn't account for exits
