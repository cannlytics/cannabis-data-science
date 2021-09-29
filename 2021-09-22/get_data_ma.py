"""
Get Massachusetts Data | Cannlytics

Author: Keegan Skeate
Contact: <keegan@cannlytics.com>
Created: 9/20/2021
Updated: 9/22/2021
License: MIT License <https://opensource.org/licenses/MIT>

Data Sources:

    - Retail Sales by Date and Product Type: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/xwf2-j7g9
    - Approved Massachusetts Licensees: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/hmwt-yiqy
    - Average Monthly Price per Ounce for Adult-Use Cannabis: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/rqtv-uenj
    - Plant Activity and Volume: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/j3q7-3usu
    - Weekly sales by product type: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/87rp-xn9v
"""
import os
from dotenv import dotenv_values
from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt
import requests


#--------------------------------------------------------------------------
# 1. Get the data.
#--------------------------------------------------------------------------

# Get the App Token.
config = dotenv_values('../.env')
app_token = config.get('APP_TOKEN', None)

# Define headers for all requests. ('Content-Type': 'application/json')
headers = {'X-App-Token': app_token}

# Define the base URL.
base = 'https://opendata.mass-cannabis-control.com/resource'
    
# Get sales by product type.
url = f'{base}/xwf2-j7g9.json'
params = {'$limit': 100, '$order': 'saledate DESC'}
response = requests.get(url,  headers=headers, params=params)
products = pd.DataFrame(response.json())

# Get licensees.
url = f'{base}/hmwt-yiqy.json'
params = {'$limit': 1000, '$order': 'app_create_date DESC'}
response = requests.get(url,  headers=headers, params=params)
licensees = pd.DataFrame(response.json())

# Get the monthly average price per ounce.
url = f'{base}/rqtv-uenj.json'
params = {'$limit': 100, '$order': 'date DESC'}
response = requests.get(url,  headers=headers, params=params)
prices = pd.DataFrame(response.json())

# Get production stats (total employees, total plants, etc.) j3q7-3usu
url = f'{base}/j3q7-3usu.json'
params = {'$limit': 100, '$order': 'activitysummarydate DESC'}
response = requests.get(url,  headers=headers, params=params)
production = pd.DataFrame(response.json())

#--------------------------------------------------------------------------
# 2. Clean the data, standardizing variables.
#--------------------------------------------------------------------------

# Initialize Fed Fred.
config = dotenv_values('../.env')
fred = Fred(api_key=config['FRED_API_KEY'])

# Calculate percent of the civilian labor force in Massachusetts.
labor_force = fred.get_series('MALF', observation_start='1/1/2020')

# Calculate total employees as a percent of all employees in MA.
total_ma_employees = fred.get_series('MANA', observation_start='1/1/2020')

# Optional: Correlate total sales with GDP (https://fred.stlouisfed.org/series/MANQGSP)


#--------------------------------------------------------------------------
# 3. Visualize the data.
#--------------------------------------------------------------------------

# plt.plot(production_data.activitysummarydate, production_data.total_employees)
    

