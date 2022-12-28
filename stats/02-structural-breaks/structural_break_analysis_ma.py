"""
Utilizing Structural Break Models
to Analyze Massachusetts Cannabis Data
Copyright (c) 2021 Cannlytics and the Cannabis Data Science Meetup Group

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 11/6/2021
Updated: 11/6/2021
License: MIT License <https://opensource.org/licenses/MIT>

References:
    
    - The Chow Test — Dealing with Heterogeneity in Python
    https://medium.com/@remycanario17/the-chow-test-dealing-with-heterogeneity-in-python-1b9057f0f07a
    
    - Tests for structural breaks in time-series data
    https://www.stata.com/features/overview/structural-breaks/
    
    - Difference-in-Difference Estimation
    https://www.publichealth.columbia.edu/research/population-health-methods/difference-difference-estimation
    
Data Sources:

    MA Cannabis Control Commission
    - Retail Sales by Date and Product Type: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/xwf2-j7g9
    - Approved Massachusetts Licensees: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/hmwt-yiqy
    - Average Monthly Price per Ounce for Adult-Use Cannabis: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/rqtv-uenj
    - Plant Activity and Volume: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/j3q7-3usu
    - Weekly sales by product type: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/87rp-xn9v

    Fed Fred
    - MA Population: https://fred.stlouisfed.org/series/MAPOP
    - MA Median Income: https://fred.stlouisfed.org/series/MEHOINUSMAA646N
    - MA Income per Capita: https://fred.stlouisfed.org/series/MAPCPI

Objective:
    
    1) Test if there has been a structural break in consumer
    preferences. Perhaps from a structural change in sales per capita
    or sales as a percent of gross domestic income.
    
    2) Test if there has been a structural break in the production function.
"""
# External imports.
from dotenv import dotenv_values
from fredapi import Fred
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import abline_plot

# Internal imports.
from utils import (
    end_of_period_timeseries,
    reverse_dataframe,
)

#--------------------------------------------------------------------------
# Get all MA public cannabis data.
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

# Get the products.
url = f'{base}/xwf2-j7g9.json'
params = {'$limit': 10000, '$order': 'saledate DESC'}
response = requests.get(url,  headers=headers, params=params)
products = pd.DataFrame(response.json(), dtype=float)
products = reverse_dataframe(products)
products.set_index('saledate', inplace=True)
product_types = list(products.productcategoryname.unique())


#--------------------------------------------------------------------------
# Get supplemental data from FRED (Federal Reserve Economic Data).
#--------------------------------------------------------------------------

# Initialize Fred client.
config = dotenv_values('../.env')
fred = Fred(api_key=config.get('FRED_API_KEY'))

# Define the observation time start.
observation_start = pd.to_datetime('2018-01-01')

# Get the Resident Population in Massachusetts.
# Conjecturing (assuming) that population remains the same in 2021 (as in 2020).
population = fred.get_series('MAPOP', observation_start=observation_start)
population = end_of_period_timeseries(population)
# population.at[pd.to_datetime('2021-01-31')] = 6893.574

# Get the Per Capita Personal Income in Massachusetts.
per_capita_income = fred.get_series('MAPCPI', observation_start=observation_start)
per_capita_income = end_of_period_timeseries(per_capita_income)

# Get the Median Household Income in Massachusetts.
median_income = fred.get_series('MEHOINUSMAA646N', observation_start=observation_start)
median_income = end_of_period_timeseries(median_income)

#--------------------------------------------------------------------------
# Calculate per capita cannabis sales and cannabis sales as a percent of income.
#--------------------------------------------------------------------------

production['sales_percent_of_income'] = 0
for day, values in production.iterrows():
    year = day.year
    current_population = population.loc[
        population.index <= pd.to_datetime(f'{year}-01-31')
    ]
    sales_percent_of_income = values.sales / (current_population * 1000)
    value = sales_percent_of_income[0]
    if np.isnan(value):
        value = 0
    production.at[day, 'sales_percent_of_income'] = value

# Plot weekly.
production['sales_percent_of_income'].resample('W-SUN').mean().plot()
plt.show()

# Plot monthly.
production['sales_percent_of_income'].resample('M').mean().plot()
plt.show()

#--------------------------------------------------------------------------
# Test for structural breaks.
#--------------------------------------------------------------------------

# Define entire time period in analysis.
data = production.loc[
    (production.index >= pd.to_datetime('2019-01-01')) &
    (production.index < pd.to_datetime('2020-12-31' ))
]

# Optional: Exclude closed period.
data = data.loc[
    (data.index <= pd.to_datetime('2020-03-26')) |
    (data.index >= pd.to_datetime('2020-05-25' ))
]

# Create a trend.
data['t'] = np.arange(0, len(data))

# Add a constant.
data = sm.add_constant(data)

# Test for various structural breaks.
y = data['sales']
X = data[['const', 't']]

# First range.
first_range = data.loc[
    (data.index >= pd.to_datetime('2019-01-01')) &
    (data.index < pd.to_datetime('2020-03-26'))
]
y_a = first_range['sales']
X_a = first_range[['const', 't']]

# Second range.
second_range = data.loc[
    (data.index >= pd.to_datetime('2020-05-25')) &
    (data.index < pd.to_datetime('2021-10-25'))
]
y_b = second_range['sales']
X_b = second_range[['const', 't']]


# Singular model
singular_model = sm.OLS(y,X).fit()
RSSd = singular_model.ssr

# First period model.
first_period_model = sm.OLS(y_a, X_a).fit()
RSSb = first_period_model.ssr

# Second period model.
second_period_model = sm.OLS(y_b, X_b).fit()
RSSnb = second_period_model.ssr

# Calculate Chow statistic.
J = X.shape[1]
k = X_b.shape[1]
N1 = X_a.shape[0]
N2 = X_b.shape[0]
numerator = ((RSSd - (RSSb + RSSnb))/J)
denomenator = ((RSSb + RSSnb)/(N1 + N2 - 2 * k))
chow = numerator/denomenator

# Calculate the p-value for the Chow statistic.
p_value = scipy.stats.f.cdf(chow, J, N1 + N2 - 2 *k)
print(f'{p_value:.2f}% confidence the models are the same.')

# Plot singular regression.
ax = data.plot(x='t', y='sales', kind='scatter')
abline_plot(model_results=singular_model, ax=ax)

# Plot first-period regression.
ax1 = first_range.plot(x='t', y='sales', kind='scatter')
abline_plot(model_results=first_period_model, ax=ax1)

# Plot second-period regression.
ax2 = second_range.plot(x='t', y='sales', kind='scatter')
abline_plot(model_results=second_period_model, ax=ax2)

# First period plot.
p = first_period_model.params
plt.scatter(first_range.t, first_range.sales, color='r')
plt.plot(first_range.t, p.const + p.t * first_range.t, color='r')

# Second period plot.
p = second_period_model.params
plt.scatter(second_range.t, second_range.sales, color='g')
plt.plot(second_range.t, p.const + p.t * second_range.t, color='g')
