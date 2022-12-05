"""
VAR Forecast with Massachusetts Cannabis Data
Cannabis Data Science

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 10/27/2021
Updated: 10/30/2021
License: MIT License <https://opensource.org/licenses/MIT>

Objective:
    
    1. Forecast cannabis sales (and plants)
    in Massachusetts for the remainder of 2021 and for 2022
    using Box-Jenkins methodology (ARIMA modelling).
    
    2. Visualize and save the forecasts.
    
    3. Compare the forecasts with actual data in the future.

Data Sources:

    MA Cannabis Control Commission
    - Average Monthly Price per Ounce for Adult-Use Cannabis: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/rqtv-uenj
    - Plant Activity and Volume: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/j3q7-3usu

    Fed Fred
    - Federal funds interest rate: https://fred.stlouisfed.org/series/FEDFUNDS
    - Interest rates: https://www.federalreserve.gov/releases/h15/

"""
# External imports
from dotenv import dotenv_values
from fredapi import Fred
import numpy as np
import pandas as pd
import requests
from statsmodels.tsa.api import VAR

# Internal imports
from utils import (
    end_of_period_timeseries,
    format_millions,
    reverse_dataframe,
)


#--------------------------------------------------------------------------
# Get MA public cannabis data.
#--------------------------------------------------------------------------

# Setup Socrata API, get the App Token, and define the headers.
config = dotenv_values('./.env')
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

# Get the monthly average price per ounce.
url = f'{base}/rqtv-uenj.json'
params = {'$limit': 10000, '$order': 'date DESC'}
response = requests.get(url,  headers=headers, params=params)
prices = pd.DataFrame(response.json(), dtype=float)
prices = reverse_dataframe(prices)
# prices = end_of_period_timeseries(prices)
# prices.set_index('date')
prices.index = pd.to_datetime(prices.date)
prices = end_of_period_timeseries(prices)

#--------------------------------------------------------------------------
# Get supplemental data from FRED (Federal Reserve Economic Data).
#--------------------------------------------------------------------------

# Initialize Fred client.
config = dotenv_values('./.env')
fred = Fred(api_key=config.get('FRED_API_KEY'))

# Find the observation time start.
observation_start = prices.index.min()

# Get the Federal funds interest rate.
interest_rate = fred.get_series('FEDFUNDS', observation_start=observation_start)
interest_rate = end_of_period_timeseries(interest_rate)

#--------------------------------------------------------------------------
# Estimate a VAR model.
#--------------------------------------------------------------------------

# Calculate inflation.
lag_price = prices.avg_1oz.shift(1)
inflation = (prices.avg_1oz - lag_price) / lag_price
inflation = inflation[1:]

# Restrict the timeframe to match available interest rate and price data.
output = monthly_total_production.loc[
    (monthly_total_production.index >= inflation.index.min()) &
    (monthly_total_production.index <= inflation.index.max())
]

# Restrict interest rate to timeframe where inflation is calculated.
interest_rate = interest_rate.loc[
    interest_rate.index >= inflation.index.min()
]

# VAR Vector
variables = [
    output.sales,
    inflation,
    interest_rate,
]
vector = np.column_stack(variables)

# Fit a VAR regression.
model = VAR(vector)
results = model.fit(1)
print(results.summary())

# Fit the best in-sample predicting VAR.
model.select_order(6)
results = model.fit(maxlags=6, ic='bic')
lag_order = results.k_ar
print('Best lag order:', results.k_ar)

# Create a forecast.
freq = 'm'
horizon = 15
last_obs = vector[-lag_order:]
forecast = results.forecast(last_obs, horizon)
forecast_intervals = results.forecast_interval(
    last_obs,
    horizon,
    alpha=0.2
)
forecast = np.append(forecast, forecast_intervals[1], axis=1)
forecast = np.append(forecast, forecast_intervals[2], axis=1)

# Format the forecast data, starting the date in the next periods.
variables = [
    'output_forecast',
    'inflation_forecast',
    'interest_rate_forecast',
    'output_forecast_lower_bound',
    'inflation_forecast_lower_bound',
    'interest_rate_forecast_lower_bound',
    'output_forecast_upper_bound',
    'inflation_forecast_upper_bound',
    'interest_rate_forecast_upper_bound',
]
forecast_data = pd.DataFrame(forecast, columns=variables)
forecast_data.index = pd.date_range(
    output.index.max(),
    periods=horizon + 1,
    freq=freq
)[1:]

# Save the data.
# forecast_data.to_excel('./data/forecasts.xlsx')

# Calculate yearly aggregates.
expected_sales = pd.concat([output.sales, forecast_data.output_forecast], axis=0)
expected_annual_sales = expected_sales.resample('Y').sum()

#--------------------------------------------------------------------------
# Plot the VAR model.
#--------------------------------------------------------------------------

# Plotting libraries.
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
# from curlyBrace import curlyBrace

# Define the plot style.
plt.style.use('fivethirtyeight')

# Set the chart's font.
plt.rcParams['font.family'] = 'Times New Roman'

# Define the chart colors.
palette = sns.color_palette('tab10')
primary_color = palette[0]
secondary_color = palette[-1]

# Create a figure.
fig, ax = plt.subplots(figsize=(15, 5))

# Plot the historical values.
output.sales.plot(
    ax=ax,
    color=primary_color,
)

# Plot the forecasts.
forecast_data['output_forecast'].plot(
    ax=ax,
    style='--',
    color=secondary_color
)

# Plot the confidence bounds.
ax.fill_between(
    forecast_data.index,
    forecast_data['output_forecast_lower_bound'],
    forecast_data['output_forecast_upper_bound'],
    color=secondary_color,
    alpha=0.1
)

# Plot aggregates.
# ax2 = ax.twinx()
# ax2.bar(
#        expected_annual_sales.index.shift(-182, freq='D')[1:],
#        expected_annual_sales.values[1:],
#        width=20,
# )
# ax2.set_ylabel('Annual Sales Total', color='b')

# Style the chart.
text_color = '#1a1a1a'
plt.rcParams['axes.edgecolor'] = text_color
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = text_color
plt.rcParams['ytick.color'] = text_color
plt.rcParams['text.color'] = text_color

# Hide unnecessary spines and ticks.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.tick_params(axis='both', which='major', labelsize=18)

# Format the axes.
plt.xlabel('')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Add a title.
title = 'Massachusetts Cannabis Sales Forecast'
plt.title(title, fontsize=24, pad=10)

# Format the Y-Axis.
yaxis_format = FuncFormatter(format_millions)
ax.yaxis.set_major_formatter(yaxis_format)
plt.gca().set(ylim=0)
plt.setp(ax.get_yticklabels()[0], visible=False)

# Add figure notes.
notes = """Data: 10,824 brand analyses between 3/24/2015 and 9/15/2021.
Data Source: Connecticut Medical Marijuana Brand Registry.
Notes: The terpenes β-eudesmol, fenchone, and camphor were present in more than 95% of
samples, so they were excluded from the top ten because they appear to be ubiquitous."""
plt.figtext(0.05, -0.25, notes, ha='left', fontsize=16)

# TODO: Annotate closure of cannabis retail in April of 2020.


# Optional: Annotate annual totals.
# ax.annotate(
#     'SDL',
#     xy=(0.5, 0.90),
#     xytext=(0.5, 1.00),
#     xycoords='axes fraction', 
#     fontsize=18,
#     ha='center',
#     va='bottom',
#     bbox=dict(boxstyle='square', fc='white'),
#     arrowprops=dict(arrowstyle='-[, widthB=7.0, lengthB=1.5', lw=2.0)
# )
# draw_brace(ax, (0, 1), pd.to_datetime('2019-01-01'), 'large brace')

plt.show()

# Save the figure.
# plt.margins(1, 1)
# plt.savefig(
#     'figures/ma_var_forecast.png',
#     dpi=300,
#     bbox_inches='tight',
#     pad_inches=0.75,
#     transparent=False,
# )

# Show the figure.
# plt.show()
