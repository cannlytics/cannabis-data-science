"""
ARIMA Forecasting with Massachusetts Cannabis Data | Cannabis Data Science
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 10/27/2021
Updated: 12/11/2022
License: MIT License <https://opensource.org/licenses/MIT>

Objective:
    
    1. Forecast cannabis sales (and plants)
    in Massachusetts for the remainder of 2021 and for 2022
    using Box-Jenkins methodology (ARIMA modelling).
    
    2. Visualize and save the forecasts.
    
    3. Compare the forecasts with actual data in the future.

Data Sources:

    MA Cannabis Control Commission

    - Average Monthly Price per Ounce for Adult-Use Cannabis
    URL: <https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/rqtv-uenj>

    - Plant Activity and Volume
    URL: <https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/j3q7-3usu>

"""
# External imports:
from dotenv import dotenv_values
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import requests

# Internal imports
from utils import (
    end_of_period_timeseries,
    format_millions,
    reverse_dataframe,
)
from arima import (
    arima_min_rmse_forecast,
    arima_min_bic_forecast
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

# Fix outlier that appears to have an extra 0.
outlier = production.loc[production.sales >= 10000000]
production.at[outlier.index, 'sales'] = 0

# Remove negative values.
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
# Forecast 1 week ahead with an ARIMA model.
# Optional: Add holiday fixed effects.
#------------------------------------------------------------------------

# Assign holidays.
cal = calendar()
holidays = cal.holidays(
    start=production.index.min(),
    end=production.index.max()
)
production['holiday'] = production.index.isin(holidays)

# Day of the week fixed effect.

#--------------------------------------------------------------------------
# Forecast with weekly sales alone and an ARIMA model.
#--------------------------------------------------------------------------

# Create weekly series.
weekly_sales = production.sales.resample('W-SUN').sum()
weekly_sales = weekly_sales.loc[
    weekly_sales.index >= pd.to_datetime('2020-06-01')
]
weekly_plants = production['total_planttrackedcount'].resample('W-SUN').mean()
weekly_plants = weekly_plants.loc[
    weekly_plants.index >= pd.to_datetime('2020-06-01')
]
weekly_employees = production.total_employees.resample('W-SUN').mean()
weekly_employees = weekly_employees.loc[
    weekly_employees.index >= pd.to_datetime('2020-06-01')
]

# Example: Estimate an ARIMA model.
# from statsmodels.tsa.arima_model import ARIMA
# model = ARIMA(weekly_sales[-50:-1], order=(4, 0, 6))
# model_fit = model.fit(disp=0)

# Forecast weekly sales.
sales_forecast = arima_min_bic_forecast(
    weekly_sales[:-1],
    # production.sales[-365:],
    lag_order=6,
    forecast_steps=10
)
sales_forecast.index = pd.date_range(
    pd.to_datetime('2021-10-25'),
    periods=10,
    freq='w'
)
sales_forecast.plot()

# # Forecast weekly plants.
# sales_forecast = arima_min_bic_forecast(
#     weekly_sales[-50:-1],
#     lag_order=6,
#     forecast_steps=9
# )
# sales_forecast.plot()

#--------------------------------------------------------------------------
# Plot the ARIMA model forecasts.
#--------------------------------------------------------------------------

# Plotting libraries.
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


def beautiful_timeseries_plot():
    """ """
    fig, ax = plt.subplots(figsize=(15, 5))
    return fig
    

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
weekly_sales[-25:-1].plot(
    ax=ax,
    color=primary_color,
)

# Plot the forecasts.
sales_forecast.plot(
    ax=ax,
    style='--',
    color=secondary_color
)

# Plot the confidence bounds.
# ax.fill_between(
#     forecast_data.index,
#     forecast_data['output_forecast_lower_bound'],
#     forecast_data['output_forecast_upper_bound'],
#     color=secondary_color,
#     alpha=0.1
# )

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

# TODO: Format Y Axis
yaxis_format = FuncFormatter(format_millions)
ax.yaxis.set_major_formatter(yaxis_format)
plt.gca().set(ylim=0)
plt.setp(ax.get_yticklabels()[0], visible=False)

# Add figure notes.
notes = """Data: 510 sales observations between November 2018 and October 2021.
Data Source: MA Cannabis Control Commission."""
plt.figtext(0.05, -0.15, notes, ha='left', fontsize=16)

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

# Save the figure.
plt.margins(1, 1)
plt.savefig(
    'figures/ma_arima_forecast.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.75,
    transparent=False,
)

# Show the figure.
plt.show()
