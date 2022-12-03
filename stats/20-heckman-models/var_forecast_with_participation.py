"""
Forecasting Cannabis Sales in Oklahoma
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 4/8/2022
Updated: 4/9/2022
License: MIT License <https://opensource.org/licenses/MIT>

Data sources:

    - SQ788 Tax Rate (7%)
    https://ballotpedia.org/Oklahoma_State_Question_788,_Medical_Marijuana_Legalization_Initiative_(June_2018)/Full_article#:~:text=A%207%20percent%20tax%20was,growing%20operations%2C%20and%20processing%20operations.

    - OMMA Reporting Data
    https://oklahoma.gov/omma/administration/archived-reporting-data.html

"""
# External imports.
from dotenv import dotenv_values
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import abline_plot
from statsmodels.tsa.api import VAR


# Internal imports.
from utils import (
    format_millions,
    get_state_population,
)


# Specify where your data lives.
DATA_DIR = '../.datasets'


#--------------------------------------------------------------------------
# Look at Oklahoma data.
#--------------------------------------------------------------------------

# Read data.
patient_data = pd.read_excel(f'{DATA_DIR}/oklahoma/patient_data.xlsx', index_col=0)

# Estimate total sales.
patient_data['total_sales'] = patient_data['sq788_tax'] * 100 / 7

# Calculate changes in sales.
change = patient_data.apply(np.log)[1:]
pct_change = patient_data.pct_change()[1:]

# Regress sales on patients.
Y = pct_change['total_sales']
x = pct_change['patients']
X = sm.add_constant(x)
regression = sm.OLS(Y, X).fit()
print(regression.summary())

# Visualize the regression.
ax = pct_change.plot(
    x='patients',
    y='total_sales',
    kind='scatter'
)
abline_plot(
    model_results=regression,
    ax=ax
)
plt.show()


#--------------------------------------------------------------------------
# Iterate on past forecasts.
#--------------------------------------------------------------------------

# Look at past forecasts and actual data.
filename = f'{DATA_DIR}/oklahoma/ok_revenue_forecast_2021-06-02.xlsx'
past_forecast_data = pd.read_excel(filename, index_col=0)
past_forecast = past_forecast_data.revenue_forecast[-6:]

# Visualize the forecast.
fig, ax = plt.subplots(figsize=(12, 8))
past_forecast.plot()
patient_data['total_sales'].plot()
yaxis_format = FuncFormatter(format_millions)
ax.yaxis.set_major_formatter(yaxis_format)
plt.show()


def RMSE(actual, predicted):
    """Calculate the root mean squared error of a series given
    the actual and predicted values.
    Args:
        actual (Series): The actual timeseries.
        predicted (Series): The predicted timeseries.
    Returns:
        (float): The RMSE.
    ."""
    t = len(predicted)
    se = 0
    for index, y_hat in predicted.iteritems():
        y = actual.loc[index]
        se += (y - y_hat) ** 2
    rmse = np.sqrt(se * (1 / t))
    return rmse


# Calculate RMSE for Oklahoma revenue forecast.
past_RMSE = RMSE(patient_data['total_sales'], past_forecast)

# Try to make a better in sample forecast given the new data.
train = patient_data.loc[patient_data.index < past_forecast.index.min()]
arima_model = auto_arima(
    train['total_sales'],
    start_p=0,
    d=0,
    start_q=0, 
    max_p=6,
    max_d=6,
    max_q=6,
    seasonal=False, 
    error_action='warn',
    trace=True,
    supress_warnings=True,
    stepwise=True,
)
print(arima_model.summary())
insample_forecast = pd.DataFrame(
    arima_model.predict(n_periods=len(past_forecast)),
    index=past_forecast.index
)
insample_forecast.columns = ['insample_forecast']
new_forecast = insample_forecast['insample_forecast']

# Plot the forecasts.
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(patient_data['total_sales'], label='Historic')
plt.plot(past_forecast, label='Past Forecast')
plt.plot(new_forecast, label='New Forecast')
plt.legend(loc='best')
plt.show()

# See if including more data makes a better prediction.
new_RMSE = RMSE(patient_data['total_sales'], new_forecast)
print('Better model:', past_RMSE > new_RMSE)


#--------------------------------------------------------------------------
# Estimate participation and sales simultaneously using a VAR.
#--------------------------------------------------------------------------

# Get Oklahoma monthly population.
config = dotenv_values('../.env')
fred_api_key = config['FRED_API_KEY']
population = get_state_population(fred_api_key, 'OK', '2020-01-01')
monthly_index = population.resample('M').mean().index
monthly_population = pd.Series(
    np.linspace(3962031.0, 3986639.0, len(monthly_index)),
    index=monthly_index,
)

# Extend population as a trend.
t = range(0, len(patient_data))
population_data = pd.DataFrame({
    'population': monthly_population,
    't': t[:len(monthly_population)],
})
regression = sm.formula.ols(formula='population ~ t', data=population_data).fit()
population_trend = regression.params['t']
trend = pd.DataFrame(sm.add_constant(t), columns=['const', 't'])
population_forecast = regression.predict(trend)
population_forecast.index = patient_data.index

# Calculate the proportion of people who are patients in Oklahoma.
participation = patient_data['patients'].div(population_forecast)

# Estimate a VAR to predict sales in Oklahoma in 2022.
variables = [
    train['total_sales'],
    participation.loc[participation.index <= train.index.max()],
]
vector = np.column_stack(variables)
model = VAR(vector)
results = model.fit(1)
print(results.summary())

# FIXME: Fit the best in-sample predicting VAR.
# model.select_order(6)
# results = model.fit(ic='bic')

# Create a forecast.
horizon = len(past_forecast)
lag_order = results.k_ar
var_forecast = results.forecast(vector[-lag_order:], horizon)
var_forecast = pd.DataFrame(
    var_forecast,
    columns=['revenue_forecast', 'participation_forecast']
)
var_forecast.index = past_forecast.index

# Plot the sales forecasts.
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(patient_data['total_sales'], label='Historic')
plt.plot(past_forecast, label='Past Forecast')
plt.plot(new_forecast, label='New Forecast')
plt.plot(var_forecast['revenue_forecast'], label='VAR Forecast')
plt.legend(loc='best')
plt.show()

# Plot the patients forecasts.
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(participation, label='Historic Participation')
plt.plot(var_forecast['participation_forecast'], label='VAR Forecast')
plt.legend(loc='best')
plt.show()

# See if including more data makes a better prediction.
var_RMSE = RMSE(patient_data['total_sales'], var_forecast['revenue_forecast'])
print('Better model:', past_RMSE > new_RMSE  > var_RMSE)


#--------------------------------------------------------------------------
# Augment the data with more explanatory variables.
#--------------------------------------------------------------------------

# Optional: Estimate sales per capita.

from fredapi import Fred

# Get Fed interest rate.
fred = Fred(api_key=fred_api_key)
interest_rate = fred.get_series('FEDFUNDS', observation_start='2020-01-01')

# Get the CPI and calculate the national inflation rate.
cpi = fred.get_series('CPIAUCSL', observation_start='2019-12-01')
inflation = cpi.pct_change()[1:] * 100

# Estimate a VAR to predict sales in Oklahoma in 2022.
variables = [
    train['total_sales'],
    participation.loc[participation.index <= train.index.max()],
    interest_rate.loc[interest_rate.index <= train.index.max()],
    inflation.loc[inflation.index <= train.index.max()],
]
vector = np.column_stack(variables)
model = VAR(vector)
results = model.fit(1)
print(results.summary())

# Create a forecast.
horizon = len(past_forecast)
lag_order = results.k_ar
varx_forecast = results.forecast(vector[-lag_order:], horizon)
varx_forecast = pd.DataFrame(
    varx_forecast,
    columns=[
        'revenue_forecast',
        'participation_forecast',
        'interest_rate_forecast',
        'inflation_forecast',
    ]
)
varx_forecast.index = past_forecast.index

# Plot the sales forecasts.
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(patient_data['total_sales'], label='Historic')
plt.plot(past_forecast, label='Past Forecast')
plt.plot(new_forecast, label='New Forecast')
plt.plot(varx_forecast['revenue_forecast'], label='VARX Forecast')
plt.legend(loc='best')
plt.show()

# Plot the patients forecasts.
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(participation, label='Historic Participation')
plt.plot(var_forecast['participation_forecast'], label='VAR Forecast')
plt.plot(varx_forecast['participation_forecast'], label='VARX Forecast')
plt.legend(loc='best')
plt.show()

# See if including more data makes a better prediction.
varx_RMSE = RMSE(patient_data['total_sales'], varx_forecast['revenue_forecast'])
print('Better model:', past_RMSE > new_RMSE  > varx_RMSE)

# Optional: Get and include fertilizer prices.

# Optional: Get and include census variables?


#--------------------------------------------------------------------------
# Use the best model to predict participation and sales in 2022.
#--------------------------------------------------------------------------

# Build the VAR with all the data.
variables = [
    patient_data['total_sales'],
    participation.loc[participation.index <= patient_data.index.max()],
    interest_rate.loc[interest_rate.index <= patient_data.index.max()],
    inflation.loc[inflation.index <= patient_data.index.max()],
]
vector = np.column_stack(variables)
model = VAR(vector)
results = model.fit(1)
print(results.summary())

# Create a forecast.
horizon = 10
lag_order = results.k_ar
final_forecast = results.forecast(vector[-lag_order:], horizon)
final_forecast = pd.DataFrame(
    final_forecast,
    columns=[
        'revenue_forecast',
        'participation_forecast',
        'interest_rate_forecast',
        'inflation_forecast',
    ]
)
final_forecast.index = pd.date_range(
    start='2022-03-01',
    end='2022-12-01',
    freq='MS'
)

# Plot the sales forecasts.
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(patient_data['total_sales'], label='Historic')
plt.plot(final_forecast['revenue_forecast'], label='Forecast')
plt.legend(loc='best')
plt.show()

# Plot the patients forecasts.
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(participation, label='Historic Participation')
plt.plot(final_forecast['participation_forecast'], label='Forecast')
plt.legend(loc='best')
plt.show()

# Forecast sales and participation in 2022 in OK.
forecast_2022 = final_forecast.revenue_forecast.sum() + \
                patient_data.loc[patient_data.index >= '2022-01-01']['total_sales'].sum()
print('Predicted OK Sales in 2022:', format_millions(forecast_2022))

# Record your predictions!
final_forecast.to_excel(f'{DATA_DIR}/oklahoma/ok_forecasts_2022-04-09.xlsx')
