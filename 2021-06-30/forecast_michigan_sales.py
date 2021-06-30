"""
Forecast Michigan Sales | Cannabis Data Science

Author: Keegan Skeate
Contact: <keegan@cannlytics.com>
Created: 6/30/2021
Updated: 6/30/2021
License: MIT License <https://opensource.org/licenses/MIT>
"""
import matplotlib.pyplot as plt
import pandas as pd
from pmdarima.arima import auto_arima
import statsmodels.api as sm

#--------------------------------------------------------------------------
# Read in the data.
#--------------------------------------------------------------------------
    
# Import the data.
data = pd.read_excel('./data/michigan_data.xlsx', index_col=0)
data = data.sort_index()

#--------------------------------------------------------------------------
# Look at the data.
#--------------------------------------------------------------------------

# Plot total sales.
(data.total_sales / 1000000 ).plot(title='Total Michigan Sales (Millions)', figure=plt.figure())

# Plot total transfers.
data.completed_transfers.plot(title='Completed Transfers in Michigan', figure=plt.figure())

# Plot sales per transfer.
data.sales_per_transfer.plot(title='Sales per Transfer in Michigan', figure=plt.figure())

# PLot pounds per transfer.
data.pounds_per_transfer.plot(title='Pounds Sold per Transfer in Michigan', figure=plt.figure())

# Estimate fix-cost transportation costs.
price_per_transfer = 75
estimated_cost = price_per_transfer * data.completed_transfers
(estimated_cost / 1000000 ).plot(title='Estimated Transportation Costs (Millions)', figure=plt.figure())

# Plot estimated transportation costs as a percent of total sales.
transportation_cost_percent = (estimated_cost / data.total_sales * 100)
transportation_cost_percent.plot(
    title='Estimated Transportation Cost as a percent of Sales',
    figure=plt.figure()
)

#--------------------------------------------------------------------------
# Calculate the trends.
#--------------------------------------------------------------------------

# Add a time index.
data['t'] = range(0, len(data))

# Run a regression of total sales on time, where t = 0, 1, ... T.
model = sm.formula.ols(formula='total_sales ~ t', data=data)
regression = model.fit()
# print(regression.summary())

# Plot the trend with total revenue.
data['trend'] = regression.predict()
data[['total_sales', 'trend']].plot(title='Trend in Sales in Michigan')

# FIXME:
# # Run a regression of completed transfers on time, where t = 0, 1, ... T.
# model = sm.formula.ols(formula='completed_transfers ~ t', data=data)
# regression = model.fit()
# # print(regression.summary())

# # Plot the trend with total revenue.
# data['trend'] = regression.predict()
# data[['completed_transfers', 'trend']].plot(title='Trend in Transfers in Michigan')

#--------------------------------------------------------------------------
# Forecast sales for 2021.
#--------------------------------------------------------------------------

# Fit the best ARIMA model to forecast total sales.
arima_model = auto_arima(
    data['total_sales'],
    start_p=0,
    d=None,
    start_q=0, 
    max_p=6,
    max_d=6,
    max_q=6,
    seasonal=False, 
    error_action='warn',
    trace = True,
    supress_warnings=True,
    stepwise = True,
)

# Summary of the model
# print(arima_model.summary())

# Predict the next 8 months.
forecast_index = pd.date_range(
    start='2021-06-01',
    end='2022-01-01',
    freq='M'
)
sales_forecast = pd.DataFrame(
    arima_model.predict(n_periods=len(forecast_index)),
    index=forecast_index
)
sales_forecast.columns = ['total_sales_forecast']

# Plot the total sales foreast.
plt.figure(figsize=(8, 5))
plt.plot(data['total_sales'], label='Historic')
plt.plot(sales_forecast['total_sales_forecast'], label='Forecast')
plt.legend(loc='upper left')
plt.title('Forecast of Sales in Michigan')
plt.show()

# Estimate total sales in 2021.
year_to_date_data = data.loc[data.index >= pd.to_datetime('2021-01-01')]
year_forecast = year_to_date_data['total_sales'].sum() + sales_forecast['total_sales_forecast'].sum()
year_forecast_millions = year_forecast / 1000000
print('Forecast of sales in Michigan in 2021: %.2f million' % year_forecast_millions)

# Save the forecasts.
sales_forecast.to_excel('data/michigan_sales_forecast.xlsx')

#--------------------------------------------------------------------------
# Forecast transfers for 2021.
#--------------------------------------------------------------------------

# Fit an ARIMA model to forecast total transfers.
arima_model = auto_arima(
    data['completed_transfers'][1:],
    start_p=0,
    d=None,
    start_q=0, 
    max_p=6,
    max_d=6,
    max_q=6,
    seasonal=False, 
    error_action='warn',
    trace = True,
    supress_warnings=True,
    stepwise = True,
)

# Summary of the model
# print(arima_model.summary())

# Predict the next 8 months.
forecast_index = pd.date_range(
    start='2021-06-01',
    end='2022-01-01',
    freq='M'
)
transfers_forecast = pd.DataFrame(
    arima_model.predict(n_periods=len(forecast_index)),
    index=forecast_index
)
transfers_forecast.columns = ['completed_transfers_forecast']

# Plot the total sales foreast.
plt.figure(figsize=(8, 5))
plt.plot(data['completed_transfers'], label='Historic')
plt.plot(transfers_forecast['completed_transfers_forecast'], label='Forecast')
plt.legend(loc='upper left')
plt.title('Forecast of Completed Transfers in Michigan')
plt.show()

# Save the forecasts.
transfers_forecast.to_excel('data/michigan_transfers_forecast.xlsx')

#--------------------------------------------------------------------------
# Analyze forecasted sales per transfer.
#--------------------------------------------------------------------------

sales_per_transfer_forecast = sales_forecast.total_sales_forecast / transfers_forecast.completed_transfers_forecast
sales_per_transfer_forecast.plot(
    title='Forecast of Sales per Transefer in Michigan',
    figure=plt.figure()
)

# TODO: Forecast of transportation costs.
estimated_transportation_costs = 75 * transfers_forecast.completed_transfers_forecast
estimated_transportation_cost_percent = estimated_transportation_costs / sales_forecast.total_sales_forecast * 100
estimated_transportation_cost_percent.plot(
    title='Forecast of Transportation Costs as a Percent of Sales in Michigan',
    figure=plt.figure()
)

