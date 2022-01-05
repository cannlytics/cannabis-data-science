"""
Forecast Colorado Sales | Cannabis Data Science

Author: Keegan Skeate
Contact: <keegan@cannlytics.com>
Created: 6/16/2021
Updated: 6/16/2021
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
data = pd.read_excel(
    './data/co_data.xlsx',
     sheet_name='CO Monthly Data',
     index_col=0
)

#--------------------------------------------------------------------------
# Calculate the trend.
#--------------------------------------------------------------------------

# Add a time index.
data['t'] = range(0, len(data))

# Run a regression of total revenue on time, where t = 0, 1, ... T.
model = sm.formula.ols(formula='total_revenue ~ t', data=data)
regression = model.fit()
print(regression.summary())

# Plot the trend with total revenue.
data['trend'] = regression.predict()
data[['total_revenue', 'trend']].plot()

#--------------------------------------------------------------------------
# Forecast revenue for 2021
#--------------------------------------------------------------------------

# Fit the best ARIMA model.
arima_model = auto_arima(
    data['total_revenue'],
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
print(arima_model.summary())

# Predict the next 8 months.
forecast_index = pd.date_range(
    start='2021-05-01',
    end='2022-01-01',
    freq='M'
)
forecast = pd.DataFrame(
    arima_model.predict(n_periods=8),
    index=forecast_index
)
forecast.columns = ['forecast_revenue']

# Plot the forecasts.
plt.figure(figsize=(8, 5))
plt.plot(data['total_revenue'], label='Historic')
plt.plot(forecast['forecast_revenue'], label='Forecast')
plt.legend(loc='upper left')
plt.show()

# Calculate estimated total revenue in 2021.
year_to_date_data = data.loc[data.index >= pd.to_datetime('2021-01-01')]
year_forecast = year_to_date_data['total_revenue'].sum() + forecast['forecast_revenue'].sum()
year_forecast_millions = year_forecast / 1000000
print('2021 Colorado Forecast: %.2f million' % year_forecast_millions)

# Save the forecasts.
forecast.to_excel('data/co_revenue_forecast.xlsx')

# TODO: Save Optimistic and Pessimistic forecast plots side-by-side

