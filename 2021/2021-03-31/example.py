"""
Forecast Colorado Sales and Plants Grown | Cannabis Data Science

Author: Keegan Skeate
Created: Wed Mar 24 2021
Copyright (c) 2017 - 2021 Keegan Skeate

License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Description:
    
    Forecast sales and total plants grown in Colorado in 2021.

Resources:
    
    https://stackoverflow.com/questions/59882714/python-generating-a-list-of-dates-between-two-dates/59882807

"""

# External imports
import pandas as pd

# Relative imports
from arima import arima_min_rmse_forecast

#-----------------------------------------------------------------------------
# Import the data.
#-----------------------------------------------------------------------------

data = pd.read_excel(
    './data/co_report_data.xlsx',
     col=0
)
data.index = pd.date_range(
    '2014-01-01',
   '2020-07-01',
    freq='m'
)


#-----------------------------------------------------------------------------
# Forecast the series.
#-----------------------------------------------------------------------------

total_revenue_forecast = arima_min_rmse_forecast(
    data.total_plants.values,
    lag_order=3,
    hold_out_period=6,
    forecast_steps=18,
    verbose=True
)
total_revenue_forecast = pd.Series(total_revenue_forecast)
total_revenue_forecast.index = pd.date_range(
    '2020-07-01',
   '2022-01-01',
    freq='m'
)


#-----------------------------------------------------------------------------
# ARMA estimation.
#-----------------------------------------------------------------------------

# from statsmodels.tsa.arima_model import ARIMA
# model = ARIMA(data.total_revenue.values, order=(4, 0, 6))
# model_fit = model.fit(disp=0)

#-----------------------------------------------------------------------------
# Plot the data.
#-----------------------------------------------------------------------------

data.total_revenue.plot()
total_revenue_forecast.plot()

