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


#-----------------------------------------------------------------------------
# Forecast the series.
#-----------------------------------------------------------------------------

total_revenue_forecast = arima_min_rmse_forecast(
    data.total_revenue.values,
    lag_order=6,
    hold_out_period=6,
    forecast_steps=18,
    verbose=True
)

