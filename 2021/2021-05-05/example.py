"""
Macroeconomic Models | Cannabis Data Science

Author: Keegan Skeate
Contact: <keegan@cannlytics.com>
Created: Tuesday April 13th, 2021

License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Description:

    Predict output, inflation, and interest rates in the Oregon cannabis market.

Data sources:

    https://data.olcc.state.or.us/t/OLCCPublic/views/MarketDataTableau/Prices
    https://data.olcc.state.or.us/t/OLCCPublic/views/MarketDataTableau/StatewideSalesTrend

Resources:    

    https://stackoverflow.com/questions/20410312/how-to-create-a-lagged-data-structure-using-pandas-dataframe
    https://www.statsmodels.org/dev/vector_ar.html
"""

import numpy as np
import pandas as pd

from dotenv import dotenv_values
from fredapi import Fred
from statsmodels.tsa.api import VAR

#-----------------------------------------------------------------------------
# Import the data.
#-----------------------------------------------------------------------------

# Read in the data with the first column as the index, limit to 2017 and onwards.
data = pd.read_excel('./data/oregon_cannabis_data.xlsx', index_col=0)
data = data.loc[data.index >= pd.to_datetime('2017-01-01')]


#-----------------------------------------------------------------------------
# Calculate inflation by month.
#-----------------------------------------------------------------------------

# Specificy basket items.
basket_items = ['flower', 'concentrate']

# Calculate the consumer price index (CPI).
cpi = pd.Series([0] * len(data))
cpi.index = data.index
for item in basket_items:
    share_of_sales = data[f'{item}_sales'] / data['total_sales']
    price = data[f'retail_{item}_price']
    cpi = cpi + (share_of_sales * price)

# Calculate inflation.
lag_cpi = cpi.shift(1)
data['inflation'] = (cpi - lag_cpi) / lag_cpi


#-----------------------------------------------------------------------------
# Get the interest rate from Fred
# Get your API key from: http://research.stlouisfed.org/fred2/
#-----------------------------------------------------------------------------

# Get the effective federal funds rate from Fred with API key saved in .env.
config = dotenv_values('../.env')
fred = Fred(api_key=config['FRED_API_KEY'])
data['interest_rate'] = fred.get_series(
    'FEDFUNDS',
    observation_start='1/1/2017',
    observation_end='4/1/2021'
)


#-----------------------------------------------------------------------------
# Estimate a VAR model.
#-----------------------------------------------------------------------------

# Drop the first observation (missing from inflation calculation).
model_data = data.loc[data.index >= pd.to_datetime('2017-02-01')]

# VAR Vector
variables = [
    model_data['total_sales'],
    model_data['inflation'],
    model_data['interest_rate'],
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
horizon = 9
forecast = results.forecast(vector[-lag_order:], horizon)

# Show the data!
results.plot_forecast(horizon)

# Save the data
variables = [
    'total_sales_forecast',
    'inflation_forecast',
    'interest_rate_forecast',
]
forecast_data = pd.DataFrame(forecast, columns=variables)
forecast_data.index = pd.date_range('2021-04-01', '2022-01-01', freq='m')
forecast_data.to_excel('./data/oregon_forecasts.xlsx')


#-----------------------------------------------------------------------
# Estimate an ARIMA (TODO:)
# https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMA.html?highlight=arima
# https://www.statsmodels.org/devel/generated/statsmodels.tsa.stattools.arma_order_select_ic.html#statsmodels.tsa.stattools.arma_order_select_ic
#-----------------------------------------------------------------------

# from statsmodels.tsa.arima_model import ARIMA
# fit = ARIMA(endog, order, exog).fit()


