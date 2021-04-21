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

# External imports.
import numpy as np
import pandas as pd
from dotenv import dotenv_values
from fredapi import Fred

from statsmodels.tsa.api import VAR

#-----------------------------------------------------------------------------
# Import the data.
#-----------------------------------------------------------------------------

# Read in the data with the first column as the index.
data = pd.read_excel('./data/oregon_cannabis_data.xlsx', index_col=0)

# Restrict the data to 2017 and onwards.
data = data.loc[data.index >= pd.to_datetime('2017-01-01')]


#-----------------------------------------------------------------------------
# Calculate inflation by month.
#-----------------------------------------------------------------------------

# Specificy basket items.
basket_items = ['flower', 'concentrate']

# Calculate an average consumer's basket.
flower_share_of_sales = data['flower_sales'] / data['total_sales']
concentrate_share_of_sales = data['concentrate_sales'] / data['total_sales']

# Calculate the CPI.
cpi = flower_share_of_sales * data['retail_flower_price'] + \
      concentrate_share_of_sales * data['retail_concentrate_price']

# Get the prior period's CPI.
lag_cpi = cpi.shift(1)

# Calculate inflation.
data['inflation'] = (cpi - lag_cpi) / lag_cpi


#-----------------------------------------------------------------------------
# Get the interest rate from Fred
# Get your API key from: http://research.stlouisfed.org/fred2/
#-----------------------------------------------------------------------------

# Read in a Fred API key.
config = dotenv_values('../.env')

# Get the effective federal funds rate from Fred.
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
print('Best lag order:', results.k_ar)

# Create a forecast. (FIXME:)
# forecast = results.forecast(data.values[-lag_order:], 5)

# Show the data!
forecasts = results.plot_forecast(9)










#-----------------------------------------------------------------------
# Drafts
#-----------------------------------------------------------------------

# Local imports.
# from VAR import VAR, VAR_forecast

# Estimate a VAR.
# Future work: Estimate multiple VAR models and select the best in-sample
# predictor.
# regressions = VAR(vector, lag_order=1)

# # VAR Forecasting                                                                                     
# inflation_fprecast = VAR_forecast(
#     vector,
#     regressions,
#     lag_order=6,
#     horizon=9,
#     shock=None
# )

# Forecasts in Levels                                     
# Forecast_Businesses = growth_to_levels(Total_Businesses.iloc[-1],
#                                         VAR_Forecasts_Business_Revenue[:,0])        
# Forecast_Revenue = growth_to_levels(Total_Revenue.iloc[-1],
#                                     VAR_Forecasts_Business_Revenue[:,1])


#-----------------------------------------------------------------------
# Optional: Forecast impact response function.
#-----------------------------------------------------------------------


#  # IRF's
# Residuals_Business_Revenue = np.column_stack([VAR_Business_Revenue['Eq1'].resid,
#                                               VAR_Business_Revenue['Eq2'].resid])                          
# Omega = cov_matrix(Residuals_Business_Revenue)
# S = np.linalg.cholesky(Omega)
# error = np.array(([0.00034/S[0][0]],[0.0]))
# shock = np.dot(S, error)
# IRF_Business_Revenue = IRF(Vector_Business_Revenue,
#                            VAR_Business_Revenue,
#                            lag_order=6,
#                            horizon=12,
#                            shock=shock)


#-----------------------------------------------------------------------
# Future work: Estimate April month effect
#-----------------------------------------------------------------------

