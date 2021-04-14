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

Resources:
    
    https://data.olcc.state.or.us/t/OLCCPublic/views/MarketDataTableau/Prices
    https://data.olcc.state.or.us/t/OLCCPublic/views/MarketDataTableau/StatewideSalesTrend
"""
import pandas as pd

#-----------------------------------------------------------------------------
# Import the data.
#-----------------------------------------------------------------------------

sales_data = pd.read_excel(
    './data/total_sales_data.xlsx',
     # col=0
)

price_data = pd.read_excel(
    './data/total_sales_data.xlsx',
     # col=0
)

#-----------------------------------------------------------------------------
# Calculate inflation by month.
#-----------------------------------------------------------------------------



#-----------------------------------------------------------------------------
# Estimate a VAR model.
#-----------------------------------------------------------------------------

# # VAR Vector
# Vector_Business_Revenue = np.column_stack([Growth_Total_Businesses,
#                                            Growth_Total_Revenue])
# # VAR Model Selection and Estimation
# VAR_Business_Revenue = VAR(Vector_Business_Revenue)                                              
# # VAR Forecasting                                                                                     
# VAR_Forecasts_Business_Revenue = VAR_forecast(Vector_Business_Revenue,
#                                           VAR_Business_Revenue,
#                                           lag_order=6,
#                                           horizon=12,
#                                           shock=None)
# # Forecasts in Levels                                     
# Forecast_Businesses = growth_to_levels(Total_Businesses.iloc[-1],
#                                        VAR_Forecasts_Business_Revenue[:,0])        
# Forecast_Revenue = growth_to_levels(Total_Revenue.iloc[-1],
#                                     VAR_Forecasts_Business_Revenue[:,1])
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

#-----------------------------------------------------------------------------
# Forecast output, inflation, and the interest rate
#-----------------------------------------------------------------------------

