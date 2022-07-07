"""
Rate of returns | Cannabis Data Science

Author: Keegan Skeate
Created: Wed Mar 17 2021

License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Description:
    Estimate the competitive wage for cannabis workers in Colorado.

Resources:
    https://sbg.colorado.gov/med-updates
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

#-----------------------------------------------------------------------------
# Reading in data.
#-----------------------------------------------------------------------------

# Import the data.
data = pd.read_excel(
    './data/co_report_data.xlsx',
     sheet_name='CO Monthly Data',
     col=0
)


#-----------------------------------------------------------------------------
# Estimating production functions.
#-----------------------------------------------------------------------------

# Natural logs.
ln_y_t = np.log(data.retail_revenue)
ln_k_t = np.log(data.retail_plants)

# Simple capital only production function.
model = sm.OLS(ln_y_t, sm.add_constant(ln_k_t)).fit()
print(model.summary())


#-----------------------------------------------------------------------------
# Estimate the real rate of return of capital.
#-----------------------------------------------------------------------------

# Estimate the real rate of return of capital.
alpha = model.params.retail_plants
r = alpha * data.retail_plants ** (alpha - 1) * 100 # As a percentage

# Calculate confidence intervals.
confidence_interval = model.conf_int()
lower_alpha = confidence_interval[0].retail_plants
upper_alpha = confidence_interval[1].retail_plants

# Estimate upper and lower bounds for real rate of return of capital.
lower_r = lower_alpha * data.retail_plants ** (lower_alpha - 1) * 100
upper_r = upper_alpha * data.retail_plants ** (upper_alpha - 1) * 100

# Simple plot.
r.plot()
lower_r.plot()
upper_r.plot()


#-----------------------------------------------------------------------------
# Future work: estimate a Cobb Douglas production function
#-----------------------------------------------------------------------------

# Code to refactor:
    
# # Cobb douglas production function (Retail)
# X = np.column_stack([np.log(k_t), np.log(l_t)])
# X = sm.add_constant(X)
# result = sm.OLS(np.log(y_t), X).fit()
# print(result.summary())
