"""
Herfindahl–Hirschman Index | Cannabis Data Science

Author: Keegan Skeate
Created: Wed Mar 24 2021

License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Description:
    Estimate the HHI for cannabis cultivators in the Washington state
    cannabis market over time in 2020.

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
    './data/wholesale_sales.xlsx',
     sheet_name='CO Monthly Data',
     col=0
)


#-----------------------------------------------------------------------------
# Calculate total sales by month.
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# Calculate market share for each wholesaler by month.
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# Calculate the HHI by month.
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# Plot the HHI
#-----------------------------------------------------------------------------
