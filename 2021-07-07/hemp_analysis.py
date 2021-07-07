"""
Hemp Analysis | Cannabis Data Science

Author: Keegan Skeate
Contact: <keegan@cannlytics.com>
Created: 6/30/2021
Updated: 6/30/2021
License: MIT License <https://opensource.org/licenses/MIT>
Data Sources:
    - Midwestern Hemp Database: https://extension.illinois.edu/global/midwestern-hemp-database
"""
import matplotlib.pyplot as plt
import pandas as pd
from pmdarima.arima import auto_arima
import statsmodels.api as sm

#--------------------------------------------------------------------------
# Read in the data.
#--------------------------------------------------------------------------
    
# Import the data.
data = pd.read_excel('./data/midwestern_hemp_database.xlsx', index_col=0)
data = data.sort_index()

#--------------------------------------------------------------------------
# Look at the data.
#--------------------------------------------------------------------------

# Calculate high THC failures (0 for pass, 1 for failure).

#--------------------------------------------------------------------------
# Analyze hemp failure rates (for high THC).
#--------------------------------------------------------------------------

# Estimate a logistic regression of failure on factors such as state, county,
# and sample date.

#--------------------------------------------------------------------------
# Predict hemp failure rates (for high THC).
#--------------------------------------------------------------------------
