"""
Hemp Analysis | Cannabis Data Science

Author: Keegan Skeate
Contact: <keegan@cannlytics.com>
Created: 6/30/2021
Updated: 6/30/2021
License: MIT License <https://opensource.org/licenses/MIT>
Data Sources:
    - Midwestern Hemp Database: https://extension.illinois.edu/global/midwestern-hemp-database
Resources:
    - https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html
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
    './data/midwestern_hemp_database.xlsx',
    index_col=0,
    sheet_name='Cultivar Cannabinoid Data',
)
data = data.sort_index()

#--------------------------------------------------------------------------
# Rule 1. Look at the data.
#--------------------------------------------------------------------------

# Calculate high THC failures (0 for pass, 1 for failure).
data['fail'] = (data['total_thc'] >= 0.3).astype(int)

# Look at failure rates conditional on state.
# FIXME: Strip spaces out of state names
states = list(data['state'].unique())
for state in states:
    avg = data.loc[data.state == state]['fail'].mean()
    print(state, 'average hemp failure rate:', avg)

#--------------------------------------------------------------------------
# Analyze hemp failure rates (for high THC).
#--------------------------------------------------------------------------

# Estimate a logistic regression of failure on factors such as state, county,
# and sample date.


#--------------------------------------------------------------------------
# Predict hemp failure rates (for high THC).
#--------------------------------------------------------------------------
