"""
Title | Project

Author: Keegan Skeate
Created: Wed Mar 10 04:20:49 2021

License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Description:
    Estimate the competitive wage for cannabis workers in Colorado.

Resources:

"""
import pandas as pd
from sodapy import Socrata

#-----------------------------------------------------------------------------
# Getting data
#-----------------------------------------------------------------------------

# Unauthenticated client only works with public data sets. Note 'None'
# in place of application token, and no username or password:
# client = Socrata("data.colorado.gov", None)


# Get total occupational licenses.
# Source: https://data.colorado.gov/Regulations/Professional-and-Occupational-Licenses-in-Colorado/7s5z-vewr
# API: https://data.colorado.gov/resource/7s5z-vewr.json
# Docs: https://dev.socrata.com/foundry/data.colorado.gov/7s5z-vewr

# First 2000 results, returned as JSON from API / converted to Python list of
# dictionaries by sodapy.
# results = client.get("7s5z-vewr", limit=10)

# Convert to pandas DataFrame
# licensee_data = pd.DataFrame.from_records(results)


# Get cannabis sales data.
# Source: https://data.colorado.gov/Revenue/State-Retail-Marijuana-Sales-Tax-Revenue-by-County/v9m8-x8dh
# API: https://data.colorado.gov/resource/v9m8-x8dh.json
# Docs: https://dev.socrata.com/foundry/data.colorado.gov/v9m8-x8dh

# First 2000 results, returned as JSON from API / converted to Python list of
# dictionaries by sodapy.
# results = client.get("v9m8-x8dh", limit=2000)

# Convert to pandas DataFrame
# results_df = pd.DataFrame.from_records(results)


#-----------------------------------------------------------------------------
# Calculating the competitive wage [REFACTOR]
#-----------------------------------------------------------------------------


# Import the data.
df = pd.read_excel('./data/colorado-cannabis-data.xlsx',
                    sheet_name='Analysis',
                    # parse_cols =57,
                    col=0)


# Identify the series of interest.
# Revenue                  
Total_Revenue = df['Total-Marijuana-Revenue']             
Medical_Revenue = df['Total-Medical-Marijuana-Revenue']
Retail_Revenue = df['Total-Retail-Marijuana-Revenue'] # Y_t

# Plants
Medical_Plants = df['Average-Cultivated-Medical-Plants']                   
Retail_Plants = df['Average-Cultivated-Retail-Plants']
Total_Plants = Medical_Plants + Retail_Plants # K_t

# Employees
Total_Labor = df['Total-Occupational-Licenses'] # L_t


# Fed Fred Import
from fredapi import Fred
fred = Fred(api_key='redacted')
Weekly_Hours = fred.get_series('SMU08000000500000002')[-40:-4]


# # Create a retail model
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
# """ Variables """
Y_Retail = Retail_Revenue[1:-6]
K_Retail = Retail_Plants[1:-6]
L_Retail = (Total_Labor[1:-6] * Weekly_Hours[1:-6].values * 4)
X = np.column_stack([np.log(K_Retail), np.log(L_Retail)])
X = np.asarray(sm.add_constant(X))

""" Model """
Retail_Model = sm.OLS(np.log(Y_Retail), X).fit()
print(Retail_Model.summary())

""" Results """
Table, Summary, Labels = summary_table(Retail_Model, alpha=0.05)
Predictions = Summary[:,2]
CI_Lower, CI_Upper = Summary[:,4:6].T
PI_Lower, PI_Upper = Summary[:,6:8].T


# Calculate competitive wage
beta = Retail_Model.params[2]
wage = beta*(Y_Retail/L_Retail)
min_wage = .292*(Y_Retail/L_Retail)
max_wage = 1.028 * (Y_Retail/L_Retail)

wage.plot()
min_wage.plot()
max_wage.plot()
