"""
Quick Sales Item Analysis
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 3/29/2022
Updated: 3/29/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: Perform a quick sales items analysis
using a sample of observed sales items.
"""
import pandas as pd
import statsmodels.api as sm

# Read observed sales.
DATA_DIR = '../.datasets'
DATA_FILE = f'{DATA_DIR}/observed_sales.xlsx'
data = pd.read_csv(DATA_FILE, index_col=0)

# Regress price per gram on TAC, type.


# Regress price per mg of TC on type.


# Probit regression of type on TAC.

