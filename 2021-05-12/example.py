"""
Dummy Variable Analysis | Tier THC Analysis | Cannabis Data Science

Author: Keegan Skeate
Contact: <keegan@cannlytics.com>
Created: Wednesday May 12th, 2021
License: MIT License

Description:

    Predict output, inflation, and interest rates in the Oregon cannabis market.

Data sources:

    https://lcb.app.box.com/s/fnku9nr22dhx04f6o646xv6ad6fswfy9?page=1

Resources:    

    https://stackoverflow.com/questions/20410312/how-to-create-a-lagged-data-structure-using-pandas-dataframe
    https://www.statsmodels.org/dev/vector_ar.html
"""

# External imports
import numpy as np
import pandas as pd
from dotenv import dotenv_values
from fredapi import Fred
from statsmodels.tsa.api import VAR

# Internal imports
from logistics import *

#-----------------------------------------------------------------------------
# Import the data.
#-----------------------------------------------------------------------------

# Specify where the data lives.
directory = r'E:\cannlytics\data_archive\leaf'

# Read in the lab resultdata.
file_name = f'{directory}\LabResults_0\LabResults_0.csv'
lab_data = pd.read_csv(
    file_name,
    sep='\t',
    encoding='utf-16',
    nrows=1000,
)

# 
lab_data.iloc[0]['for_mme_id'] # 'WAWA1.MMCY'

# Read in the licensee data.
file_name = f'{directory}\Licensees_0\Licensees_0.csv'
license_data = pd.read_csv(
    file_name,
    sep='\t',
    encoding='utf-16',
)

# TODO: Get county by zip code

# Combine the data sets.
merged = pd.merge(
    left=lab_data,
    right=license_data,
    how='left',
    left_on='for_mme_id',
    right_on='global_id'
)

#-----------------------------------------------------------------------------
# Match lab results to licensee.
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# Create dummy variable for Eastern vs Western Washington
#-----------------------------------------------------------------------------

eastern_wa = [
    'Clallam',
    'Clark',
    'Columbia',
    'Cowlitz',
    'Ferry',
    'Grays Harbor',
    'Island',
    'Jefferson',
    'King',
    'Kitsap',
    'Lewis',
    'Mason',
    'Pacific',
    'Pierce',
    'San Juan',
    'Skagit',
    'Skamania',
    'Snohomish',
    'Thurston',
    'Wahkiakum',
    'Whatcom',
]

western_wa = [
    'Adams',
    'Asotin',
    'Benton',
    'Douglas',
    'Chelan',
    'Garfield',
    'Grant',
    'Okanogan',
    'Kittitas',
    'Klickitat',
    'Whitman',
    'Spokane',
    'Pend Oreille',
    'Stevens',
    'Lincoln',
    'Franklin',
    'Walla Walla',
    'Yakima',
]



#-----------------------------------------------------------------------------
# Run a regression of THC concentration on tier size.
# https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
# https://stackoverflow.com/questions/11587782/creating-dummy-variables-in-pandas-for-python
# https://stackoverflow.com/questions/50733014/linear-regression-with-dummy-categorical-variables
# https://www.statsmodels.org/stable/examples/notebooks/generated/contrasts.html
#-----------------------------------------------------------------------------

# cannabinoid_d9_thca_percent, thc_percent

# Dummy variable regressions with `statsmodels`.
# from statsmodels.formula.api import ols
# fit = ols('Wage ~ C(Sex_male) + C(Job) + Age', data=df).fit() 
# fit.summary()


# Dummy variable regression with `pandas`.
