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
import pandas as pd

#-----------------------------------------------------------------------------
# Import the data and match lab results to licensee.
#-----------------------------------------------------------------------------

# Specify where the data lives.
directory = r'E:\cannlytics\data_archive\leaf'
# directory = r'E:\cannlytics\data_archive\leaf'

# Read in the lab result data.
file_name = f'{directory}\LabResults_0\LabResults_0.csv'
lab_data = pd.read_csv(
    file_name,
    sep='\t',
    encoding='utf-16',
    nrows=10000, # FIXME: Read in all the data!
)

# Read in the licensee data.
file_name = f'{directory}\Licensees_0\wa_licensee_data.xlsx'
license_data = pd.read_excel(file_name)

# Combine the data sets.
data = pd.merge(
    left=lab_data,
    right=license_data,
    how='left',
    left_on='for_mme_id',
    right_on='global_id'
)

#-----------------------------------------------------------------------------
# Create dummy variable for Eastern vs Western Washington
#-----------------------------------------------------------------------------

eastern_wa = [
    'Clallam County',
    'Clark County',
    'Columbia County',
    'Cowlitz County',
    'Ferry County',
    'Grays Harbor County',
    'Island County',
    'Jefferson County',
    'King County',
    'Kitsap County',
    'Lewis County',
    'Mason County',
    'Pacific County',
    'Pierce County',
    'San Juan County',
    'Skagit County',
    'Skamania County',
    'Snohomish County',
    'Thurston County',
    'Wahkiakum County',
    'Whatcom County',
]

western_wa = [
    'Adams County',
    'Asotin County',
    'Benton County',
    'Douglas County',
    'Chelan County',
    'Garfield County',
    'Grant County',
    'Okanogan County',
    'Kittitas County',
    'Klickitat County',
    'Whitman County',
    'Spokane County',
    'Pend Oreille County',
    'Stevens County',
    'Lincoln County',
    'Franklin County',
    'Walla Walla County',
    'Yakima County',
]

# Create region dummy variable.
data['eastern_wa'] = 0
for index, values in data.iterrows():
    if values.county in eastern_wa:
        data.at[index, 'eastern_wa'] = 1
    else:
        data.at[index, 'eastern_wa'] = 0


#-----------------------------------------------------------------------------
# Run a regression of THC concentration on tier size.
# https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
# https://stackoverflow.com/questions/11587782/creating-dummy-variables-in-pandas-for-python
# https://stackoverflow.com/questions/50733014/linear-regression-with-dummy-categorical-variables
# https://www.statsmodels.org/stable/examples/notebooks/generated/contrasts.html
#-----------------------------------------------------------------------------

# Dummy variable regressions with `statsmodels`.
from statsmodels.formula.api import ols
fit = ols('thc_percent ~ eastern_wa', data=data).fit() 
print(fit.summary())

# Optional: Dummy variable regression with `pandas`.
