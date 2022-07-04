"""
Analyzing the Profitability of High-Quality Cannabis in Washington State
Cannabis Data Science Meetup Group
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 1/11/2022
Updated: 1/11/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: Using data on analyses performed by labs in Washington State,
this script analyzes the potential profitability of growing high-quality
cannabis and manufacturing high-concentration processed products.
"""
# Standard imports.
import json
from typing import Any, List, Optional

# External imports.
from dateutil import relativedelta
from dotenv import dotenv_values
from fredapi import Fred
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

# Internal imports.
from utils import (
    format_billions,
    format_millions,
)

#------------------------------------------------------------------------------
# Read in the laboratory and licensee data.
#------------------------------------------------------------------------------

# Define lab datasets.
lab_datasets = ['LabResults_0', 'LabResults_1', 'LabResults_2']

# Read in the lab result data.
shards = []
for dataset in lab_datasets:
    lab_data = pd.read_csv(
        f'../.datasets/{dataset}.csv',
        sep='\t',
        encoding='utf-16',
        nrows=1000,
    )
    shards.append(lab_data)

# Aggregate lab data.
data = pd.concat(shards)
data = data.sort_index()
data.index = data['global_id']

# Read in the licensee data.
file_name = '../.datasets/Licensees_0.csv'
licensee_data = pd.read_csv(
    file_name,
    sep='\t',
    encoding='utf-16',
)

# Combine the data sets.
data = pd.merge(
    left=lab_data,
    right=licensee_data,
    how='left',
    left_on='for_mme_id',
    right_on='global_id'
)

#------------------------------------------------------------------------------
# Create supplementary statistics.
#------------------------------------------------------------------------------

# Add a time column.
data['date'] = pd.to_datetime(data['created_at'])

# Remove values with nonsensical THCA.
data = data[data['cannabinoid_d9_thca_percent'] <= 100]

# Calculate total cannabinoids.
cannabinoids_wa = [
    'cannabinoid_d9_thca_percent',
    'cannabinoid_d9_thc_percent',
    'cannabinoid_d8_thc_percent',
    'cannabinoid_thcv_percent',
    'cannabinoid_cbd_percent',
    'cannabinoid_cbda_percent',
    'cannabinoid_cbdv_percent',
    'cannabinoid_cbg_percent',
    'cannabinoid_cbga_percent',
    'cannabinoid_cbc_percent',
    'cannabinoid_cbn_percent',
]
data['total_cannabinoids'] = data[cannabinoids_wa].sum(axis=1)


#------------------------------------------------------------------------------
# Perform analysis.
#------------------------------------------------------------------------------

# Estimate the quantity sold based on quality control requirements and assuming
# that all products that pass quality control testing are sold.


# Plot the density of total cannabinoids in flower for the growers testing the
# most samples.


# Regress estimated monthly quantity sold against the monthly average total
# cannabinoids (in flower and in concentrates).


# Plot monthly estimated quantity sold against the average total cannabinoids.




