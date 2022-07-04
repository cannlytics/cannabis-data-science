"""
Wholesale Analysis in Washington State
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 2/17/2022
Updated: 2/17/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script calculates various statistics about strains from the
Washington State traceability data (2018-01-31 to 11-10-2021).

Data sources:

    - Random sample of sales items
    https://cannlytics.page.link/cds53

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd?page=1
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd?page=2

Data Guide:

    - Washington State Leaf Data Systems Guide
    https://lcb.wa.gov/sites/default/files/publications/Marijuana/traceability/WALeafDataSystems_UserManual_v1.37.5_AddendumC_LicenseeUser.pdf
"""

# External imports.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # pip install basemap
import pandas as pd
import seaborn as sns


# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
})


#--------------------------------------------------------------------------
# Analyze the data.
#--------------------------------------------------------------------------

# Specify where the data lives.
DATA_DIR = 'D:\\leaf-data'
DATA_FILE = f'{DATA_DIR}/samples/random-sales-items-2022-02-16.csv'

# Read in the data.
data = pd.read_csv(DATA_FILE)


#--------------------------------------------------------------------------
# Analyze wholesale transactions in Washington State.
#--------------------------------------------------------------------------

# Measure the number of vendors that each retailer has.


# Measure the number of retailers to which each producer distributes.


# Identify the retailer with the highest number of vendors.


# Identify the producer with the highest number of retailer connections.


# Count the number of transfers from producers to retailers (by day/month/year?).


# Measure (or estimate) the distance producers are travelling to retailers.

