"""
Shelf-Life Analysis of Products in Washington State
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 2/18/2022
Updated: 2/18/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script calculates various statistics about the time products
spend on the shelf (the time from testing to the time of sale) using
Washington State traceability data (2018-01-31 to 11-10-2021).

Data sources:

    - Random sample of sales items
    https://cannlytics.page.link/cds53

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd

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
# Read the data.
#--------------------------------------------------------------------------

# Specify where the data lives.
DATA_DIR = 'D:\\leaf-data'
DATA_FILE = f'{DATA_DIR}/samples/random-sales-items-2022-02-16.csv'

# Read in the data.
data = pd.read_csv(DATA_FILE)


#--------------------------------------------------------------------------
# TODO: Calculate shelf-life.
#--------------------------------------------------------------------------



#--------------------------------------------------------------------------
# TODO: Visualize the data.
#--------------------------------------------------------------------------


#--------------------------------------------------------------------------
# TODO: Calculate various statistics.
#--------------------------------------------------------------------------

