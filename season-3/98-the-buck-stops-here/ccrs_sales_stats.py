"""
Curate CCRS Sales
Copyright (c) 2022-2023 Cannabis Data

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 1/1/2023
Updated: 2/1/2023
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Data Source:

    - Washington State Liquor and Cannabis Board (WSLCB)
    URL: <https://lcb.box.com/s/xseghpsq2t4i1musxj6mgd7b8rhxe7bm>

"""
# Standard imports:
import os

# External imports:
from cannlytics.data import create_hash
from cannlytics.data.ccrs import anonymize
import pandas as pd


#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

# Specify where your data lives.
base = 'D:\\data\\washington\\'
data_dir = f'{base}\\CCRS PRR (12-7-22)\\CCRS PRR (12-7-22)\\'
stats_dir = f'{base}\\ccrs-stats\\'

# Get the licensees.
licensee_stats_dir = os.path.join(stats_dir, 'licensee_stats')
licensees = os.listdir(licensee_stats_dir)


#------------------------------------------------------------------------------
# Data curation
#------------------------------------------------------------------------------

# Iterate over the licensees sales.
solid_edibles = pd.DataFrame()
liquid_edibles = pd.DataFrame()
for licensee_id in licensees:

    # Read the data.
    month = '2022-11'
    filename = f'{licensee_id}/sales-{licensee_id}-{month}.xlsx'
    datafile = os.path.join(licensee_stats_dir, filename)
    try:
        data = pd.read_excel(datafile)
        assert 'InventoryType' in data.columns
    except:
        continue

    # Anonymize the data.
    data = anonymize(data)

    # Create a hash of the data.
    # create_hash(data.to_json())

    # Keep track of data by sample type.
    solid_edibles_data = data.loc[data['InventoryType'] == 'Solid Edible']
    liquid_edibles_data = data.loc[data['InventoryType'] == 'Liquid Edible']
    solid_edibles = pd.concat([solid_edibles, solid_edibles_data])
    liquid_edibles = pd.concat([liquid_edibles, liquid_edibles_data])


#------------------------------------------------------------------------------
# Analysis
#------------------------------------------------------------------------------

# What is the most popular edible?
