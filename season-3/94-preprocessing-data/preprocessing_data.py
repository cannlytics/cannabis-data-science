"""
CCRS Strain Analysis (Continued)
Copyright (c) 2022 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 12/28/2022
Updated: 12/28/2022
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Data Source:

    - WSLCB
    URL: <https://lcb.box.com/s/xseghpsq2t4i1musxj6mgd7b8rhxe7bm>

"""
# Standard imports:
import os

# External imports:
from cannlytics.data.ccrs import (
    CCRS_DATASETS,
    anonymize,
    get_datafiles,
    merge_datasets,
    unzip_datafiles,
)
from cannlytics.utils import rmerge, sorted_nicely
import pandas as pd


# Specify where your data lives.
STATS_DIR = 'D:\\data\\washington\\ccrs-stats\\'

# Augment inventory files with lab results.
inventory_files = []
inventory_dir = os.path.join(STATS_DIR, 'inventory')
inventory_files = sorted_nicely(os.listdir(inventory_dir))

# Find all ACDC inventory items.
acdc = pd.DataFrame()
for datafile in inventory_files:
    data = pd.read_excel(os.path.join(inventory_dir, datafile))
    data['InventoryId'] = data['InventoryId'].astype(str)
    match = data.loc[
        # (data['Name'].str.contains('ACDC', case=False)) &
        # (data['InventoryType'] == 'Usable Marijuana')
        (data['InventoryId'] == '8383745')
    ]
    acdc = pd.concat([acdc, match])
    print('Found', len(acdc), 'ACDC inventory items.')


# Find all Cannatonic inventory items.
cannatonic = pd.DataFrame()
for datafile in inventory_files:
    data = pd.read_excel(os.path.join(inventory_dir, datafile), nrows=10_000)
    match = data.loc[
        (data['Name'].str.contains('Cannatonic', case=False)) &
        (data['InventoryType'] == 'Usable Marijuana')
    ]
    cannatonic = pd.concat([cannatonic, match])
    print('Found', len(cannatonic), 'Cannatonic inventory items.')

# Augment inventory files with lab results.
lab_results_dir = os.path.join(STATS_DIR, 'lab_results')
lab_results = pd.read_excel(os.path.join(lab_results_dir, 'lab_results_0.xlsx'))
lab_results['InventoryId'] = lab_results['InventoryId'].astype(str)
acdc['InventoryId'] = acdc['InventoryId'].astype(str)
acdc_data = rmerge(
    acdc,
    lab_results,
    on='InventoryId',
    how='left',
    # validate='1:1',
)

cannatonic['InventoryId'] = cannatonic['InventoryId'].astype(str)
cannatonic_data = rmerge(
    cannatonic,
    lab_results,
    on='InventoryId',
    how='left',
    validate='m:1',
)

# TODO: Find the 100 most similar inventory items to ACDC.


#------------------------------------------------------------------------------
# Analysis: Find strains chemically similar to ACDC sold in the Seattle area.
#------------------------------------------------------------------------------

# TODO: Calculate the average THC to CBD ratio of ACDC strains.


# TODO: Identify lab results with similar THC to CBD ratios.


# TODO: Find inventory in Seattle that correspond to lab results similar to ACDC.

