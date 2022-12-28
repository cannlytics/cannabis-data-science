"""
Lineage Tracing
Copyright (c) 2022 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 12/11/2022
Updated: 12/20/2022
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# Standard imports:

# External imports:
from cannlytics.data.ccrs import CCRS
from cannlytics.utils import unzip_files
import pandas as pd


# Specify where your data lives.
DATA_DIR = 'D:\\data\\washington\\ccrs-2022-11-22\\ccrs-2022-11-22'

# Initialize a CCRS client.
ccrs = CCRS(data_dir=DATA_DIR)

# === Data curation ===

# TODO: Read the licensee data.
licenses = ccrs.read_licensees()


# TODO: Read the lab result data.



# TODO: Read in the inventory data.



# TODO: Read in the sales data.


# === Lab result analysis ===

# TODO: Benford's Law analysis of digits.


# TODO: Distribution of THC / CBD by product type, by lab, by strain.


# TODO: Estimate revenue by lab.


# TODO: Estimate failure rates by analyte by lab.


# === Strain statistics ===

# TODO: Identify all of the crosses


# TODO: Plot strain popularity over time


# TODO: Estimate the canopy by licensee, by strain.


# === Sales statistics ===

# TODO: Estimate sales by licensee by day


# TODO: Estimate inventory by licensee by month (current).

