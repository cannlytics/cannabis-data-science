"""
Upload Cannabis Strains Data
Copyright (c) 2023 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 5/26/2023
Updated: 7/12/2023
License: <https://github.com/cannlytics/cannlytics/blob/main/LICENSE>

Command-line Usage:

    python data/archive/upload_strains.py all

"""
import os

import pandas as pd


# === Inventory analysis ===

variables = {
    'inventory_id': str,
    'product_name': str,
    'strain_name': str,
    'inventory_type': str,
    'inventory_updated_at': str,
    'licensee_name': str,
    'producer_licensee_id': str,
    'unit_weight_grams': float,
}

# Research question: What is the growers' favorite strain?
# That is, what strain has been grown by the largest
# number of cultivators?

# 
inventory_dir = 'D://data//washington//ccrs-stats//inventory'
os.listdir(inventory_dir)

# Read in data of interest.




# Merge lab results.
variables = {
    'inventory_id': str,
    'lab_licensee_id': str,
    'lab_result_id': str,
    'test_date': str,
    delta_9_thc
    thca
    total_thc
    cbd
    cbda
    total_cbd
    moisture_content
    water_activity
    status

}
lab_results_datafile = 'D://data//washington//ccrs-stats//lab_results//inventory_lab_results_0.xlsx'
lab_results = pd.read_excel(lab_results_datafile)
