"""
Heavy Metal Analysis
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 10/3/2023
Updated: 10/3/2023
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    

Data Source:

   

"""
# Standard imports.
from datetime import datetime
import math
import os
from time import sleep

# External imports.
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.neighbors import NearestNeighbors


# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})



# Read the lab results.
results = pd.read_csv('data/md-lab-results-2023-09-27.csv')


print(results['product_subtype'].unique().tolist())

# Get the flower results.
flower = results[results['product_subtype'] == 'Raw Plant Material']


# === Strain name analysis ===


# StrainName



# === Lab Analysis ===

# TestingFacilityId


# === Cannabinoid Analysis ===


# CBD, CBDA, CBG, CBN, THC, THCA


# Total THC and total CBD.

# THC to CBD

# THC to CBG

# CBN


# === Terpene Analysis ===

# Identify outliers.
outliers = flower[flower['Terpenes'] > 6]
print(outliers)

# Filter outliers.
sample = flower[flower['Terpenes'] < 10]
print(sample['Terpenes'].describe())

# Plot the distribution of terpene content.
sns.histplot(sample['Terpenes'], bins=50, kde=True)
plt.xlabel('Terpene Content (%)')
plt.ylabel('Count')
plt.title('Distribution of Terpene Content in Raw Plant Material')
plt.show()
print('Mean:', round(sample['Terpenes'].mean(), 2))


# === Heavy metal analysis ===

# Summarize silver data.
silver_data = results[results['Silver'] > 0]
silver_fail = silver_data[silver_data['TestPassed'] == 'Fail'].shape[0]
silver_pass = silver_data[silver_data['TestPassed'] == 'Pass'].shape[0]
print(f"Number of Fail tests with Silver: {silver_fail}")
print(f"Number of Pass tests with Silver: {silver_pass}")

# Distribution of Silver.
filtered_silver = silver_data[(silver_data['Silver'] >= 0) & (silver_data['Silver'] <= 1)]
sns.histplot(filtered_silver['Silver'], bins=100)
plt.xlabel('Silver (ppm)')
plt.ylabel('Count')
plt.title('Distribution of Silver in Maryland Flower (0 to 1 ppm)')
plt.show()

# Define heavy metals.
metals = [
    "Arsenic",
    "Barium",
    "Cadmium",
    "Chromium",
    "Lead",
    "Mercury",
    "Selenium",
    "Silver",
]

# Analyze all metals.
for metal in metals:
    print(f'{metal} Analysis')
    
    # Association with Testing Outcomes
    # FIXME: Use the Maryland limit.
    metal_data = flower[flower[metal] > 0]
    metal_fail = metal_data[metal_data['TestPassed'] == 'Fail'].shape[0]
    metal_pass = metal_data[metal_data['TestPassed'] == 'Pass'].shape[0]
    
    # Calculate Failure Rate
    fail_rate = (metal_fail / len(flower)) * 100
    
    print(f"Number of Fail tests with {metal}: {metal_fail}")
    print(f"Number of Pass tests with {metal}: {metal_pass}")
    print(f"Failure rate for {metal}: {fail_rate:.2f}%")
    
    # Distribution of Metal (You can adjust the range if necessary)
    filtered_metal = metal_data[(metal_data[metal] >= 0) & (metal_data[metal] <= 1)]
    sns.histplot(filtered_metal[metal], bins=100)
    plt.xlabel(f'{metal} (ppm)')
    plt.ylabel('Count')
    plt.title(f'Distribution of {metal} in Maryland Flower (0 to 1 ppm)')
    plt.show()

    # Additional analyses or plots can be added here if necessary
