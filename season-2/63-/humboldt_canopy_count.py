"""
Humboldt County Canopy Count
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 4/25/2022
Updated: 5/4/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script analyzes observed cannabis plants in California,
principally Humboldt County and the "Emerald Triangle" surrounding areas.
The climates of these areas will be studied.

Data Sources:

    - Google Earth
    earth.google.com/web/

    - Fed FRED
    https://fred.stlouisfed.org/

    - PRISM USDA Plant Hardiness Zone GIS Datasets
    https://prism.oregonstate.edu/projects/plant_hardiness_zones.php

    - TODO: Census
    https://www.census.gov/data/developers/data-sets.html
    E.g. https://www.census.gov/quickfacts/humboldtcountycalifornia

References:

    - Characteristics of farms applying for cannabis cultivation permits
    https://calag.ucanr.edu/archive/?article=ca.2019a0019

"""
# External imports.
from dotenv import dotenv_values
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Internal imports.
from data import get_state_population


# Specify where your data lives.
DATA_DIR = '../.datasets'

# Setup plotting style.
colors = sns.color_palette('Set2', n_colors=10)
primary_colors = sns.color_palette('husl', n_colors=10)
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
})



#-----------------------------------------------------------------------
# Read the data.
#-----------------------------------------------------------------------

# Read in the canopy data.
datafile = '../.datasets/california/humboldt_canopy_count.xlsx'
canopy = pd.read_excel(datafile, sheet_name='Data')

# Read in altitude data.


# TODO: Find zipcode for each latitude, longitude

# Read in plant hardiness data.
datafile = '../.datasets/public/phm_us_zipcode.csv'
phz = pd.read_csv(datafile, index_col=0)


#-----------------------------------------------------------------------
# Augment the data.
#-----------------------------------------------------------------------

counties = {
    'Humboldt': {'population_code': 'CAHUMB0'},
    'Trinity': {'population_code': 'CATRIN5POP'},
    'Shasta': {'population_code': 'CASHAS5POP'},
    # 'Mendocino': {'population_code': 'CAHUMB0'},
    # 'Tehama': {'population_code': 'CAHUMB0'},
    # 'Siskiyou': {'population_code': 'CAHUMB0'},
    # 'Del Norte': {'population_code': 'CAHUMB0'},
}

# Calculate population density of locations.
config = dotenv_values('../.env')
fred_api_key = config['FRED_API_KEY']
population = get_state_population(
    api_key=fred_api_key,
    state='CAHUMB0',
    obs_start='2021-01-01',
    obs_end='2021-01-01',
)


# TODO: Look at distribution of elevations (altitudes)!


# TODO: Calculate plants per capita.


# TODO: Calculate distance to nearest city.


# TODO: Get crime rates.


# TODO: Get property values.


# TODO: Visualize the data on a map (bubble map).



#-----------------------------------------------------------------------
# Analyze the data.
#-----------------------------------------------------------------------

# Look at typical space for outdoor plants.
sq_m_per_plant = canopy['area_outdoor'] / canopy['plants']

# Look at typical space for greenhouses



#-----------------------------------------------------------------------
# # Visualize the data.
#-----------------------------------------------------------------------

# Get cultivators and plant count from MA Open Data.


# Get cultivators in Oklahoma.


# Get cultivators and plant count in Washington.


# Plant hardiness of cultivations in WA, MA, OK, etc. in comparison.
