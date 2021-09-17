"""
Licensees per Capita in Oklahoma | Cannabis Data Science Meetup

Author: Keegan Skeate <keegan@cannlytics.com>
Created: 9/1/2021
Updated: 9/8/2021
License: MIT License <https://opensource.org/licenses/MIT>
"""
# External imports
import pandas as pd
from dotenv import dotenv_values
from fredapi import Fred

# Internal imports
from metadata_ok import counties

#--------------------------------------------------------------------------
# Get the data (licensees and population).
#--------------------------------------------------------------------------

# Get licensees in Oklahoma (See Cannlytics data collection tools).
# https://github.com/cannlytics/cannlytics/blob/main/tools/data/state_data/get_data_OK.py
licensees = pd.read_excel('data/licensees_OK_2021-09-01.xlsx')

# Get the population data from Fred.
config = dotenv_values('../.env')
fred = Fred(api_key=config['FRED_API_KEY'])
population = fred.get_series('OKPOP', observation_start='1/1/2020')

#--------------------------------------------------------------------------
# Calculate statistics.
#--------------------------------------------------------------------------

# Calculate licensees per capita in Oklahoma.
licensees_per_capita = len(licensees) / (population.iloc[0] * 1000)
print('Licensees per capita:', licensees_per_capita)

# Calculate licensees per capita for each county in Oklahoma.
licensees_per_capita = {}
for county in counties:

    # Find all licensees in that county.
    county_name = county['name']
    county_licensees = licensees.loc[licensees.county == county_name]

    # Get the population of that county.
    county_population = fred.get_series(county['pop_ref'], observation_start='1/1/2020')

    # Calculate the licensees per capita in that county.
    county_licensees_per_capita = len(county_licensees) / (county_population.iloc[0] * 1000)

    # Keep track of the data.
    # key = 'OK_' + county_name.replace(' ', '_')
    # licensees_per_capita[county_name] = {'licensees_per_capita': county_licensees_per_capita}
    licensees_per_capita[county_name] = county_licensees_per_capita
