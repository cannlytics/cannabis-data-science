"""
Get Cannabis Data for New Jersey | Cannlytics

Author: Keegan Skeate <keegan@cannlytics.com>
Created: 9/14//2021
Updated: 9/14/2021
License: MIT License <https://opensource.org/licenses/MIT>
"""
# External imports
import pandas as pd
from dotenv import dotenv_values
from fredapi import Fred


state_data = {
    'dispensary': 'https://www.nj.gov/cannabis/businesses/currently-licensed/',
    'regulations': 'https://www.nj.gov/cannabis/resources/cannabis-laws/',
    'fees': 'https://www.nj.gov/cannabis/businesses/personal-use/'
}


#--------------------------------------------------------------------------
# Get the data (licensees and population).
#--------------------------------------------------------------------------

# Get the population data from Fred.
config = dotenv_values('../.env')
fred = Fred(api_key=config['FRED_API_KEY'])
population = fred.get_series('NJPOP', observation_start='1/1/2020')

# Get state date.



