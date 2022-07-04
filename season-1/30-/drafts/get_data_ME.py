"""
Get Cannabis Data for Maine | Cannlytics

Author: Keegan Skeate <keegan@cannlytics.com>
Created: 9/15//2021
Updated: 9/15/2021
License: MIT License <https://opensource.org/licenses/MIT>
"""
# External imports
import pandas as pd
from dotenv import dotenv_values
from fredapi import Fred


state_data = {
    'licensees': 'https://www.maine.gov/dafs/omp/open-data/adult-use'
}


#--------------------------------------------------------------------------
# Get the data (licensees and population).
#--------------------------------------------------------------------------

# Get the population data from Fred.
config = dotenv_values('../.env')
fred = Fred(api_key=config['FRED_API_KEY'])
population = fred.get_series('MEPOP', observation_start='1/1/2020')

# TODO: Get state date.



