"""
Get Cannabis Data for Illinois | Cannlytics

Author: Keegan Skeate <keegan@cannlytics.com>
Created: 9/8//2021
Updated: 9/8/2021
License: MIT License <https://opensource.org/licenses/MIT>
"""
# External imports
import pandas as pd
from dotenv import dotenv_values
from fredapi import Fred


documents = {
    'cultivator': 'https://www2.illinois.gov/sites/agr/Plants/Documents/early%20approval%20adult%20use%20cultivation%20centers.pdf',
    'processor': '',
    'dispensary': 'https://www.idfpr.com/LicenseLookup/AdultUseDispensaries.pdf',
    'lab': 'https://www2.illinois.gov/sites/agr/Plants/Pages/Adult-Use-Cannabis.aspx',
}


#--------------------------------------------------------------------------
# Get the data (licensees and population).
#--------------------------------------------------------------------------

# Get the population data from Fred.
config = dotenv_values('../.env')
fred = Fred(api_key=config['FRED_API_KEY'])
population = fred.get_series('ILPOP', observation_start='1/1/2020')



