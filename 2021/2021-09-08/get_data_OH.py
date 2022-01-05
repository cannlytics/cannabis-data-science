"""
Get Cannabis Data for Ohio | Cannlytics

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
    'cultivator': 'https://www.medicalmarijuana.ohio.gov/documents/ProgramUpdate/COO%20Cultivator%20Licensees.pdf',
    'processor': 'https://www.medicalmarijuana.ohio.gov/documents/ProgramUpdate/COO%20Processor%20Licensees.pdf',
    'dispensary': 'https://www.medicalmarijuana.ohio.gov/Documents/Dispensaries/List%20of%20Ohio%20Medical%20Marijuana%20Dispensaries%20with%20Certificates%20of%20Operation.pdf',
    'lab': 'https://www.medicalmarijuana.ohio.gov/documents/testing/COO%20Testing%20Lab%20Licensees.pdf',
}


#--------------------------------------------------------------------------
# Get the data (licensees and population).
#--------------------------------------------------------------------------

# Get the population data from Fred.
config = dotenv_values('../.env')
fred = Fred(api_key=config['FRED_API_KEY'])
population = fred.get_series('OHPOP', observation_start='1/1/2020')
