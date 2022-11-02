"""
State GDP and Adult-Use Cannabis Analysis
Copyright (c) 2022 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 9/22/2022
Updated: 10/5/2022
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# Standard imports.
import os

# External imports.
import pandas as pd


#------------------------------------------------------------------------------
# Get the data!
#------------------------------------------------------------------------------

# Specify where your data lives.
DATA_DIR = '../../.datasets/cannabis_licenses'
DATA_FILES = {
    'az': 'az/retailers-az-2022-10-05T10-55-24.xlsx',
    'ca': 'ca/licenses-ca-2022-09-21T19-02-29.xlsx',
    'co': 'co/retailers-co-2022-10-04T11-52-44.xlsx',
    'ct': 'ct/retailers-ct-2022-10-04T09-39-10.xlsx',
    'il': 'il/retailers-il-2022-10-03T07-43-27.xlsx',
    'ma': 'ma/retailers-ma-2022-10-02T17-23-18.xlsx',
    'me': 'me/licenses-me-2022-09-30T16-44-03.xlsx',
    'mi': 'mi/licenses-mi-2022-10-04T18-48-23.xlsx',
    'mt': 'mt/retailers-mt-2022-10-05T09-08-10.xlsx',
    'nj': 'nj/licenses-nj-2022-09-29T16-17-38.xlsx',
    'nm': 'nm/retailers-nm-2022-10-05T15-09-21.xlsx',
    'nv': 'nv/retailers-nv-2022-09-30T07-41-59.xlsx',
    'ri': 'ri/licenses-ri-2022-10-03T09-56-30.xlsx',
    'or': 'or/licenses-or-2022-09-28T10-11-12.xlsx',
    'vt': 'vt/retailers-vt-2022-10-03T11-07-16.xlsx',
    'wa': 'wa/licenses-wa-2022-09-29T14-44-25.xlsx',
}


def aggregate_retailers(
        datafiles,
        lat='premise_latitude',
        long='premise_longitude',
    ):
    """Aggregate retailer license data files,
    keeping only those with latitude and longitude."""

    # Read license data for each state.
    obs = []
    for data_file in datafiles:
        filename = os.path.join(DATA_DIR, data_file)
        licenses = pd.read_excel(filename, index_col=0)
        obs.append(licenses)

    # Return retailers with known latitude and longitude.
    data = pd.concat(obs)
    data = data.loc[(~data[lat].isnull()) & (~data[long].isnull())]
    return data


# Aggregate retailers.
retailers = aggregate_retailers(DATA_FILES.values())


#-----------------------------------------------------------------------
# Get supplementary state data.
#-----------------------------------------------------------------------

from bs4 import BeautifulSoup
from cannlytics.data.gis import get_state_population
from cannlytics.utils.constants import state_names
from dotenv import dotenv_values
from fredapi import Fred
import requests

# Read your FRED API key.
config = dotenv_values('../../.env')
fred_api_key = config['FRED_API_KEY']

# Get the population for each state (in 2021).
state_data = {}
for state, abbv in state_names.items():
    population = get_state_population(
        abbv,
        fred_api_key=fred_api_key,
        obs_start='2021-01-01',
    )
    state_data[state] = {'population': population['population']}

# Get the square miles of land for each state.
url = 'https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_area'
response = requests.get(url).text
soup = BeautifulSoup(response, 'lxml')
table = soup.find('table', class_='wikitable')
for items in table.find_all('tr'):
    data = items.find_all(['th', 'td'])
    if data:
        try:
            rank = int(data[1].text)
        except:
            continue
        state = data[0].text.replace('\n', '')
        land_area = float(data[5].text.replace('\n', '').replace(',', ''))
        state_data[state]['land_area_sq_mi']

# Get the change in GDP for each state in 2022 Q1.
code = 'NQGSP'
fred = Fred(api_key=fred_api_key)
for state, abbv in state_names.items():
    try:
        series = fred.get_series(abbv + code, '2021-10-01')
    except:
        continue
    current, past = series[-1], series[-2]
    change_gdp = ((current - past) / past) * 100
    state_data[state]['change_gdp_2022_q1'] = change_gdp


#-----------------------------------------------------------------------
# Analyze the data.
#-----------------------------------------------------------------------

import seaborn as sns
import statsmodels.api as sm

from cannlytics.utils.constants import states



# Count the number of retailers by state.
statistics = []
state_count = retailers.groupby('premise_state')['license_number'].count()
for state, count in state_count.iteritems():
    try:
        state_name = states[state]
        gdp = state_data[state_name]['change_gdp_2022_q1']
        population = state_data[state_name]['population']
        retailers_per_capita = count / population * 100_000
        statistics.append({
            'state': state,
            'retailers_per_capita': retailers_per_capita,
            'gdp': gdp,
        })
    except:
        continue

stats = pd.DataFrame(statistics)

# TODO: Calculate retailers per capita (100,000) by state.
retailers_per_capita = None


# TODO: Calculate retailers per 100 square miles by state.
retailers_per_100_miles = None


# TODO: Create `adult_use` dummy variable. Assign 0 `retailers_per_capita`.


# Visualize the relationship between retailers per capita and GDP.
ax = sns.regplot(
    data=stats,
    x='retailers_per_capita',
    y='gdp',
)


# # Regress GDP on adult-use status and retailers per capita.
# Y = stats['change_gdp_2022_q1']
# X = stats[['adult_use', 'retailers_per_capita']]
# X = sm.add_constant(X)
# regression = sm.OLS(Y, X).fit()
# print(regression.summary())

# # Interpret the relationships.
# beta = regression.params.adult_use
# statement = """If a state permitted adult-use at the start of 2022,
# then everything else held constant one would expect
# GDP in 2022 Q1 to change by {}.
# """.format(beta)
# print(statement)

# # Interpret the relationships.
# beta = regression.params.retailers_per_capita
# statement = """If retailers per 100,000 adults increases by 1,
# then everything else held constant one would expect
# GDP in 2022 Q1 to change by {}.
# """.format(beta)
# print(statement)
