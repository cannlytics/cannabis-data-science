"""
Interstate Cannabis Commerce
Copyright (c) 2022 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 9/22/2022
Updated: 9/27/2022
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description: 

    Map the adult-use cannabis retailers permitted in the United States.

Data Sources (16):

    - Alaska
    URL: <>

    - Arizona Department of Health Services | Division of Licensing
    URL: <https://azcarecheck.azdhs.gov/s/?licenseType=null>

    - Colorado
    URL: <>

    - Connecticut
    URL: <>

    - Illinois
    URL: <>

    - Maine
    URL: <>

    - Massachusetts
    URL: <>

    - Michigan
    URL: <>

    - Montana Department of Revenue | Cannabis Control Division
    URL: <https://mtrevenue.gov/cannabis/#CannabisLicenses>

    - New Mexico
    URL: <https://nmrldlpi.force.com/bcd/s/public-search-license?division=CCD&language=en_US>
    
    - Nevada Cannabis Compliance Board | Nevada Cannabis Licensees
    URL: <https://ccb.nv.gov/list-of-licensees/>

    - New Jersey
    URL: <>

    - Oregon Liquor and Cannabis Commission
    URL: <https://www.oregon.gov/olcc/marijuana/pages/recreational-marijuana-licensing.aspx>

    - Rhode Island
    URL: <>

    - Vermont
    URL: <>

    - Washington
    URL: <https://lcb.wa.gov/records/frequently-requested-lists>

Coming Soon (3):
    
    - New York
    - Virginia
    - D.C.

Medical (17):

    - Utah
    - Oklahoma
    - North Dakota
    - South Dakota
    - Minnesota
    - Missouri
    - Arkansas
    - Louisiana
    - Mississippi
    - Alabama
    - Florida
    - Ohio
    - West Virginia
    - Pennsylvania
    - Maryland
    - Delaware
    - New Hampshire

"""
# Standard imports.

# External imports.
import folium
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Specify where your data lives.
DATA_DIR = r'C:\Users\keega\Documents\cannlytics\datasets\cannabis_licenses\data'
# DATA_DIR = '../../.datasets/licenses'

#-----------------------------------------------------------------------
# Get the data.
#-----------------------------------------------------------------------

# California retailers.
filename = f'{DATA_DIR}/ca/licenses-ca-2022-09-21T19-02-29.xlsx'
ca_licenses = pd.read_excel(filename, index_col=0)

# Alaska retailers.

# Arizona retailers.

# Colorado retailers.

# Connecticut retailers.

# Illinois retailers.

# Maine retailers.

# Massachusetts retailers.

# Michigan retailers.

# Montana retailers.

# New Mexico retailers.

# Nevada retailers.

# New Jersey retailers.

# Oregon retailers.
filename = f'{DATA_DIR}/or/licenses-or-2022-09-28T10-11-12.xlsx'
or_licenses = pd.read_excel(filename, index_col=0)

# Rhode Island retailers.

# Vermont retailers.

# Washington retailers.


#-----------------------------------------------------------------------
# Look at the data!
#-----------------------------------------------------------------------

# Aggregate all of the retailer data.
retailers = pd.concat([
    ca_licenses,
    or_licenses,
])
retailers = retailers.loc[
    (~retailers['premise_longitude'].isnull()) &
    (~retailers['premise_latitude'].isnull())
]

# Create a scatterplot of latitude and longitude with hue as license type.
sns.scatterplot(
    data=retailers,
    x='premise_longitude',
    y='premise_latitude',
    hue='license_type',
)
plt.show()

# Optional: Create a nice static map.

# Create an interactive map.
locations = retailers[['premise_latitude', 'premise_longitude']].to_numpy()
m = folium.Map(
    location=[45.5236, -122.6750],
    zoom_start=4,
    control_scale=True,
)
for index, row in retailers.iterrows():
    folium.Circle(
        radius=10,
        location=[row['premise_latitude'], row['premise_longitude']],
        color='crimson',
    ).add_to(m)
m.save('map.html')


#-----------------------------------------------------------------------
# Get supplementary data.
#-----------------------------------------------------------------------

from bs4 import BeautifulSoup
from cannlytics.data.gis import get_state_population
from cannlytics.utils.constants import state_names
from dotenv import dotenv_values
from fredapi import Fred
import requests

# Read your FRED API key.
config = dotenv_values('../.env')
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
    except ValueError:
        continue
    current, past = series[-1], series[-2]
    change_gdp = ((current - past) / past) * 100
    state_data[state]['change_gdp_2022_q1'] = change_gdp


#-----------------------------------------------------------------------
# Analyze the data.
#-----------------------------------------------------------------------

import statsmodels.api as sm

# FIXME: Compile all of the state statistics.
stats = pd.DataFrame()

# TODO: Count the number of retailers by state.


# TODO: Calculate retailers per capita (100,000) by state.


# TODO: Calculate retailers per 100 square miles by state.


# TODO: Create `adult_use` dummy variable. Assign 0 `retailers_per_capita`.


# Regress GDP on adult-use status and retailers per capita.
Y = stats['change_gdp_2022_q1']
X = stats[['adult_use', 'retailers_per_capita']]
X = sm.add_constant(X)
regression = sm.OLS(Y, X).fit()
print(regression.summary())

# Interpret the relationships.
beta = regression.params.adult_use
statement = """If a state permitted adult-use at the start of 2022,
then everything else held constant one would expect
GDP in 2022 Q1 to change by {}.
""".format(beta)
print(statement)

# Interpret the relationships.
beta = regression.params.retailers_per_capita
statement = """If retailers per 100,000 adults increases by 1,
then everything else held constant one would expect
GDP in 2022 Q1 to change by {}.
""".format(beta)
print(statement)
