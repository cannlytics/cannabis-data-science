"""
Environment Analysis
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 9/6/2023
Updated: 9/13/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# Standard imports:
from datetime import datetime, timedelta

# External imports:
from dotenv import dotenv_values
import pandas as pd
import statsmodels.api as sm
import requests

# Get the air quality API key.
config = dotenv_values('../../.env')
api_key = config['AIR_QUALITY_API_KEY']

# Format URL.
url = f'https://airquality.googleapis.com/v1/history:lookup?key={api_key}'

# Augment California environment data.
address = '5601 Casitas Pass Rd, Carpinteria, CA 93013'
latitude = 34.404930
longitude = -119.503227
date = '2023-08-21T20:00:00Z'

# Request air quality data.
headers = {'Content-Type': 'application/json'}
data = {
    'dateTime': date,
    'location': {
        'latitude': latitude,
        'longitude': longitude
    }
}
response = requests.post(url, json=data, headers=headers)
aqi = response.json()['hoursInfo'][0]['indexes'][0]['aqi']


#------------------------------------------------------------------------------
# California air quality analysis.
#------------------------------------------------------------------------------

# Read California lab results.
ca_results = pd.read_excel('./data/ca-lab-results-2023-08-30.xlsx', sheet_name='Values')
ca_results['state'] = 'ca'

# Create a sample.
ca_results['date'] = ca_results['date_collected'].apply(pd.to_datetime)
sample = ca_results.loc[ca_results['date'] >= pd.to_datetime('2023-08-01')].copy()

# Augment with environment data.
air_quality = []
headers = {'Content-Type': 'application/json'}
latitude, longitude = 34.404930, -119.503227
for i, row in sample.iterrows():
    dt = datetime.fromisoformat(row['date_collected'])
    standard_dt = dt + timedelta(hours=20)
    date_string = standard_dt.isoformat() + 'Z'
    data = {
        'dateTime': date_string,
        'location': {
            'latitude': latitude,
            'longitude': longitude
        }
    }
    response = requests.post(url, json=data, headers=headers)
    aqi = response.json()['hoursInfo'][0]['indexes'][0]['aqi']
    air_quality.append(aqi)
    print('AQI on {}: {}'.format(date_string, aqi))

# Add air quality to the sample.
sample['air_quality'] = air_quality

# === Regress terpene and cannabinoid concentrations on air quality. ===

# Regress terpene concentrations on air quality.
X = sample['air_quality']
X = sm.add_constant(X)
y_terpene = sample['beta_caryophyllene']
model_terpene = sm.OLS(y_terpene, X)
results_terpene = model_terpene.fit()
print(results_terpene.summary())

# Regress cannabinoid concentrations on air quality.
y_cannabinoid = sample['thcva']
model_cannabinoid = sm.OLS(y_cannabinoid, X)
results_cannabinoid = model_cannabinoid.fit()
print(results_cannabinoid.summary())


#------------------------------------------------------------------------------
# Florida air quality analysis.
#------------------------------------------------------------------------------

# TODO: Read Florida lab results.


# TODO: Augment with Florida environment data.


# TODO: Regress terpene and cannabinoid concentrations
# against environment data.


# TODO: Perform MANOVA on terpene and cannabinoid concentrations
# in different environments.



#------------------------------------------------------------------------------
# Washington air quality analysis.
#------------------------------------------------------------------------------

# Read Washington lab results.
wa_results = pd.read_excel('./data/wa-lab-results-2023-08-30.xlsx')
wa_results['producer_licensee_id'] = wa_results['producer_licensee_id'].apply(lambda x: str(x).replace('.0', ''))
wa_results['licensee_dba'] = wa_results['licensee_dba'].apply(lambda x: str(x).title())

# FIXME: Restrict to 2023-08-01 and later.
sample = wa_results.loc[wa_results['created_date'] >= pd.to_datetime('2023-08-01')].copy()

# FIXME: Augment producer latitude and longitude.
licenses = pd.read_csv(
    './data/Licensee_0.csv',
    sep='\t',
    encoding='utf-16',
    engine='python',
)
wa_results = wa_results.merge(
    licenses,
    left_on='licensee_name',
    right_on='Name',
    how='left',
)

# TODO: Geocode licenses.

# Augment with Washington environment data.
# With `created_date`, `premise_latitude`, `premise_longitude`.
air_quality = []
sample = wa_results.sample(100)
sample[['premise_latitude', 'premise_longitude']]
for i, row in sample.iterrows():
    dt = datetime.fromisoformat(row['created_date'])
    new_dt = dt + timedelta(hours=20)
    new_date_str = new_dt.isoformat() + 'Z'
    data = {
        'dateTime': new_date_str,
        'location': {
            'latitude': row['premise_latitude'],
            'longitude': row['premise_longitude']
        }
    }
    response = requests.post(url, json=data, headers=headers)
    aqi = response.json()['hoursInfo'][0]['indexes'][0]['aqi']
    air_quality.append(aqi)

# Add air quality to the sample.
sample['air_quality'] = air_quality

# TODO: Estimate probit models of failure rates against environment data.
