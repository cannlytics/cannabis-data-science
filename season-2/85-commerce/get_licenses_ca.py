"""
Cannabis Licenses | Get California Licenses
Copyright (c) 2022 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 9/16/2022
Updated: 9/21/2022
License: <https://github.com/cannlytics/cannlytics/blob/main/LICENSE>

Description:

    Collect California cannabis license data.

Data Source:

    - California Department of Cannabis Control Cannabis Unified License Search
    URL: <https://search.cannabis.ca.gov/>

"""
# Standard imports.
from datetime import datetime
from time import sleep

# External imports.
from cannlytics.utils import camel_to_snake
from cannlytics.utils.constants import DEFAULT_HEADERS
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns


# Specify where your data lives.
DATA_DIR = '../../.datasets/licenses'

# Define the license data API.
base = 'https://as-cdt-pub-vip-cannabis-ww-p-002.azurewebsites.net'
endpoint = '/licenses/filteredSearch'
query = f'{base}{endpoint}'
params = {'pageSize': 50, 'searchQuery': ''}

# Iterate over all of the pages to get all of the data.
iterate = True
page = 1
licenses = []
verbose = True
while(iterate):
    params['pageNumber'] = page
    response = requests.get(query, headers=DEFAULT_HEADERS, params=params)
    body = response.json()
    data = body['data']
    licenses.extend(data)
    if not body['metadata']['hasNext']:
        iterate = False
    if verbose:
        print('Recorded %i/%i pages.' % (page, body['metadata']['totalPages']))
    page += 1
    sleep(0.2)

# Standardize the licensee data.
license_data = pd.DataFrame(licenses)
columns = list(license_data.columns)
columns = [camel_to_snake(x) for x in columns]
license_data.columns = columns

# Save the data locally.
timestamp = datetime.now().isoformat()[:19].replace(':', '-')
license_data.to_excel(f'{DATA_DIR}/licenses-ca-{timestamp}.xlsx')

# # Optional: Archive the licensee data in Firebase.
# from cannlytics.firebase import initialize_firebase, update_documents
# n = 420
# database = initialize_firebase()
# collection = 'public/data/licenses'
# shards = [df[i:i + n] for i in range(0, df.shape[0], n)]
# for shard in shards:
#     refs = shard['license_number'].apply(lambda x: f'{collection}/{x}')
#     docs = shard.to_dict('records')
#     update_documents(refs, docs, database=database)

# Create a scatterplot of latitude and longitude with hue as license type.
sns.scatterplot(
    data=license_data.loc[
        (~license_data['premise_longitude'].isnull()) &
        (~license_data['premise_latitude'].isnull())
    ],
    x='premise_longitude',
    y='premise_latitude',
    hue='license_type',
)
plt.show()


# TODO: Create a data map.

# === Example ===
# https://jakevdp.github.io/PythonDataScienceHandbook/04.13-geographic-data-with-basemap.html


# === Example ===
# https://gawron.sdsu.edu/python_for_ss/course_core/book_draft/visualization/visualizing_geographic_data.html
