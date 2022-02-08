"""
Calculate Transporation Statistics
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 1/29/2022
Updated: 2/7/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script calculates various transportation statistics from the
Washington State traceability data (2018-01-31 to 11-10-2021).

Data sources:

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd?page=1
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd?page=2

Data Guide:

    - Washington State Leaf Data Systems Guide
    https://lcb.wa.gov/sites/default/files/publications/Marijuana/traceability/WALeafDataSystems_UserManual_v1.37.5_AddendumC_LicenseeUser.pdf
"""
# Standard imports.
import gc
from time import sleep

# External imports.
from dotenv import dotenv_values
import googlemaps
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # pip install basemap
import pandas as pd
import seaborn as sns

# Internal imports
from augmentation import augment_dataset
# from utils import format_thousands


# Specify where your data lives.
DATA_DIR = 'D:\\leaf-data'


#------------------------------------------------------------------------------
# Read transfer data.
#------------------------------------------------------------------------------

# Define useful metadata.
transfer_datatypes = {
    # 'global_id': 'string',
    # 'user_id': 'string',
    # 'external_id': 'string',
    # 'number_of_edits': 'float',
    'void': 'bool',
    'from_mme_id': 'string',
    'to_mme_id': 'string',
    # 'transporting_mme_id': 'string',
    # 'from_user_id': 'string',
    # 'to_user_id': 'string',
    # 'status': 'string',
    # 'transfer_type': 'string',
    # 'sale_id': 'string',
}
transfer_date_fields = [
    'created_at',
    # 'updated_at',
    # 'hold_starts_at',
    # 'hold_ends_at',
    # 'deleted_at',
    # 'transferred_at',
]
transfer_columns = list(transfer_datatypes.keys()) + transfer_date_fields

# Example: Read in a portion of the data.

transfers = pd.read_csv(
    f'{DATA_DIR}/InventoryTransfers_0.csv',
    sep='\t',
    encoding='utf-16',
    usecols=transfer_columns,
    dtype=transfer_datatypes,
    parse_dates=transfer_date_fields,
)

#------------------------------------------------------------------------------
# Augment the data.
#------------------------------------------------------------------------------

# Specify stats to collect.
daily_transfers = {}
daily_transfers_by_license = []

# Specify the time range to calculate statistics.
time_range = pd.date_range(start='2018-03-31', end='2018-05-30')
transfers['created_at_day'] = transfers['created_at'].dt.date
for date in time_range:

    # Get the day's data.
    day = date.date()
    day_data = transfers.loc[
        (transfers['created_at_day'] == date) &
        (~transfers['void'])
    ]
    total_transfers = len(day_data)
    daily_transfers[day] = total_transfers

    # Count the number of transfers for each licensee
    transfers_by_license = []
    senders = list(day_data['from_mme_id'].unique())
    for sender in senders:
        # license_transfers = {}
        sender_data = day_data.loc[day_data['from_mme_id'] == sender]
        recipient_count = sender_data['to_mme_id'].value_counts()
        try:
            for index, value in recipient_count.iterrows():
                daily_transfers_by_license.append({
                    'day': day,
                    'sender': sender,
                    'recipient': index,
                    'count': value,
                })
        except AttributeError:
            index = recipient_count.index[0]
            value = recipient_count.values[0]
            daily_transfers_by_license.append({
                'day': day,
                'sender': sender,
                'recipient': index,
                'count': value,
            })
        except IndexError:
            pass

    print('Counted %i transfers for %s' % (total_transfers, day))


#------------------------------------------------------------------------------
# Get recipient and transporter licensee details (latitude and longitude).
#------------------------------------------------------------------------------

# Create panel data.
panel = pd.DataFrame(daily_transfers_by_license)

# Read licensees data.
licensee_fields = {
    'global_id': 'string',
    'latitude': 'float',
    'longitude': 'float',
    'name': 'string',
    'type': 'string',
}
licensees = pd.read_csv(
    'D:/leaf-data/augmented/augmented-washington-state-licensees.csv',
    usecols=list(licensee_fields.keys()),
    dtype=licensee_fields,
)

# Get the sender's data.
panel = pd.merge(
    left=panel,
    right=licensees,
    how='left',
    left_on='sender',
    right_on='global_id',
)
panel.rename(
    columns={
        'latitude': 'sender_latitude',
        'longitude': 'sender_longitude',
        'name': 'sender_name',
        'type': 'sender_type',
    },
    inplace=True,
)
panel.drop(['global_id'], axis=1, inplace=True, errors='ignore')

# Get the recipient's data.
panel = pd.merge(
    left=panel,
    right=licensees,
    how='left',
    left_on='recipient',
    right_on='global_id',
)
panel.rename(
    columns={
        'latitude': 'recipient_latitude',
        'longitude': 'recipient_longitude',
        'name': 'recipient_name',
        'type': 'recipient_type',
    },
    inplace=True,
)
panel.drop(['global_id'], axis=1, inplace=True, errors='ignore')

# Initialize Google Maps.
config = dotenv_values('../.env')
google_maps_api_key = config['GOOGLE_MAPS_API_KEY']
gmaps = googlemaps.Client(key=google_maps_api_key)  

# Get all of the unique from / to combinations and calculate
# the route (distance and time) for each transfer combination.
combinations = panel.drop_duplicates(
    subset=['sender', 'recipient'],
    keep='first'
).reset_index(drop=True)
distances = []
durations = []
for index, row in combinations.iterrows():

    # TODO: Get the direction for a given from / to combination.
    # distance = gmaps.distance_matrix('Delhi','Mumbai')['rows'][0]['elements'][0]
    directions = gmaps.distance_matrix(
        [str(row['sender_latitude']) + ' ' + str(row['sender_longitude'])],
        [str(row['recipient_latitude']) + ' ' + str(row['recipient_longitude'])],
        mode='driving')
    
    elements = directions['rows'][0]['elements'][0]

    km = elements['distance']['value']
    duration = elements['duration']['value']
    distances.append(km)
    durations.append(duration)

    # TODO: Save route for plotting

combinations['distance'] = pd.Series(distances)
combinations['duration'] = pd.Series(durations)


# Combine distances with panel data (optional: optimize this code).
def combine_distances(row, combinations):
    match = combinations.loc[
        (combinations['sender'] == row['sender']) &
        (combinations['recipient'] == row['recipient'])
    ].iloc[0]
    return match['distance'], match['duration']

# Merge distance and duration for each observation.
distances = panel.apply(lambda row : combine_distances(row, combinations), axis=1)
# distances, durations = pd.Series(
#     combine_distances(row, combinations)
#     for index, row in panel.iterrows()
# )
panel['distance'] = distances.apply(lambda x: [y[0] for y in x])
panel['duration'] = distances.apply(lambda x: [y[1] for y in x])

# Optional: Optimize with map
# distances, durations =  map(combine_distances, panel, combinations)

# Optional: Optimize matching with vectorization.
# def match_combinations(df, combinations):
#     return (
#         combinations.loc[
#             (combinations['sender'] == df['sender']) &
#             (combinations['recipient'] == df['recipient']) 
#         ].iloc[0][['distance', 'duration']]
#     )
# distances = match_combinations(panel, combinations)


# TODO: Save the panel data.
# panel.to_csv('D:/leaf-data/augmented/transfer-statistics.csv')

#------------------------------------------------------------------------------
# Calculate statistics.
#------------------------------------------------------------------------------

# Plot daily, monthly, transfers.
timeseries = pd.DataFrame.from_dict(daily_transfers, orient='index')
timeseries.index = pd.to_datetime(timeseries.index)
timeseries.rename(columns={0: 'total_transfers'}, inplace=True)
timeseries.resample('M')['total_transfers'].sum().plot()
plt.show()

# How many total transfers, how many transfers from license A to B?



# Calculate the total number of miles driven (time spent driving).

# TODO: Save the daily statistics.
# timeseries.to_csv('D:/leaf-data/augmented/transfer-timeseries.csv')

#------------------------------------------------------------------------------
# Visualize the data.
#------------------------------------------------------------------------------

# Plot of all routes driven.
