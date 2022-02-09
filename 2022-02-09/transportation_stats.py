"""
Calculate Transporation Statistics
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 1/29/2022
Updated: 2/8/2022
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
from datetime import datetime
import gc
from statistics import mode
from time import sleep
from typing import Any, Optional, Tuple

# External imports.
from dotenv import dotenv_values
import googlemaps
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.basemap import Basemap # pip install basemap
import pandas as pd
import polyline # pip install polyline
import seaborn as sns

# Internal imports
from augmentation import augment_dataset
from utils import format_thousands


# Specify where your data lives.
DATA_DIR = 'D:\\leaf-data'

# Define the plot style.
palette = sns.color_palette('Set2', n_colors=10)
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
})


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

# transfers = pd.read_csv(
#     f'{DATA_DIR}/InventoryTransfers_0.csv',
#     sep='\t',
#     encoding='utf-16',
#     usecols=transfer_columns,
#     dtype=transfer_datatypes,
#     parse_dates=transfer_date_fields,
# )

#------------------------------------------------------------------------------
# Augment the data.
#------------------------------------------------------------------------------

# Specify stats to collect.
# daily_transfers = {}
# daily_transfers_by_license = []

# # Specify the time range to calculate statistics.
# time_range = pd.date_range(start='2018-01-31', end='2021-11-10')
# transfers['created_at_day'] = transfers['created_at'].dt.date
# for date in time_range:

#     # Get the day's data.
#     day = date.date()
#     day_data = transfers.loc[
#         (transfers['created_at_day'] == date) &
#         (~transfers['void'])
#     ]
#     total_transfers = len(day_data)
#     daily_transfers[day] = total_transfers

#     # Count the number of transfers for each licensee
#     transfers_by_license = []
#     senders = list(day_data['from_mme_id'].unique())
#     for sender in senders:
#         # license_transfers = {}
#         sender_data = day_data.loc[day_data['from_mme_id'] == sender]
#         recipient_count = sender_data['to_mme_id'].value_counts()
#         try:
#             try:
#                 for index, value in recipient_count.iterrows():
#                     daily_transfers_by_license.append({
#                         'day': day,
#                         'sender': sender,
#                         'recipient': index,
#                         'count': value,
#                     })
#             except AttributeError:
#                 index = recipient_count.index[0]
#                 value = recipient_count.values[0]
#                 daily_transfers_by_license.append({
#                     'day': day,
#                     'sender': sender,
#                     'recipient': index,
#                     'count': value,
#                 })
#         except IndexError:
#             pass

#     print('Counted %i transfers for %s' % (total_transfers, day))


#------------------------------------------------------------------------------
# Get recipient and transporter licensee details (latitude and longitude).
# Map data ©2022 Google
#------------------------------------------------------------------------------

# # Create panel data.
# panel = pd.DataFrame(daily_transfers_by_license)

# # Read licensees data.
# licensee_fields = {
#     'global_id': 'string',
#     'latitude': 'float',
#     'longitude': 'float',
#     'name': 'string',
#     'type': 'string',
# }
# licensees = pd.read_csv(
#     'D:/leaf-data/augmented/augmented-washington-state-licensees.csv',
#     usecols=list(licensee_fields.keys()),
#     dtype=licensee_fields,
# )

# # Get the sender's data.
# panel = pd.merge(
#     left=panel,
#     right=licensees,
#     how='left',
#     left_on='sender',
#     right_on='global_id',
# )
# panel.rename(
#     columns={
#         'latitude': 'sender_latitude',
#         'longitude': 'sender_longitude',
#         'name': 'sender_name',
#         'type': 'sender_type',
#     },
#     inplace=True,
# )
# panel.drop(['global_id'], axis=1, inplace=True, errors='ignore')

# # Get the recipient's data.
# panel = pd.merge(
#     left=panel,
#     right=licensees,
#     how='left',
#     left_on='recipient',
#     right_on='global_id',
# )
# panel.rename(
#     columns={
#         'latitude': 'recipient_latitude',
#         'longitude': 'recipient_longitude',
#         'name': 'recipient_name',
#         'type': 'recipient_type',
#     },
#     inplace=True,
# )
# panel.drop(['global_id'], axis=1, inplace=True, errors='ignore')

# # Initialize Google Maps.
# config = dotenv_values('../.env')
# google_maps_api_key = config['GOOGLE_MAPS_API_KEY']
# gmaps = googlemaps.Client(key=google_maps_api_key)  

# # Get all of the unique from / to combinations and calculate
# # the route (distance and time) for each transfer combination.
# combinations = panel.drop_duplicates(
#     subset=['sender', 'recipient'],
#     keep='first'
# ).reset_index(drop=True)
# # now = datetime.now()
# # distances = []
# # durations = []
# # routes = []
# # for index, row in combinations.iterrows():

# #     # Simple: Get distance as the bird flies.


# #     # Get the distance for a given route.
# #     # driving_distances = gmaps.distance_matrix(
# #     #     [str(row['sender_latitude']) + ' ' + str(row['sender_longitude'])],
# #     #     [str(row['recipient_latitude']) + ' ' + str(row['recipient_longitude'])],
# #     #     mode='driving')
# #     # elements = driving_distances['rows'][0]['elements'][0]
# #     # km = elements['distance']['value']
# #     # duration = elements['duration']['value']
# #     # distances.append(km)
# #     # durations.append(duration)

# #     # Get driving directions.
# #     # driving_directions = gmaps.directions(
# #     #     str(row['sender_latitude']) + ',' + str(row['sender_longitude']),
# #     #     str(row['recipient_latitude']) + ',' + str(row['recipient_longitude']),
# #     #     mode='driving',
# #     #     departure_time=now
# #     # )
# #     # routes.append(driving_directions[0]['overview_polyline']['points'])


# # def combine_distances(row, combinations):
# #     """Combine distances with panel data (optional: optimize this code)."""
# #     match = combinations.loc[
# #         (combinations['sender'] == row['sender']) &
# #         (combinations['recipient'] == row['recipient'])
# #     ].iloc[0]
# #     return match['distance'], match['duration']


# # # Merge distance and duration for each observation.
# # combinations['distance'] = pd.Series(distances)
# # combinations['duration'] = pd.Series(durations)
# # distances = panel.apply(lambda row : combine_distances(row, combinations), axis=1)
# # panel['distance'] = distances.apply(lambda x: [y[0] for y in x])
# # panel['duration'] = distances.apply(lambda x: [y[1] for y in x])

# # Save the data.
# panel.to_csv('D:/leaf-data/augmented/transfer-statistics-b.csv', index=False)
# timeseries = pd.DataFrame.from_dict(daily_transfers, orient='index')
# timeseries.index = pd.to_datetime(timeseries.index)
# timeseries.rename(columns={0: 'total_transfers'}, inplace=True)
# timeseries.to_csv('D:/leaf-data/augmented/transfer-timeseries.csv', index=False)


#------------------------------------------------------------------------------
# Calculate statistics.
#------------------------------------------------------------------------------

# Read the data.
panel = pd.read_csv('D:/leaf-data/augmented/transfer-statistics.csv')
timeseries = pd.read_csv(
    'D:/leaf-data/augmented/transfer-timeseries.csv',
    index_col=0,
)
timeseries.index = pd.to_datetime(timeseries.index)
timeseries = timeseries.loc[
    (timeseries.index >= pd.to_datetime('2018-04-01')) &
    (timeseries.index <= pd.to_datetime('2021-10-31'))
]

# Plot daily, monthly, transfers.
ax = timeseries.resample('M')['total_transfers'].sum().plot()
ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
plt.show()

# TODO: How many total transfers, how many transfers from license A to B?


# TODO: Bar chart of different license type to license type combinations over time.


# TODO: Calculate the total number of miles driven (time spent driving).


#------------------------------------------------------------------------------
# Visualize the data.
#------------------------------------------------------------------------------

def initialize_googlemaps(env_file: Optional[str] = './.env') -> Any:
    """Initialize the Google Maps client.
    Args:
        env_file (str): A file path to a .env file with a `GOOGLE_MAPS_API_KEY`
            environment variable.
    Returns:
        (Client): A googlemaps API client.
    """
    config = dotenv_values(env_file)
    google_maps_api_key = config['GOOGLE_MAPS_API_KEY']
    client = googlemaps.Client(key=google_maps_api_key)
    return client


def get_transfer_distance(
        client,
        start,
        end,
        mode='driving',
) -> Tuple[int, int]:
    """Get the distance and duration of a transfer.
    Args:
        client (Client): A googlemaps API client.
        start (string): The starting point, either lat,long as a string or an address.
        end (string): The ending point, either lat,long as a string or an address.
        mode (string): The transporation method, driving by default.
    Returns:
        (int, int): Returns a tuple of the distance in kilometers and the
            duration in seconds.
    """
    driving_distances = client.distance_matrix(start, end, mode=mode)
    elements = driving_distances['rows'][0]['elements'][0]
    km = elements['distance']['value']
    duration = elements['duration']['value']
    return km, duration


def get_transfer_route(
        client,
        start,
        end,
        departure_time=None,
        mode='driving',
) -> str:
    """Get the route of a transfer.
    Args:
        client (Client): A googlemaps API client.
        start (string): The starting point, either lat,long as a string or an address.
        end (string): The ending point, either lat,long as a string or an address.
        departure_time (datetime): The time of departure, defaults to now (optional).
        mode (string): The transporation method, driving by default (optional).
    Returns:
        (str): Returns the route as a polyline string.
    """
    if departure_time is None:
        departure_time = datetime.now()
    driving_directions = client.directions(
        start,
        end,
        mode=mode,
        departure_time=departure_time
    )
    m = driving_directions[0]['legs'][0]['distance']['value']
    min = driving_directions[0]['legs'][0]['duration']['value']
    polyline = driving_directions[0]['overview_polyline']['points']
    return m, min, polyline



# TODO: Plot miles driven (time spent driving) on a weekly, monthly basis.


# TODO: Plot of all routes driven.


# Initialize a googlemaps API client.
gmaps = initialize_googlemaps('../.env')


#------------------------------------------------------------------------------
# Save all unique routes.
#------------------------------------------------------------------------------

# Save all lab transfer routes.
lab_transfers = panel.loc[panel['recipient_type'] == 'lab']
lab_routes = []
lab_distances = []
lab_durations = []

# TODO: Only find route for unique routes.
combinations = lab_transfers.drop_duplicates(
    subset=['sender', 'recipient'],
    keep='first'
).reset_index(drop=True)
for index, row in combinations.iterrows():

    distance, duration, route_polyline = get_transfer_route(
        gmaps,
        str(row['sender_latitude']) + ',' + str(row['sender_longitude']),
        str(row['recipient_latitude']) + ',' + str(row['recipient_longitude']),
        departure_time=None,
        mode='driving',
    )
    lab_routes.append(route_polyline)
    lab_distances.append(distance)
    lab_durations.append(duration)


# TODO: Merge found routes with lab_transfers.
# # combinations['distance'] = pd.Series(distances)
# # combinations['duration'] = pd.Series(durations)
# # distances = panel.apply(lambda row : combine_distances(row, combinations), axis=1)
# # panel['distance'] = distances.apply(lambda x: [y[0] for y in x])
# # panel['duration'] = distances.apply(lambda x: [y[1] for y in x])

# lab_transfers['route'] = pd.Series(lab_routes)
# lab_transfers['distance'] = pd.Series(lab_distances)
# lab_transfers['duration'] = pd.Series(lab_durations)
lab_transfers.to_csv('D:/leaf-data/augmented/lab-transfers.csv', index=False)

#------------------------------------------------------------------------------
# How far are people transporting lab samples?
#------------------------------------------------------------------------------

# Get 10 random lab transfers.
lab_transfers = panel.loc[panel['recipient_type'] == 'lab']
lab_sample = lab_transfers.sample(10)

# # Get the routes for the transfers.
# lab_routes = []
# for index, row in lab_sample.iterrows():
#     _, _, route_polyline = get_transfer_route(
#         gmaps,
#         str(row['sender_latitude']) + ',' + str(row['sender_longitude']),
#         str(row['recipient_latitude']) + ',' + str(row['recipient_longitude']),
#         departure_time=None,
#         mode='driving',
#     )
#     lab_routes.append(route_polyline)

# lab_sample['route'] = pd.Series(lab_routes, index=lab_sample.index)
# lab_sample.to_csv('D:/leaf-data/augmented/lab-transfer-sample.csv', index=False)


def washington_route_map(
        routes=[],
        title=None,
        notes=None,
        color='yellow',
        file_name=None,
        file_type='png',
        dpi=96,
):
    """Draws a series of routes on a map of Washington State."""

    # Draw the map.
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    m = Basemap(
        lat_0=44.5,
        lon_0=-116,
        llcrnrlon=-125.0,
        llcrnrlat=44.5,
        urcrnrlon=-116,
        urcrnrlat=49.5,
        resolution='i',
    )
    m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
    m.fillcontinents(color='Grey', alpha=0.3)
    m.drawcoastlines(linewidth=0.1, color='white')
    m.drawcounties()

    # Plot the routes on the map.
    for route in routes:
        path = polyline.decode(route)
        lat = [x[0] for x in path]
        long = [x[1] for x in path]
        plt.plot(long, lat, '-', color=color)

    # Add text.
    if notes:
        plt.text(
            -125,
            44.25,
            notes,
            ha='left',
            va='bottom',
            size=20,
            color='#000000',
        )
    if title:
        plt.title(title, fontsize=28, pad=14)
    if file_name:
        fig.savefig(file_name, format=file_type, dpi=dpi, facecolor='white')
    plt.tight_layout(pad=-0.5)
    plt.show()
    return fig


# # Plot the sample of transfers to labs.
# washington_route_map(
#     lab_routes,
#     title='Sample of Washington State Lab Transfers',
#     file_name=f'{DATA_DIR}/figures/transfers-map-labs.png',
#     notes='Data Source: Washington State Leaf Traceability Data',
#     color='yellow',
# )


#------------------------------------------------------------------------------
# What does the distribution of cannabis look like?
#------------------------------------------------------------------------------

# production_transfers = panel.loc[
#     (panel['sender_type'] == 'cultivator_production') |
#     (panel['sender_type'] == 'cultivator')
# ]
# production_sample = production_transfers.sample(10)


#------------------------------------------------------------------------------
# How far are people transporting products for processing?
#------------------------------------------------------------------------------

# # Get 10 random transfers to processors.
# gmaps = initialize_googlemaps('../.env')
# processor_transfers = panel.loc[
#     (panel['recipient_type'] == 'cultivator_production') |
#     (panel['recipient_type'] == 'production')
# ]
# processor_sample = processor_transfers.sample(10)

# # Get the routes for the transfers.
# processor_routes = []
# for index, row in processor_sample.iterrows():
#     _, _, route_polyline = get_transfer_route(
#         gmaps,
#         str(row['sender_latitude']) + ',' + str(row['sender_longitude']),
#         str(row['recipient_latitude']) + ',' + str(row['recipient_longitude']),
#         departure_time=None,
#         mode='driving',
#     )
#     processor_routes.append(route_polyline)

# # Plot the sample of transfers to processors.
# washington_route_map(
#     processor_routes,
#     title='Sample of Washington State Processor Transfers',
#     file_name=f'{DATA_DIR}/figures/transfers-map-processors.png',
#     notes='Data Source: Washington State Leaf Traceability Data',
#     color='green',
# )


#------------------------------------------------------------------------------
# How far are people transporting products for retail?
#------------------------------------------------------------------------------

# # Get 10 random transfers to retailers.
# gmaps = initialize_googlemaps('../.env')
# retail_transfers = panel.loc[panel['recipient_type'] == 'dispensary']
# retail_sample = retail_transfers.sample(10)

# # Get the routes for the transfers.
# retail_routes = []
# for index, row in retail_sample.iterrows():
#     _, _, route_polyline = get_transfer_route(
#         gmaps,
#         str(row['sender_latitude']) + ',' + str(row['sender_longitude']),
#         str(row['recipient_latitude']) + ',' + str(row['recipient_longitude']),
#         departure_time=None,
#         mode='driving',
#     )
#     retail_routes.append(route_polyline)

# # Plot the sample of transfers to retailers.
# washington_route_map(
#     retail_routes,
#     title='Sample of Washington State Retail Transfers',
#     file_name=f'{DATA_DIR}/figures/transfers-map-retail.png',
#     notes='Data Source: Washington State Leaf Traceability Data',
#     color='pink',
# )