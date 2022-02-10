"""
Calculate Transporation Statistics
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 2/9/2022
Updated: 2/9/2022
License: MIT License <https://opensource.org/licenses/MIT>
"""
# Standard imports.
from time import sleep

# External imports.
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # pip install basemap
import pandas as pd
import polyline # pip install polyline
import seaborn as sns

# Internal imports
from mapping_utils import (
    get_transfer_route,
    initialize_googlemaps,
)
from utils import sorted_nicely

# Specify where your data lives.
DATA_DIR = 'D:\\leaf-data'

# Define the plot style.
palette = sns.color_palette('Set2', n_colors=10)
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
})

# Dev: Read in the data.
# lab_sample = pd.read_csv(f'{DATA_DIR}/augmented/lab-transfer-sample.csv')
# processor_sample = pd.read_csv(f'{DATA_DIR}/augmented/processor-transfer-sample.csv')
# retail_sample = pd.read_csv(f'{DATA_DIR}/augmented/retail-transfer-sample.csv')
# lab_sample['distance'] = 0
# lab_sample['duration'] = 0
# lab_sample['recipient_code'] = 'TBD'

#------------------------------------------------------------------------------
# Get routes for lab transfers in 2021.
#------------------------------------------------------------------------------

# Initialize a googlemaps API client.
gmaps = initialize_googlemaps('../.env')

# Read in the panel data.
panel = pd.read_csv(f'{DATA_DIR}/augmented/transfer-statistics.csv')
panel['date'] = pd.to_datetime(panel['day'])
lab_transfers = panel.loc[panel['recipient_type'] == 'lab']

# Find all unique lab transfer routes.
past_year_lab_transfers = lab_transfers.loc[
    (lab_transfers['date'] >= pd.to_datetime('2021-01-01')) &
    (lab_transfers['date'] <= pd.to_datetime('2021-10-31'))
]
# past_year_lab_transfers['combination'] = past_year_lab_transfers['sender'] + '_to_' + past_year_lab_transfers['recipient']

combinations = past_year_lab_transfers.drop_duplicates(
    subset=['sender', 'recipient'],
    keep='first'
).reset_index(drop=True)

# Get the route for each transfer combination.
routes = []
distances = []
durations = []
for index, row in combinations.iterrows():
    distance, duration, route = get_transfer_route(
        gmaps,
        str(row['sender_latitude']) + ',' + str(row['sender_longitude']),
        str(row['recipient_latitude']) + ',' + str(row['recipient_longitude']),
        departure_time=None,
        mode='driving',
    )
    routes.append(route)
    distances.append(distance)
    durations.append(duration)
    sleep(0.021)  # Sleep to abide my request limits (50 requests/s).
    print('Found route', row['sender'], 'to', row['recipient'])


# FIXME:
def combine_distances(row, combinations):
    """Combine distances with panel data (optional: optimize this code)."""
    try:
        match = combinations.loc[
            (combinations['sender'] == row['sender']) &
            (combinations['recipient'] == row['recipient'])
        ].iloc[0]
        return match['distance'], match['duration'], match['route']
    except IndexError:
        return None, None, None

# Merge distance and duration for each observation.
# combinations['route'] = pd.Series(routes, index=combinations.index)
# combinations['distance'] = pd.Series(distances, index=combinations.index)
# combinations['duration'] = pd.Series(durations, index=combinations.index)

values = panel.apply(lambda row : combine_distances(row, combinations), axis=1)
panel['distance'] = values.apply(lambda x: [y[0] for y in x])
panel['duration'] = values.apply(lambda x: [y[1] for y in x])
panel['route'] = values.apply(lambda x: [y[2] for y in x])

# Save the data.
panel.to_csv(f'{DATA_DIR}/augmented/lab-transfer-statistics.csv', index=False)


#------------------------------------------------------------------------------
# Calculate statistics for the map.
#------------------------------------------------------------------------------

# Identify the data for plotting.
data = panel.loc[
    (panel['date'] >= pd.to_datetime('2021-01-01')) &
    (panel['date'] <= pd.to_datetime('2021-10-31'))
]

# Identify totals.
miles_driven = round((data['distance'] * data['count']).sum() * 0.000621371192)
transit_time = round((data['duration'] * data['count']).sum() / 60)

# Identify all the labs.
labs = data.drop_duplicates(
    subset=['recipient_name', 'recipient_latitude', 'recipient_longitude'],
    keep='first'
).reset_index(drop=True)

# Count the number of transfers to each lab.
lab_stats = {}
for index, row in labs.iterrows():
    total_lab_transfers = len(data.loc[data['recipient'] == row['recipient']])
    proportion = total_lab_transfers / len(data)
    lab_stats[row['recipient_code']] = {
        'total': total_lab_transfers,
        'proportion': proportion,
        'code': row['recipient_code'],
        'name': row['recipient_name'],
    }

#------------------------------------------------------------------------------
# Draw the lab transfers map. 
#------------------------------------------------------------------------------

# Draw the map.
fig = plt.figure(figsize=(19.8, 12))
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
for route in data['route']:
    path = polyline.decode(route)
    lat = [x[0] for x in path]
    long = [x[1] for x in path]
    plt.plot(
        long,
        lat,
        '-',
        color=palette[5],
        # label='Lab Transfers',
        alpha=0.6,
        linewidth=2,
    )

# Plot the labs on the map.
# Optional: Size the labs by number of transfers.
for index, row in labs.iterrows():
    lab_transfers_total = len(data.loc[data['recipient'] == row['recipient']])
    proportion = lab_transfers_total / len(data)
    plt.plot(
        row['recipient_longitude'],
        row['recipient_latitude'],
        marker='o',
        color=palette[9],
        markersize=100 * proportion,
        # label=row['recipient_code'],
    )
    plt.text(
        row['recipient_longitude'],
        row['recipient_latitude'],
        row['recipient_code'],
    )

# Add a legend.
# Optional: Remove patches.
lab_codes = sorted_nicely(list(lab_stats.keys()))
labels = [f'{lab_stats[code]["code"]}: {lab_stats[code]["name"]} ({lab_stats[code]["total"]:,})' for code in lab_codes]
plt.legend(
    bbox_to_anchor=(0.9875, 0.15),
    loc='upper right',
    labels=labels,
    ncol=4,
    fontsize='x-small',
    handlelength=0,
    title='Transfers by Lab'
)

# Add text.
plt.text(
    -125,
    44.325,
    'Data Source: Washington State Traceability Data (January 2021 to October 2021).',
)
plt.text(
    -124.875,
    49.25,
    'Total Transfers: {:,}'.format(len(data)),
    size=24,
)
plt.text(
    -124.875,
    49,
    'Miles Driven: {:,}'.format(miles_driven),
    size=24,
)
plt.text(
    -124.875,
    48.75,
    'Transit Time (hours): {:,}'.format(transit_time),
    size=24,
)
plt.title('Transfers of Washington State Lab Samples in 2021', fontsize=28, pad=10)
fig.savefig(
    f'{DATA_DIR}/figures/transfers-random-sample.png',
    format='png',
    dpi=96,
    facecolor='white'
)
plt.tight_layout(pad=-0.5)
plt.show()


#------------------------------------------------------------------------------
# Calculate supplementary statistics.
#------------------------------------------------------------------------------

# TODO: Calculate the total distance and time spent transferring lab samples.
# Remember to multiply by the count!
# total_distance = (lab_transfers['distance'] * lab_transfers['count']).sum()
# total_duration = (lab_transfers['duration'] * lab_transfers['count']).sum()
# print('Total miles travelled to transfer lab samples:', total_distance)
# print('Total time spent transferring lab samples:', total_duration)
