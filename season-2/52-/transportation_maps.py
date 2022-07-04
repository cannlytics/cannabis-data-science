"""
Map of Lab Sample Transfers
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 2/9/2022
Updated: 2/10/2022
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
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
})

# Read in the panel data.
panel = pd.read_csv(f'{DATA_DIR}/augmented/transfer-statistics.csv')
panel['date'] = pd.to_datetime(panel['day'])
lab_transfers = panel.loc[panel['recipient_type'] == 'lab']

# Find all unique lab transfer routes in the desired time period.
past_year_lab_transfers = lab_transfers.loc[
    (lab_transfers['date'] >= pd.to_datetime('2021-01-01')) &
    (lab_transfers['date'] <= pd.to_datetime('2021-10-31'))
]

#------------------------------------------------------------------------------
# First pass: Get routes for lab transfers in 2021.
#------------------------------------------------------------------------------

# # Initialize a googlemaps API client.
# gmaps = initialize_googlemaps('../.env')

# combinations = past_year_lab_transfers.drop_duplicates(
#     subset=['sender', 'recipient'],
#     keep='first'
# ).reset_index(drop=True)

# # Get the route for each transfer combination.
# routes = []
# distances = []
# durations = []
# for index, row in combinations.iterrows():
#     distance, duration, route = get_transfer_route(
#         gmaps,
#         str(row['sender_latitude']) + ',' + str(row['sender_longitude']),
#         str(row['recipient_latitude']) + ',' + str(row['recipient_longitude']),
#         departure_time=None,
#         mode='driving',
#     )
#     routes.append(route)
#     distances.append(distance)
#     durations.append(duration)
#     sleep(0.021)  # Sleep to abide my request limits (50 requests/s).
#     print('Found route', row['sender'], 'to', row['recipient'])

# Save the data.
# combinations.to_csv(f'{DATA_DIR}/augmented/combined-lab-transfer-statistics.csv', index=False)


#------------------------------------------------------------------------------
# Calculate statistics for the map.
#------------------------------------------------------------------------------

def combine_distances(row, combinations):
    """Combine distances with panel data (optional: optimize this code)."""
    match = combinations.loc[
        (combinations['sender'] == row['sender']) &
        (combinations['recipient'] == row['recipient'])
    ].iloc[0]
    return match['distance'], match['duration'], match['route']


# Read in the combinations.
combinations = pd.read_csv(f'{DATA_DIR}/augmented/combined-lab-transfer-statistics.csv')

# Identify the data for plotting.
data = past_year_lab_transfers

# Merge routes with the data.
values = data.apply(lambda row: combine_distances(row, combinations), axis=1)
data['distance'] = pd.Series([y[0] for y in list(values.values)], index=data.index)
data['duration'] = pd.Series([y[1] for y in list(values.values)], index=data.index)
data['route'] = pd.Series([y[2] for y in list(values.values)], index=data.index)

# Exclude routes that fall outside of Washington.
invalid_routes = []
for index, row in data.iterrows():
    path = polyline.decode(row['route'])
    lat = [x[0] for x in path]
    long = [x[1] for x in path]
    max_lat = max(lat)
    min_lat = min(lat)
    max_long = min(long)
    min_long = max(long)
    if max_lat > 49 or min_lat < 45.33 or max_long > -116.55 or max_long < -124.46:
        invalid_routes.append(index)
data.drop(invalid_routes, inplace=True)

# Identify totals.
miles_driven = round((data['distance'] * data['count']).sum() * 0.000621371192)
transit_time = round((data['duration'] * data['count']).sum() / 60 / 60)

# Identify all the labs.
labs = data.drop_duplicates(
    subset=['recipient_name', 'recipient_latitude', 'recipient_longitude'],
    keep='first'
).reset_index(drop=True)

# Count the number of transfers to each lab.
lab_stats = {}
for index, row in labs.iterrows():
    lab_specific_transfers = data.loc[data['recipient'] == row['recipient']]
    total_lab_transfers = lab_specific_transfers['count'].sum()
    proportion = total_lab_transfers / data['count'].sum()
    lab_stats[row['recipient_code']] = {
        'total': total_lab_transfers,
        'proportion': proportion,
        'code': row['recipient_code'],
        'name': row['recipient_name'],
    }

#------------------------------------------------------------------------------
# Draw the lab transfers map.
#------------------------------------------------------------------------------

# Color the labs based on number of transfers.
palette = sns.color_palette('YlOrBr', n_colors=12)
top_transfers = sorted(list(lab_stats.values()), key=lambda d: d['total']) 
for i, transfer in enumerate(top_transfers):
    lab_stats[transfer['code']]['color'] = palette[i]

# Draw the map.
fig, ax = plt.subplots(figsize=(19.8, 12))
m = Basemap(
    lat_0=44.5,
    lon_0=-116.55,
    llcrnrlon=-125.0,
    llcrnrlat=44.5,
    urcrnrlon=-116.55,
    urcrnrlat=49.5,
    resolution='i',
    ax=ax,
)
m.drawmapboundary(fill_color='#ffffff', linewidth=0) #A6CAE0
m.fillcontinents(color='Grey', alpha=0.3)
m.drawcoastlines(linewidth=0.1, color='white')
m.drawcounties()

# Plot the routes on the map.
lab_codes = sorted_nicely(list(lab_stats.keys()))
for i, code in enumerate(lab_codes):
    stats = lab_stats[code]
    lab = labs.loc[labs['recipient_code'] == code]
    lab_specific_transfers = data.loc[data['recipient_code'] == code]
    for route in lab_specific_transfers['route']:
        path = polyline.decode(route)
        lat = [x[0] for x in path]
        long = [x[1] for x in path]
        plt.plot(
            long,
            lat,
            '-',
            color=stats['color'],
            alpha=0.6,
            linewidth=2,
        )

    # Plot the labs on the map.
    label_name = f'{stats["code"]} - {stats["name"]} ({stats["total"]:,})'
    ax.plot(
        lab['recipient_longitude'],
        lab['recipient_latitude'],
        marker='o',
        color=stats['color'],
        # markersize=260 * stats['proportion'], # Optional: Size markers proportionally.
        markersize=18,
        linewidth=0,
        label=label_name,
    )
    ax.text(
        lab['recipient_longitude'].iloc[0],
        lab['recipient_latitude'].iloc[0],
        lab['recipient_code'].iloc[0],
        zorder=99,
        fontsize=24,
    )

# Add a legend.
labels = [f'{lab_stats[code]["code"]} - {lab_stats[code]["name"]} ({lab_stats[code]["total"]:,})' for code in lab_codes]
handles, _ = ax.get_legend_handles_labels()
legend = plt.legend(
    bbox_to_anchor=(0.9975, 0.1875),
    loc='upper right',
    labels=labels,
    handles=handles,
    ncol=4,
    fontsize=16,
    handlelength=0.7,
    handleheight=0.7,
    markerscale=0.75,
    title='Total Number of Transfers by Lab'
)
legend._legend_box.sep = 12

# Add caption.
plt.text(
    -125,
    44.15,
    'Data Sources: Washington State Traceability Data (January 2021 through October 2021).\nRoute data is © 2022 Google.',
)

# Add stats data block.
plt.text(
    -124.875,
    49.25,
    'Total Transfers: {:,}'.format(data['count'].sum()),
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
    'Hours of Transit: {:,}'.format(transit_time),
    size=24,
)

# Add title.
plt.title('Transfers of Washington State Lab Samples in 2021', fontsize=32, pad=14)

# Save and show the figure.
fig.savefig(
    f'{DATA_DIR}/figures/lab-transfers-2021.png',
    format='png',
    dpi=300,
    facecolor='white'
)
plt.tight_layout(pad=-0.5)
plt.show()
