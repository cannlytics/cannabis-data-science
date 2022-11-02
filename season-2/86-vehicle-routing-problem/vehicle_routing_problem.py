"""
Vehicle Routing Problem
Copyright (c) 2022 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 9/22/2022
Updated: 10/5/2022
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# Standard imports.
import os
from time import sleep

# External imports.
from cannlytics.data.gis import (
    initialize_googlemaps,
    get_transfer_route,
)
from dotenv import dotenv_values
import folium
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


#------------------------------------------------------------------------------
# Find the route from each cannabis retailer to Humboldt County.
#------------------------------------------------------------------------------

def create_retailer_map(
        df,
        color='crimson',
        filename=None,
        lat='premise_latitude',
        long='premise_longitude',
    ):
    """Create a map of licensed retailers."""
    m = folium.Map(
        location=[39.8283, -98.5795],
        zoom_start=3,
        control_scale=True,
    )
    for _, row in df.iterrows():
        folium.Circle(
            radius=5,
            location=[row[lat], row[long]],
            color=color,
        ).add_to(m)
    if filename:
        m.save(filename)
    return m


# Create the retailers map.
map_file = '../../.datasets/figures/cannabis-licenses-map.html'
m = create_retailer_map(retailers, filename=map_file)


#------------------------------------------------------------------------------
# Find the route from each cannabis retailer to Humboldt County.
#------------------------------------------------------------------------------

# Initialize a googlemaps API client.
config = dotenv_values('../../.env')
api_key = config['GOOGLE_MAPS_API_KEY']

# Get the route for each transfer combination.
routes = []
distances = []
durations = []
for index, row in retailers.iterrows():
    distance, duration, route = get_transfer_route(
        api_key,
        str(row['sender_latitude']) + ',' + str(row['sender_longitude']),
        str(row['recipient_latitude']) + ',' + str(row['recipient_longitude']),
        departure_time=None,
        mode='driving',
    )
    routes.append(route)
    distances.append(distance)
    durations.append(duration)
    sleep(0.021)  # Sleep to abide my request limits (50 requests/s).

# Save the data.
retailers.to_excel(f'{DATA_DIR}/geocoded-retailers.xlsx')


#------------------------------------------------------------------------------
# Draw the lab transfers map.
#------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # pip install basemap
import pandas as pd
import polyline # pip install polyline
import seaborn as sns

# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
})

# Draw the map.
fig, ax = plt.subplots(figsize=(19.8, 12))
m = Basemap(
    lat_0=39.8283,
    lon_0=-98.5795,
    llcrnrlon=-124.848974,
    llcrnrlat=24.396308,
    urcrnrlon=-66.885444,
    urcrnrlat=49.384358,
    resolution='c',
    ax=ax,
)
m.drawmapboundary(fill_color='#ffffff', linewidth=0) #A6CAE0
m.fillcontinents(color='Grey', alpha=0.3)
m.drawcoastlines(linewidth=0.1, color='white')
m.drawcounties()

# Plot the routes on the map.
for route in routes:
    path = polyline.decode(route)
    lat = [x[0] for x in path]
    long = [x[1] for x in path]
    plt.plot(
        long,
        lat,
        '-',
        color='lightgreen',
        alpha=0.6,
        linewidth=2,
    )

# Plot the retailers on the map.
for index, retailer in retailers.iterrows():
    ax.plot(
        retailer['premise_longitude'],
        retailer['premise_latitude'],
        marker='o',
        color='crimson',
        markersize=5,
        linewidth=0,
    )

# Add title.
title = 'Predicted Future Cannabis Retail Routes to Humboldt County'
plt.title(title, fontsize=32, pad=14)

# TODO: Add notes with data sources.

# Save and show the figure.
fig.savefig(
    f'{DATA_DIR}/map.png',
    format='png',
    dpi=300,
    facecolor='white'
)
plt.tight_layout(pad=-0.5)
plt.show()
