"""
Calculate Plants Statistics
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 1/29/2022
Updated: 2/2/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script calculates various statistics from the plants data using
relevant fields from the lab results, licensees, inventories,
inventory types, sales, and strains datasets.

Data sources:

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd?page=1
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd?page=2

Data Guide:

    - Washington State Leaf Data Systems Guide
    https://lcb.wa.gov/sites/default/files/publications/Marijuana/traceability/WALeafDataSystems_UserManual_v1.37.5_AddendumC_LicenseeUser.pdf

Resources:

    - Plotting data on a map
    https://matplotlib.org/basemap/users/examples.html

"""
# Standard imports.
import gc
from time import sleep

# External imports.
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # pip install basemap
import pandas as pd
import seaborn as sns

# Internal imports
from utils import format_thousands


# Specify where your data lives.
DATA_DIR = 'D:\\leaf-data'


#------------------------------------------------------------------------------
# Read plants data.
#------------------------------------------------------------------------------

# Define useful metadata about the plants data.
plants_datatypes = {
    'mme_id': 'string',
}
plants_date_fields = [
    'created_at',
    'updated_at',
]
plants_columns = list(plants_datatypes.keys()) + plants_date_fields

# Example: Read in 10% of the plants data.
# plants_size = get_number_of_lines(f'{DATA_DIR}/Plants_0.csv') # 30_538_973
# plants = pd.read_csv(
#     f'{DATA_DIR}/Plants_0.csv',
#     sep='\t',
#     encoding='utf-16',
#     usecols=plants_columns,
#     dtype=plants_datatypes,
#     parse_dates=plants_date_fields,
#     # skiprows=skiprows,
#     nrows=plants_size * .1,
# )

#------------------------------------------------------------------------------
# Count the number of plants by date by licensee.
# Future work: Determine which licensees are operating at any given time.
# Optimize: This section currently takes a long time to run.
#------------------------------------------------------------------------------

# Specify the time range to calculate statistics.
time_range = pd.date_range(start='2018-01-31', end='2021-11-30')

# Read all plants, then iterate over dates.
plants = pd.read_csv(
    f'{DATA_DIR}/Plants_0.csv',
    sep='\t',
    encoding='utf-16',
    usecols=plants_columns,
    dtype=plants_datatypes,
    parse_dates=plants_date_fields,
)

# Iterate over the days, counting plants in total and by licensee.
daily_plant_count = []
plant_panel = []

for date in time_range:

    day = date.date()

    # Count the total number of plants.
    current_plants = plants.loc[
        (plants.created_at.dt.date >= day) &
        (plants.updated_at.dt.date <= day)
    ]
    total_plants = len(current_plants)

    # Count the total plants by licensee.
    licensees = list(current_plants.mme_id.unique())
    licensees_total_plants = {}
    for licensee in licensees:
        licensee_plants = current_plants.loc[current_plants['mme_id'] == licensee]
        licensees_total_plants[licensee] = len(licensee_plants)

    # Keep track of the daily totals.
    daily_plant_count.append([date, total_plants, len(licensees)])

    # Keep track of the totals by licensee.
    for mme_id, count in licensees_total_plants.items():
        plant_panel.append({
            'date': date,
            'mme_id': mme_id,
            'total_plants': count,
        })
    print(day, 'total plants:', total_plants)

# Clean up unused variables.
try:
    del current_plants
    del licensee_plants
    del plants
    gc.collect()
except NameError:
    pass

# Save the daily total series.
daily_plant_data = pd.DataFrame(daily_plant_count)
daily_plant_data.columns = ['date', 'total_plants', 'total_cultivators']
daily_plant_data.to_csv(f'{DATA_DIR}/augmented/daily_plant_data.csv')

# Save the daily panel series.
panel_plant_data = pd.DataFrame(plant_panel)
panel_plant_data.to_csv(f'{DATA_DIR}/augmented/daily_licensee_plant_data.csv')


#------------------------------------------------------------------------------
# Augment plants data with licensee data.
#------------------------------------------------------------------------------

# Read the statistics.
daily_plant_data = pd.read_csv(
    f'{DATA_DIR}/augmented/daily_plant_data.csv',
    index_col=0,
)
panel_plant_data = pd.read_csv(
    f'{DATA_DIR}/augmented/daily_licensee_plant_data.csv',
    index_col=0,
)

# Add a time index.
daily_plant_data.index = pd.to_datetime(daily_plant_data.date)

# Aggregate by month.
monthly_plants = daily_plant_data.resample('M').mean().pad()
monthly_plants['month'] = monthly_plants.index.strftime('%Y-%m')

# Add latitude and longitude for each day / licensee observation.
licensee_fields = {
    'global_id': 'string',
    'latitude': 'float',
    'longitude': 'float',
    'name': 'string',
}
licensees = pd.read_csv(
    f'{DATA_DIR}/augmented/augmented-washington-state-licensees.csv',
    usecols=list(licensee_fields.keys()),
    dtype=licensee_fields,
)
geocoded_plant_data = pd.merge(
    left=panel_plant_data,
    right=licensees,
    how='left',
    left_on='mme_id',
    right_on='global_id'
)
geocoded_plant_data = geocoded_plant_data.loc[
    (~geocoded_plant_data.longitude.isnull()) &
    (~geocoded_plant_data.latitude.isnull())
]

# Aggregate data by the month.
geocoded_plant_data['time'] = pd.to_datetime(geocoded_plant_data['date'])
geocoded_plant_data['month'] = geocoded_plant_data.time.dt.strftime('%Y-%m')
# monthly_plant_data = geocoded_plant_data.groupby(['month', 'mme_id']).mean()


#------------------------------------------------------------------------------
# Plot the data on a map.
#------------------------------------------------------------------------------

# Define the plot style.
palette = sns.color_palette('Set2', n_colors=10)
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
})

# Create a plot for each day.
dates = list(geocoded_plant_data.month.unique())
for date in dates[:5]:

    # Identify the data for the time period.
    data = geocoded_plant_data.loc[
        geocoded_plant_data['month'] == date
    ]
    month = pd.to_datetime(data.iloc[0]['date']).strftime('%B %Y')

    # Make the background map.
    fig = plt.gcf()
    fig.set_size_inches(15, 10) # 19.2, 10.8
    m = Basemap(
        lat_0=44.5,
        lon_0=-116,
        llcrnrlon=-125.0,
        llcrnrlat=44.5,
        urcrnrlon=-116,
        urcrnrlat=49.5,
        resolution='i',
        # projection='tmerc',
    )
    m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
    m.fillcontinents(color='Grey', alpha=0.3)
    m.drawcoastlines(linewidth=0.1, color='white')
    m.drawcounties()

    # Add bubbles at each position.
    x, y = m(data['longitude'], data['latitude'])
    points = plt.scatter(
        x,
        y,
        s=data['total_plants'] * 0.25,
        marker='s',
        color=palette[0],
        alpha=0.8,
        zorder=1
        # cmap, # Optional: Add color map dimension.
    )

    # Optional: Label / plot cities.

    # Optional: Label top 5 cultivators.
    # top_producers = data.sort_values('total_plants', ascending=False).iloc[0:5]
    # for index, producer in top_producers.iterrows():
    #     x, y = m(producer['longitude'], producer['latitude'])
    #     plt.annotate(
    #         producer['name'],
    #         xy=(x, y),
    #         xycoords='data',
    #         xytext=(x, y),
    #         # textcoords='offset points',
    #         color='black',
    #         fontsize=14
    #         # arrowprops=dict(arrowstyle='fancy', color='g')
    #     )

    # TODO: Add legend?

    # Notes and data source.
    plt.text(
        -125,
        44.25,
        'Data Source: Washington State Leaf Traceability Data',
        ha='left',
        va='bottom',
        size=20,
        color='#000000',
    )

    # Annotate statistics
    # stats = monthly_plants.loc[monthly_plants.month == date].iloc[0]
    total_plants = data['total_plants'].sum()
    if total_plants > 1_000:
        total_plants = format_thousands(total_plants)
    total_cultivators = len(data['mme_id'].unique())
    plt.text(
        -124.875,
        49.125,
        month,
        ha='left',
        va='bottom',
        size=22,
        color='#000000'
    )
    plt.text(
        -124.875,
        48.875,
        'Cultivators: %i' % total_cultivators,
        ha='left',
        va='bottom',
        size=22,
        color='#000000',
    )
    plt.text(
        -124.875,
        48.625,
        'Total Plants: %s' % total_plants,
        ha='left',
        va='bottom',
        size=22,
        color='#000000',
    )
    plt.title('Washington State Canopy', fontsize=28, pad=14)
    fig.savefig(
        f'{DATA_DIR}/figures/canopy-{date}.pdf',
        format='pdf',
        dpi=96,
        facecolor='white'
    )
    plt.tight_layout(pad=-0.5)
    plt.show()
    sleep(.05)


#------------------------------------------------------------------------------
# TODO: Plot weekly and monthly average number of plants.
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# TODO: Video Plot of Plant Production (The Plant Canopy).
#  f"frames/frame_{ii:04d}.png",
# ffmpeg -framerate 21 -i D:/leaf-data/figures/canopy-%4d.png -c:v h264 -r 30 -s 1920x1080 ./canopy.mp4
# Optional: Add Vivaldi music!
# ffmpeg -i input.mp4 -i music.mp3 -codec:v copy -codec:a aac -b:a 192k \
# -strict experimental -ac 2 -shortest output.mp4
#------------------------------------------------------------------------------

# TODO: Create a video from each image.
