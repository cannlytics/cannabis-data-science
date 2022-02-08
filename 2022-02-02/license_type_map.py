"""
Plot Washington Licensees by License Type | Cannabis Data Science
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 2/2/2022
Updated: 2/2/2022
License: MIT License <https://opensource.org/licenses/MIT>
"""
# External imports.
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # pip install basemap
import pandas as pd
import seaborn as sns


# Specify where your data lives.
DATA_DIR = 'D:\\leaf-data'

#------------------------------------------------------------------------------
# Read in licensees data.
#------------------------------------------------------------------------------

# Read licensees.
licensee_fields = {
    'global_id': 'string',
    'latitude': 'float',
    'longitude': 'float',
    'name': 'string',
    'type': 'string',
}
licensees = pd.read_csv(
    f'{DATA_DIR}/augmented/augmented-washington-state-licensees.csv',
    usecols=list(licensee_fields.keys()),
    dtype=licensee_fields,
)

# Restrict to main license types.
license_types = [
    'cultivator',
    'production',
    'cultivator_production',
    'dispensary',
]
licensees = licensees.loc[
    licensees['type'].isin(license_types)
]


#------------------------------------------------------------------------------
# Scatterplot with different colors for different license types.
#------------------------------------------------------------------------------

# Define plot style.
palette = sns.color_palette('Set2', n_colors=10)
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
})

# Make the background map.
fig = plt.gcf()
fig.set_size_inches(15, 10) # 19.2, 10.8
m = Basemap(
    lat_0=44.5,
    lon_0=-115.75,
    llcrnrlon=-125.0,
    llcrnrlat=44.5,
    urcrnrlon=-115.75,
    urcrnrlat=49.5,
    resolution='i',
)
m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)
m.fillcontinents(color='Grey', alpha=0.3)
m.drawcoastlines(linewidth=0.1, color='white')
m.drawcounties()

# Add bubbles at each licensee coordinates.
licensees.rename(columns={'type': 'Type'}, inplace=True)
x, y = m(licensees['longitude'], licensees['latitude'])
points = sns.scatterplot(
    x,
    y,
    hue=licensees['Type'],
    s=200,
    marker='o',
    palette=palette[:4],
    hue_order=license_types,
    alpha=0.8,
    # color=palette[2],
)

# Optional: Label / plot cities.

# Optional: Style legend.
# plt.legend().remove()

# Add notes and data source.
plt.text(
    -125,
    44.25,
    'Data Source: Washington State Leaf Traceability Data (November 2021).',
    ha='left',
    va='bottom',
    size=20,
    color='#000000'
)

# Add title.
plt.title('Washington State Licensees', fontsize=28, pad=14)

# Save and show the figure.
fig.savefig(
    f'{DATA_DIR}/figures/licensees_scatterplot.png',
    format='png',
    dpi=96,
    facecolor='white',
)
plt.tight_layout(pad=-0.5)
plt.show()
