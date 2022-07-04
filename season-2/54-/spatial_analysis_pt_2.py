"""
Spatial Analysis - Sales Prices in Washington State
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 2/16/2022
Updated: 2/16/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script calculates various sales statistics from the
Washington State traceability data (2018-01-31 to 11-10-2021).

Data sources:

    - Random sample of sales items
    https://cannlytics.page.link/cds53

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd?page=1
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd?page=2

Data Guide:

    - Washington State Leaf Data Systems Guide
    https://lcb.wa.gov/sites/default/files/publications/Marijuana/traceability/WALeafDataSystems_UserManual_v1.37.5_AddendumC_LicenseeUser.pdf
"""

# External imports.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # pip install basemap
import pandas as pd
import seaborn as sns


# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
})


#--------------------------------------------------------------------------
# Analyze the data.
#--------------------------------------------------------------------------

# Specify where the data lives.
DATA_DIR = 'D:\\leaf-data'
DATA_FILE = f'{DATA_DIR}/samples/random-sales-items-2022-02-16.csv'

# Read in the data.
data = pd.read_csv(DATA_FILE)


#--------------------------------------------------------------------------
# Clean the data.
#--------------------------------------------------------------------------

# Determine wholesale vs retail transactions.
data = data.loc[data['sale_type'] != 'wholesale']

# Drop observations with negative prices? or prices > $1000?
data = data.loc[data.price_total > 0]

# Add a date column.
data['date'] = pd.to_datetime(data['created_at'])
data['day'] = data['date'].dt.date


#--------------------------------------------------------------------------
# 1. Calculate the average price by sample type by day, month, and year.
#--------------------------------------------------------------------------

# Estimate the average_price by sample type.
sample_types = list(data.intermediate_type.unique())
for sample_type in sample_types:

    print(sample_type, round(len(data.loc[data.intermediate_type == sample_type]) / len(data), 2))

    sample_type_data = data.loc[data.intermediate_type == sample_type]
    print(sample_type, 'avg price:', sample_type_data.price_total.mean())

# Estimate the average price by sample type by day.
sample_timeseries = {}
for sample_type in sample_types:
    sample_type_data = data.loc[data.intermediate_type == sample_type]
    sample_prices = sample_type_data.groupby('day').mean()['price_total']
    sample_timeseries[sample_type] = sample_prices

# Estimate the average price by sample type by month.
monthly_sample_timeseries = {}
for sample_type in sample_types:
    timeseries = sample_timeseries[sample_type]
    monthly_timeseries = timeseries.groupby(pd.Grouper(freq='M')).mean()
    monthly_sample_timeseries[sample_type] = monthly_timeseries

# Estimate the average price by sample type by year.
annual_sample_timeseries = {}
for sample_type in sample_types:
    timeseries = sample_timeseries[sample_type]
    annual_timeseries = timeseries.groupby(pd.Grouper(freq='Y')).mean()
    annual_sample_timeseries[sample_type] = annual_timeseries

# Optional: Create nice visualizations of the timeseries.


#--------------------------------------------------------------------------
# 2. Create histograms, with estimated probability density functions,
# for prices by sample type.
#--------------------------------------------------------------------------

# Define a color palette.
colors = sns.color_palette('Set2', n_colors=10)

# Create histograms for each sample type.
for i, sample_type in enumerate(sample_types):
    sample_type_data = data.loc[data.intermediate_type == sample_type]
    series = sample_type_data['price_total']
    fig, ax = plt.subplots(figsize=(12, 7))
    n, bins, patches = ax.hist(
        series,
        40,
        density=1,
        color=colors[i],
        alpha=0.8
    )
    sigma = series.std()
    mu = series.mean()
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    ax.plot(bins, y, '--', color='r')
    ax.set_xlabel('Price')
    ax.set_ylabel('Probability density')
    ax.set_title('Histogram of %s Price' % sample_type.replace('_', ' ').title())
    # Optional: Also display mean and std. For example: r'$\mu={}$, $\sigma={}$'
    fig.tight_layout()
    plt.show()

#--------------------------------------------------------------------------
# 3. Create choropleths of average price by sample type by zip code.
#--------------------------------------------------------------------------

# Create a figure for each sample type.
for sample_type in sample_types:

    # Get the sample data.
    sample_type_data = data.loc[data.intermediate_type == sample_type]

    # Get the average price by Zip code (optional: use county instead?).
    zip_code_prices = sample_type_data.groupby('postal_code').mean()['price_total']

    # TODO: Create a choropleth!

    # Create a basemap.
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

    # TODO: Create a colormap.

    # TODO: Shade each zip code (or county instead?) by its avg. price.

    # Optional: Label counties with their avg. price.

    # TODO: Add a legend.
    # labels = [f'{lab_stats[code]["code"]} - {lab_stats[code]["name"]} ({lab_stats[code]["total"]:,})' for code in lab_codes]
    # handles, _ = ax.get_legend_handles_labels()
    # legend = plt.legend(
    #     bbox_to_anchor=(0.9975, 0.1875),
    #     loc='upper right',
    #     labels=labels,
    #     handles=handles,
    #     ncol=4,
    #     fontsize=16,
    #     handlelength=0.7,
    #     handleheight=0.7,
    #     markerscale=0.75,
    #     title='Average Item Price'
    # )
    # legend._legend_box.sep = 12

    # Add caption.
    plt.text(
        -125,
        44.15,
        'Data Sources: Washington State Traceability Data (January 2018 to November 2021).',
    )

    # Optional: Add a data block with sample type statistics (mean / std.)

    # Add title.
    sample_type_name = sample_type.replace('_', ' ').title()
    plt.title(
        f'Average Price of {sample_type_name} by Zip Code in Washington State',
        fontsize=32,
        pad=14
    )

    # Save and show the figure.
    # fig.savefig(
    #     f'{DATA_DIR}/figures/choropleth_avg_prices_{sample_type}.png',
    #     format='png',
    #     dpi=300,
    #     facecolor='white'
    # )
    plt.tight_layout(pad=-0.5)
    plt.show()
    


# Future work: Get sample weights from product name.
# Make sure to lower the case.
common_weights = {
    '1g': {'qty': 1, 'uom': 'gm'},
    '1.0g': {'qty': 1, 'uom': 'gm'},
    '1.0 g': {'qty': 1, 'uom': 'gm'},
    '.75g': {'qty': 0.75, 'uom': 'gm'},
    '.8g': {'qty': 0.8, 'uom': 'gm'},
    '1.5g': {'qty': 1, 'uom': 'gm'},
    '2g': {'qty': 2, 'uom': 'gm'},
    '2.5g': {'qty': 2.5, 'uom': 'gm'},
    '3.5g': {'qty': 3.5, 'uom': 'gm'},
    '3.50 grams': {'qty': 3.5, 'uom': 'gm'},
    '3.5 gram': {'qty': 3.5, 'uom': 'gm'},
    'eighth': {'qty': 3.5, 'uom': 'gm'},
    '7.5g': {'qty': 7.5, 'uom': 'gm'},
    # 10PK=100MG | 10pack 100mgTHC
    # Joints .5g (2)
    # 0.5g x 2
    # 1\/8 oz
}
common_multipliers = {
    'x 2': 2,
    # '(2)': 2,
    '2 Pk': 2,
    '2pack': 2,
    '3pk': 3,
    '7 x ': 7,
    '10-pack': 10,
}

