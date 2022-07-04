"""
Spatial Analysis - Sales Prices in Washington State
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 2/16/2022
Updated: 2/16/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script creates a histogram of flower prices from the
Washington State traceability data (2021-01-31 to 11-10-2021).

Data sources:

    - Random sample of sales items
    https://cannlytics.page.link/cds53

"""
# External imports.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 32,
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

# Drop observations with negative prices and prices in the upper quantile.
data = data.loc[data.price_total > 0]
data = data[data.price_total < data.price_total.quantile(.95)]

# Add a date column.
data['date'] = pd.to_datetime(data['created_at'])
data['day'] = data['date'].dt.date

# Identify flower sales.
sample_type = 'usable_marijuana'
sample_type_data = data.loc[data.intermediate_type == sample_type]


#--------------------------------------------------------------------------
# Create a histogram of flower prices.
#--------------------------------------------------------------------------

def pdf(mu, sigma, bins):
    """Calculate a PDF given mean, standard deviation, and number of bins."""
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu)**2 / (2 * sigma**2))

# Identify the series.
series = sample_type_data['price_total'].loc[
    (sample_type_data['date'] >= pd.to_datetime('2021-01-01')) &
    (sample_type_data['date'] <= pd.to_datetime('2022-01-01'))
]

# Define a color palette.
colors = sns.color_palette('Set2', n_colors=10)
green = colors[0]
orange = colors[9]

# Create a histogram.
fig, ax = plt.subplots(figsize=(19.8, 12))
n, bins, patches = ax.hist(
    series,
    bins=200,
    density=1,
    color=green,
    alpha=0.8,
    edgecolor='#ccc',
)

# Calculate interesting statistics.
sigma = series.std()
mu = series.mean()
median = np.percentile(series, 50)
lower = np.percentile(series, 10)
upper = np.percentile(series, 90)

# Plot the PDF.
pdf_values = pdf(mu, sigma, bins)
ax.plot(bins, pdf_values, '--', color=orange, alpha=.6)

# Shade the inner 90% of values.
pdf_bins = bins[(bins >= lower) & (bins <= upper)]
ax.fill_between(
    pdf_bins,
    pdf(mu, sigma, pdf_bins),
    0,
    alpha=.3,
    color=orange
)

# Annotate the median and lower and upper percentiles.
summary_stats = series.describe()
median = summary_stats['50%']
ax.annotate(
    'Median: $%0.2f' % round(median, 2),
    xy=(median, pdf(mu, sigma, median) + 0.005),
    fontsize=32,
    horizontalalignment='center',
    verticalalignment='bottom',
)
ax.annotate(
    'Q10: $%0.2f' % round(lower, 2),
    xy=(lower, pdf(mu, sigma, lower) - 0.005),
    fontsize=32,
    horizontalalignment='center',
    verticalalignment='bottom',
)
ax.annotate(
    'Q90: $%0.2f' % round(upper, 2),
    xy=(upper, pdf(mu, sigma, upper) + 0.005),
    fontsize=32,
    horizontalalignment='center',
    verticalalignment='bottom',
)

# Add text.
ax.set_xlabel('Price ($)')
ax.set_ylabel('Density')
ax.set_title(
    'Cannabis Flower Sale Prices in Washington State in 2021',
    fontsize=42,
    pad=24,
)
plt.text(
    0,
    -0.0575,
    """Data: A random sample of 36,481 “usable marijuana” sale items.
Data Source: Washington State Traceability Data from January 2021 to November 2021.
Notes: The top 5% of sale item observations by price were excluded as outliers.
The estimated probability distribution is depicted by the dotted orange line.""",
    fontsize=32,
)
fig.savefig(
    f'{DATA_DIR}/figures/histogram_{sample_type}_prices_2021.png',
    format='png',
    dpi=300,
    bbox_inches='tight',
)
plt.show()
