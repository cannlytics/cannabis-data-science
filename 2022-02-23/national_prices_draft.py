"""
Analyze National Prices
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 2/17/2022
Updated: 2/18/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script creates statistics and visualizations for cannabis
prices across various states, including: IL, MA, ME, OR, and WA.

Data sources:

    - Massachusetts Cannabis Control Commission Data Catalog
    https://masscannabiscontrol.com/open-data/data-catalog/
    
    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd

"""

# External imports.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # pip install basemap
import pandas as pd
import requests
import seaborn as sns


# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
})

# Specify where your data lives.
DATA_DIR = 'D:\\leaf-data'


#--------------------------------------------------------------------------
# Get prices for Massachusetts.
#--------------------------------------------------------------------------

# Make request to the Open Data API endpoint.
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1667.0 Safari/537.36',
}
url = 'https://masscannabiscontrol.com/resource/rqtv-uenj.json'
response = requests.get(url, headers=headers)

# Format the data appropriately.
ma_prices = pd.DataFrame(response.json())
ma_prices.rename(columns={'Date': 'date', '= 1 Oz': 'dollars_per_ounce'}, inplace=True)
ma_prices.set_index(pd.to_datetime(ma_prices['date']), inplace=True)
ma_prices['dollars_per_ounce'] = pd.to_numeric(ma_prices['dollars_per_ounce'])
ma_prices['dollars_per_gram'] = ma_prices['dollars_per_ounce'] / 28

# Plot monthly prices.
ma_prices['dollars_per_gram'].plot()


#--------------------------------------------------------------------------
# Get prices for Washington State.
#--------------------------------------------------------------------------

# Read the sample of augmented sales item data lives.
wa_sales = pd.read_csv(f'{DATA_DIR}/samples/random-sales-items-2022-02-16.csv')

# Exclude wholesale transactions.
wa_sales = wa_sales.loc[wa_sales['sale_type'] != 'wholesale']

# Drop observations with negative prices and prices in the upper quantile.
wa_sales = wa_sales.loc[wa_sales.price_total > 0]
wa_sales = wa_sales[wa_sales.price_total < wa_sales.price_total.quantile(.95)]

# Add a date column.
wa_sales['date'] = pd.to_datetime(wa_sales['created_at'])
wa_sales['day'] = wa_sales['date'].dt.date

# Estimate the average price of flower.
wa_flower_sales = wa_sales.loc[wa_sales.intermediate_type == 'usable_marijuana']

# Identify the time period.
series = wa_flower_sales.loc[
    (wa_flower_sales['date'] >= pd.to_datetime('2019-01-01')) &
    (wa_flower_sales['date'] <= pd.to_datetime('2022-01-01'))
]

# Estimate daily flower prices.
wa_avg_flower_price = series.groupby('day').mean()['price_total']
wa_avg_flower_price.index = pd.to_datetime(wa_avg_flower_price.index)

# Estimate monthly flower prices.
wa_monthly_avg_flower_price = wa_avg_flower_price.groupby(pd.Grouper(freq='M')).mean()
wa_monthly_avg_flower_price.plot()
plt.show()


#--------------------------------------------------------------------------
# Get prices for Maine
#--------------------------------------------------------------------------

# TODO: Read collected ME prices.


#--------------------------------------------------------------------------
# Get prices for Oregon
#--------------------------------------------------------------------------

# TODO: Read collected OR prices.


#--------------------------------------------------------------------------
# Get prices for Illinois
#--------------------------------------------------------------------------

# TODO: Read collected IL prices.



#--------------------------------------------------------------------------
# TODO: Visualize the data.
#--------------------------------------------------------------------------

# Kernel density estimate (KDE) plot,
# https://seaborn.pydata.org/generated/seaborn.kdeplot.html#seaborn.kdeplot
# sns.kdeplot(
#    data=tips,
#    x='total_bill',
#    hue='size',
#    fill=True,
#    common_norm=False,
#    palette='crest',
#    alpha=.5,
#    linewidth=0,
# )

# Overlapping densities (ridge plot).
# https://seaborn.pydata.org/examples/kde_ridgeplot.html


# 4D Scatterplot (X, Y, color, and size)
# sns.scatterplot(
#     data=tips,
#     x='total_bill',
#     y='tip',
#     hue='size',
#     size='size',
#     sizes=(20, 200),
#     hue_norm=(0, 7),
#     legend='full',
# )


# Scatterplot by group.
# index = pd.date_range("1 1 2000", periods=100, freq="m", name="date")
# data = np.random.randn(100, 4).cumsum(axis=0)
# wide_df = pd.DataFrame(data, index, ["a", "b", "c", "d"])
# sns.scatterplot(data=wide_df)


# Linear regression with marginal distributions
# https://seaborn.pydata.org/examples/regression_marginals.html
# g = sns.jointplot(
#     x='total_bill',
#     y='tip',
#     data=tips,
#     kind='reg',
#     truncate=False,
#     xlim=(0, 60),
#     ylim=(0, 12),
#     color='m',
#     height=7
# )


# Plotting model residuals
# https://seaborn.pydata.org/examples/residplot.html
# sns.residplot(x=x, y=y, lowess=True, color='g')

#--------------------------------------------------------------------------
# TODO: Create national choropleth of prices.
#--------------------------------------------------------------------------



