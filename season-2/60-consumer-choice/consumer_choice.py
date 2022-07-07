"""
Consumer Choice (Part 1)
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 4/5/2022
Updated: 4/6/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description: This script begins to analyze consumer choice in Massachusetts
by measuring inflation for various product types.

Data sources:

    - Massachusetts Cannabis Control Commission Data Catalog
    https://masscannabiscontrol.com/open-data/data-catalog/
"""
# External imports.
import matplotlib.pyplot as plt
import seaborn as sns

# Internal imports.
from opendata import CCC

#--------------------------------------------------------------------------
# Wrangle Massachussetts Open Data.
#--------------------------------------------------------------------------

# Initialize a CCC Data Catalog client.
ccc = CCC()

# Get licensees data and stats.
licensees = ccc.get_licensees()
licensees_approved = ccc.get_licensees('approved')
licensees_pending = ccc.get_licensees('pending')
demographics = ccc.get_licensees('demographics')
under_review_stats = ccc.get_licensees('under-review-stats')
application_stats = ccc.get_licensees('application-stats')

# Get retail stats.
sales_stats = ccc.get_retail('sales-stats')
sales_weekly = ccc.get_retail('sales-weekly')
prices = ccc.get_retail('price-per-ounce')

# Get agent stats.
gender_stats = ccc.get_agents('gender-stats')
ethnicity_stats = ccc.get_agents('ethnicity-stats')

# Get medical stats.
medical = ccc.get_medical()

# Get cultivation data.
plants = ccc.get_plants()

# Get sales data.
sales = ccc.get_sales()


#--------------------------------------------------------------------------
# Look at the data!
#--------------------------------------------------------------------------

# Quick look at indoor vs. outdoor (for fun!).
cultivators = licensees_approved.loc[licensees_approved['license_type'] == 'Marijuana Cultivator']
indoor = cultivators.loc[cultivators['cultivation_environment'] == 'Indoor']
outdoor = cultivators.loc[cultivators['cultivation_environment'] == 'Outdoor']
percent_indoor = len(indoor)/ len(cultivators)
percent_outdoor = len(outdoor)/ len(cultivators)

# Plot indoor vs outdoor.
colors = sns.color_palette('Set2', n_colors=2)
fig, ax = plt.subplots(figsize=(12, 8))
plt.pie(
    [percent_indoor, percent_outdoor],
    labels=['Indoor', 'Outdoor'],
    colors=colors,
    autopct='%.0f%%'
)
plt.title('Breakdown of Indoor vs. Outdoor Cultivators in Massachussetts')
plt.show()


#--------------------------------------------------------------------------
# Curate the data.
#--------------------------------------------------------------------------

# Identify sales by product type.
flower_sales = sales.loc[
    (sales['productcategoryname'] == 'Buds') &
    (sales['unitofmeasurename'] == 'Grams')
]
oil_sales = sales.loc[
    (sales['productcategoryname'] == 'Concentrate') &
    (sales['unitofmeasurename'] == 'Grams')
]
vape_sales = sales.loc[
    (sales['productcategoryname'] == 'Vape Product')
]
beverage_sales = sales.loc[
    (sales['productcategoryname'] == 'Infused Beverage')
]
edible_sales = sales.loc[
    (sales['productcategoryname'] == 'Infused (edible)')
]
preroll_sales = sales.loc[
    (sales['productcategoryname'] == 'Raw Pre-Rolls') |
    (sales['productcategoryname'] == 'Infused Pre-Rolls')
]


def calculate_avg_price(
        series,
        price='totalprice',
        quantity='quantity',
        index='saledate',
        period='M'
):
    """Calculate average price for a series given the price and quantity fields."""
    price = series[price].div(series[quantity])
    price.index = series[index]
    return price.resample(period).mean()

# Format prices.
price_per_gram_flower = calculate_avg_price(flower_sales)
price_per_gram_oil = calculate_avg_price(oil_sales)
price_per_vape = calculate_avg_price(vape_sales)
price_per_beverage = calculate_avg_price(beverage_sales)
price_per_edible = calculate_avg_price(edible_sales)
price_per_preroll = calculate_avg_price(preroll_sales)


def calculate_inflation_rate(series):
    """Calculate the inflation rate for a series."""
    lag = series.shift(1)
    return (series - lag) / lag


# Calculate inflation for various product types.
flower_inflation_rate = calculate_inflation_rate(price_per_gram_flower)
oil_inflation_rate = calculate_inflation_rate(price_per_gram_oil)
vape_inflation_rate = calculate_inflation_rate(price_per_vape)
beverage_inflation_rate = calculate_inflation_rate(price_per_beverage)
edible_inflation_rate = calculate_inflation_rate(price_per_edible)
preroll_inflation_rate = calculate_inflation_rate(price_per_preroll)

# Visualize inflation.
fig, ax = plt.subplots(figsize=(12, 8))
flower_inflation_rate.tail(12).plot(label='Flower')
oil_inflation_rate.tail(12).plot(label='Oil')
vape_inflation_rate.tail(12).plot(label='Vape')
beverage_inflation_rate.tail(12).plot(label='Beverage')
edible_inflation_rate.tail(12).plot(label='Edible')
preroll_inflation_rate.tail(12).plot(label='Preroll')
plt.hlines(0, xmin=flower_inflation_rate.index[0], xmax=flower_inflation_rate.index[-1])
plt.legend()
plt.show()
