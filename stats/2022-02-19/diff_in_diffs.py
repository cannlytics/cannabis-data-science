"""
Difference-in-Differences Model
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 2/17/2022
Updated: 2/19/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script uses a difference-in-differences model to answer the
question of what effect, if any, did the temporary closure of retail in
Massachusetts from March to May of 2020 have on prices?

Data sources:
    
    - Massachusetts Cannabis Control Commission Data Catalog
    https://masscannabiscontrol.com/open-data/data-catalog/
    
    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd

    - Random Sample of Sale Items
    https://cannlytics.page.link/cds53

Resources:

    - Difference-in-Difference Estimation
    https://www.publichealth.columbia.edu/research/population-health-methods/difference-difference-estimation
    
    
    - Difference in Differences in Python + Pandas
    https://stackoverflow.com/a/37297258

"""

# External imports.
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import statsmodels.formula.api as sm

# Internal imports.
from utils import end_of_period_timeseries


# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
})
sns.set_palette('Set2', 10, .75)


#--------------------------------------------------------------------------
# Get Washington State price data.
#--------------------------------------------------------------------------

# Specify where the data lives.
DATA_DIR = 'D:\\leaf-data'
DATA_FILE = f'{DATA_DIR}/samples/random-sales-items-2022-02-16.csv'

# Read in the data.
data = pd.read_csv(DATA_FILE)

# Determine wholesale vs retail transactions.
data = data.loc[data['sale_type'] != 'wholesale']

# Drop observations with negative prices and prices in the upper quantile.
data = data.loc[data.price_total > 0]
data = data[data.price_total < data.price_total.quantile(.95)]

# Add a date column.
data['date'] = pd.to_datetime(data['created_at'])
data['day'] = data['date'].dt.date

# Estimate the average price of flower.
sample_type = 'usable_marijuana'
sample_type_data = data.loc[data.intermediate_type == sample_type]

# Identify the time period.
series = sample_type_data.loc[
    (sample_type_data['date'] >= pd.to_datetime('2018-11-01')) &
    (sample_type_data['date'] <= pd.to_datetime('2021-11-30'))
]

# Estimate daily flower prices.
daily_avg_flower_price = series.groupby('day').mean()['price_total']
daily_avg_flower_price.index = pd.to_datetime(daily_avg_flower_price.index)

# Estimate monthly flower prices.
monthly_avg_flower_price = daily_avg_flower_price.groupby(pd.Grouper(freq='M')).mean()

# Plot monthly WA prices.
monthly_avg_flower_price.plot()
plt.title('Washington State Average Flower Price per Item')
plt.ylabel('Dollars ($)')
plt.show()

# TODO: Calculate price per quantity (so we can compare apples to apples i.e.
# price per gram to price per gram). This may require scrounging the weight
# and unit of measure from the product name if possible.


#--------------------------------------------------------------------------
# Get Massachusetts price data.
#--------------------------------------------------------------------------

# Make a request to the Open Data API endpoint.
url = 'https://masscannabiscontrol.com/resource/rqtv-uenj.json'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1667.0 Safari/537.36'}
response = requests.get(url, headers=headers)

# Format the data appropriately.
ma_prices = pd.DataFrame(response.json())
ma_prices.rename(columns={'Date': 'date', '= 1 Oz': 'dollars_per_ounce'}, inplace=True)
ma_prices['date'] = pd.to_datetime(ma_prices['date'])
ma_prices.set_index(ma_prices['date'], inplace=True)

# Limit to the desired timeframe.
ma_prices = ma_prices.loc[
    (ma_prices.index >= pd.to_datetime('2018-11-01')) &
    (ma_prices.index <= pd.to_datetime('2021-11-30'))
]

# Calculate dollars per various quantities.
ma_prices['dollars_per_ounce'] = pd.to_numeric(ma_prices['dollars_per_ounce'])
ma_prices['dollars_per_gram'] = ma_prices['dollars_per_ounce'] / 28
ma_prices = end_of_period_timeseries(ma_prices)

# Plot monthly MA prices.
ma_prices['dollars_per_gram'].plot()
plt.title('Massachusetts Average Flower Price per Gram')
plt.ylabel('Dollars ($)')
plt.show()


#--------------------------------------------------------------------------
# Visualize the prices!
#--------------------------------------------------------------------------

# Plot MA and WA monthly prices together,
# annotatting and shading the period of closure.
fig, ax = plt.subplots(figsize=(15, 7))
ma_prices['dollars_per_gram'].plot(label='MA - Price per gram')
monthly_avg_flower_price.plot(label='WA - Price per item')
closure = pd.date_range(start='2020-03-26', end='2020-05-25', freq='D')
ax.fill_between(
    closure,
    10,
    16,
    facecolor='grey',
    alpha=0.5,
)
plt.annotate(
    'MA retail not permitted \n3/26/2020 to 5/25/2020.',
    (closure[-1], 15.75)
)
plt.title(
    'Average Price of Cannabis Flower in Massachusetts and Washington State',
    pad=24
)
plt.ylabel('Dollars ($)')
plt.xlabel('Month')
plt.legend(loc='best')
plt.savefig(
    f'{DATA_DIR}/figures/avg_price_ma_wa.png',
    format='png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.75,
    transparent=False,
)
plt.show()

# Optional: Forecast MA prices for March 2020 through November 2021
# using data from November 2018 to February 2020.


#--------------------------------------------------------------------------
# Create panel data.
#--------------------------------------------------------------------------

# Identify WA prices.
wa_prices = pd.DataFrame(monthly_avg_flower_price)
wa_prices.rename(columns={'price_total': 'dollars_per_gram'}, inplace=True) # Note: This is not apples to apples!!!
wa_prices['date'] = wa_prices.index

# Add a state identifier.
wa_prices['state'] = 'WA'
ma_prices['state'] = 'MA'

# Add a trend.
wa_prices['t'] = range(len(wa_prices))
ma_prices['t'] = range(len(ma_prices))

# Specify when treatment occurred (March 2020).
treatment = pd.get_dummies(ma_prices.date >= pd.to_datetime('2020-03-01'))
ma_prices['treatment'] = treatment[True]
wa_prices['treatment'] = 0

# Create the panel.
panel = pd.concat([ma_prices, wa_prices], ignore_index=True)

# Optional: Plot the timeseries with Seaborn.
# sns.lineplot(
#     x='date',
#     y='dollars_per_gram',
#     hue='state',
#     # style='treatment',
#     data=panel
# )


#--------------------------------------------------------------------------
# Estimate a difference-in-differences model (with state fixed-effects).
# Make an interpretation to answer the question:
# What effect, if any, did the policies in Massachusetts
# in March of 2020 have on prices?
#--------------------------------------------------------------------------

# Estimate the model, clustering standard errors at the group level.
formula = 'dollars_per_gram ~ t + treatment + t * treatment + C(state)'
model = sm.ols(formula, data=panel)
regression = model.fit(cov_type='cluster', cov_kwds={'groups': panel['state']})
print(regression.summary())

# Identify the parameters and trend direction.
beta_0 = regression.params['Intercept']
beta_1 = regression.params['t']
beta_2 = regression.params['treatment']
beta_3 = regression.params['t:treatment']
direction = '+' if beta_1 > 0 else '-'

# Print out the results.
print('\nBaseline price: ${:.2f}'.format(beta_0))
print('Time trend: {}${:.2f} per month'.format(direction, beta_1))
print('Difference between groups pre-intervention: ${:.2f}'.format(beta_2))
print('Difference in differences: ${:.2f}'.format(beta_3))
print("""\nConclusion: Assuming that the average price per flower item in Washington State
is an appropriate control for the average price per gram of flower in Massachusetts (MA),
then the policies adopted in MA in March 2020
caused an immediate change in prices of ${:.2f}
and had a ${:.2f} per month effect on prices since then through November 2021,
for an additional ${:.2f} effect in the average price per gram of cannabis.
""".format(beta_2, beta_3, beta_3 * sum(panel['treatment'])))
