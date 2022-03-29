"""
Market Entry into Cannabis Markets in Washington State
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 3/27/2022
Updated: 3/28/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script studies the choice of licensees to produce various
cannabis product types using historic Washington State cannabis traceability data.

Data sources:

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd

    - Random Sample of WA Sales Items from 2022-03-16
    https://cannlytics.page.link/cds57

"""
# Standard imports.
from calendar import monthrange
# from datetime import timedelta
import math

# External imports.
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# import numpy as np
import pandas as pd
import seaborn as sns
# import statsmodels.api as sm

# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 24})
colors = sns.color_palette('Set2', n_colors=10)
primary_colors = sns.color_palette('husl', n_colors=10)


#--------------------------------------------------------------------------
# Wrangle the sample sales data.
# Random sample of sales: https://cannlytics.page.link/cds57
#--------------------------------------------------------------------------

# Read in the data from where your data lives.
DATA_DIR = '../.datasets'
DATA_FILE = f'{DATA_DIR}/random-sales-items-2022-03-16.csv'
data = pd.read_csv(DATA_FILE, low_memory=False, parse_dates=['date'], index_col=0)

# Restrict the time period.
first_period = pd.to_datetime('2018-04-01')
last_period = pd.to_datetime('2021-11-01')
data = data.loc[(data['date'] >= first_period) & (data['date'] < last_period)]

# Remove sale outliers.
data = data.loc[data['price_total'] >= 0.25]


#--------------------------------------------------------------------------
# Organize the data.
#--------------------------------------------------------------------------

# Identify product type data.
flower_data = data.loc[data['intermediate_type'] == 'usable_marijuana']
oil_data = data.loc[data['intermediate_type'] == 'concentrate_for_inhalation']
edible_data = data.loc[data['intermediate_type'] == 'solid_edible']
beverage_data = data.loc[(data['intermediate_type'] == 'liquid_edible')]
preroll_data = data.loc[data['intermediate_type'] == 'infused_mix']


def identify_unique(series):
    """Return a list of all unique values, excluding NA, from a Pandas series."""
    return [i for i in list(series.unique()) if pd.notna(i)]


# Find all businesses that operated.
retailers = identify_unique(data['mme_id'])
producers = identify_unique(data['producer_mme_id'])

# Find business that produce each product.
flower_producers = identify_unique(flower_data['producer_mme_id'])
oil_producers = identify_unique(oil_data['producer_mme_id'])
edible_producers = identify_unique(edible_data['producer_mme_id'])
beverage_producers = identify_unique(beverage_data['producer_mme_id'])
preroll_producers = identify_unique(preroll_data['producer_mme_id'])


#--------------------------------------------------------------------------
# ***Saturday Morning Statistics Teaser***
# Augment the data with entry and exit dates.
#--------------------------------------------------------------------------

def calculate_entry_date(series, field='date'):
    """Calculate the date a licensee began operating."""
    return series[field].min()


def calculate_exit_date(series, field='date'):
    """Calculate the date a licensee ceased operating."""
    return series[field].max()


def calculate_duration(series, first='entry_date', last='exit_date'):
    """Calculate the duration a given observation has been operating."""
    first_date = series[first]
    last_date = series[last]
    delta = relativedelta(last_date, first_date)
    month_days = monthrange(last_date.year, last_date.month)[1]
    return delta.years * 12 + delta.months + (delta.days / month_days)


# Determine historic entries and exits.
entry_dates = data.groupby('producer_mme_id')['date'].min()
exit_dates = data.groupby('producer_mme_id')['date'].max()
producer_data = pd.DataFrame({
    'entry_date': entry_dates,
    'exit_date': exit_dates,
})
durations = producer_data.apply(calculate_duration, axis=1)
producer_data = producer_data.assign(duration=durations)

# Visualize when licensees operated.
fig, ax = plt.subplots(figsize=(7.5, 8))
sample = producer_data.sample(20, random_state=1)
sample.sort_values('duration', ascending=False, inplace=True)
exited = sample.loc[sample['exit_date'] < last_period - relativedelta(months=1)]
ax.hlines(
    sample.index,
    0,
    sample['duration'],
    color=colors[0],
)
ax.scatter(
    exited['duration'],
    exited.index,
    zorder=10,
    label='Exited',
    marker='x',
    color='red'
)
ax.set_yticks([])
ax.set_ylim(-0.25, len(sample) + 0.25)
ax.set_xlim(left=0)
ax.legend(loc='upper right', fontsize=18)
ax.set_xlabel('Months of Operations', fontsize=21)
ax.set_ylabel('Licensee', fontsize=21)
plt.title(
    'Operation Duration of a Sample of \nLicensees in Washington State',
    fontsize=21,
    pad=20,
)
plt.show()


#--------------------------------------------------------------------------
# Curate variables of interest: producers, products, and time.
#--------------------------------------------------------------------------

# Define time horizon.
T = pd.date_range(start=first_period, end=last_period, freq='M')

# Calculate N and n_t for t = 1,..., T.
N = len(producer_data)
N_t = pd.DataFrame([
        len(producer_data.loc[
            (producer_data['entry_date'] <= t) &
            (producer_data['exit_date'] < t)
        ])
    for t in T
])

# Calculate n_m for m = 1,..., M for t = 1,..., T
products = ['flower', 'oil', 'edibles', 'beverages', 'prerolls']
producer_data = producer_data.assign(
    flower=0,
    oil=0,
    edibles=0,
    beverages=0,
    prerolls=0,
)
producer_data.loc[producer_data.index.isin(flower_producers), 'flower'] = 1
producer_data.loc[producer_data.index.isin(oil_producers), 'oil'] = 1
producer_data.loc[producer_data.index.isin(edible_producers), 'edibles'] = 1
producer_data.loc[producer_data.index.isin(beverage_producers), 'beverages'] = 1
producer_data.loc[producer_data.index.isin(preroll_producers), 'prerolls'] = 1

N_m = pd.DataFrame([
        producer_data.loc[
            (producer_data['entry_date'] <= t) &
            (producer_data['exit_date'] < t)
        ][products].sum()
    for t in T
])


#--------------------------------------------------------------------------
# Calculate variables:
# Count the number of entries / exits for each product type over time
# Optional: Only count if the licensee has never produced that product type
# up to time t.
#--------------------------------------------------------------------------

# Simplify the code with a dictionary.
product_dict = {
    'flower': 'usable_marijuana',
    'oil': 'concentrate_for_inhalation',
    'edibles': 'solid_edible',
    'beverages': 'liquid_edible',
    'prerolls': 'infused_mix',
}

# Define entries and exits at time 0.
entries = {'total': [0]}
exits = {'total': [0]}
for product, _ in product_dict.items():
    entries[product] = [0]
    exits[product] = [0]

# 1. Iterate over time.
for t in T[1:]:

    # 2. Find the producers who exited or entered each product type.
    dates = data['date']
    t_minus_2 = t - relativedelta(months=2)
    t_minus_1 = t - relativedelta(months=1)
    data_t_minus_1 = data.loc[(dates <= t_minus_1) & (dates > t_minus_2)]
    data_t = data.loc[(dates <= t) & (dates > t_minus_1)]
    producers_t_minus_1 = identify_unique(data_t_minus_1['producer_mme_id'])
    producers_t = identify_unique(data_t['producer_mme_id'])
    entries_t = list(set(producers_t) - set(producers_t_minus_1))
    exits_t = list(set(producers_t_minus_1) - set(producers_t))
    entries['total'].append(len(entries_t))
    exits['total'].append(len(exits_t))

    # 3. Find the producers who exited or entered by product type.
    for product, field in product_dict.items():
        type_data_t_minus_1 = data_t_minus_1.loc[data_t_minus_1['intermediate_type'] == field]
        type_data_t = data_t.loc[data_t['intermediate_type'] == field]
        type_producers_t_minus_1 = identify_unique(type_data_t_minus_1['producer_mme_id'])
        type_producers_t = identify_unique(type_data_t['producer_mme_id'])
        type_entries_t = list(set(type_producers_t) - set(type_producers_t_minus_1))
        type_exits_t = list(set(type_producers_t_minus_1) - set(type_producers_t))
        entries[product].append(len(type_entries_t))
        exits[product].append(len(type_exits_t))

# Create timeseries.
timeseries = {}
for key, type_entries in entries.items():
    type_exits = exits[key]
    timeseries[f'{key}_entries'] = type_entries
    timeseries[f'{key}_exits'] = type_exits
timeseries = pd.DataFrame(timeseries)
timeseries.set_index(T, inplace=True)


#--------------------------------------------------------------------------
# Visualize the newly calculated entry and exit data.
#--------------------------------------------------------------------------

# Visualize entries and exits over time.
fig, axs = plt.subplots(3, 2, figsize=(12, 9))
fields = ['total'] + list(product_dict.keys())
for m, product in enumerate(fields):

    # Identify the subplot.
    column = 0 if m % 2 == 0 else 1
    row = math.floor(m / 2)
    ax = axs[row, column]

    # Plot the series.
    ax.plot(
        timeseries[1:].index,
        timeseries[1:][f'{product}_exits'],
        color=primary_colors[0],
    )
    ax.plot(
        timeseries[1:].index,
        timeseries[1:][f'{product}_entries'],
        color=colors[0],
    )

    # Format the subplot.
    ax.set_xticks(ax.get_xticks()[::3])
    ax.tick_params(axis='both', labelsize=21)
    ax.set_title(product.title(), fontsize=21)
    ax.set_ylim(-5, 100)

# Add y-axis label.
for ax in axs.flat:
    ax.set(ylabel='Count')
    ax.yaxis.label.set_size(21)

# Hide x-axis labels for all except bottom subplots.
for ax in axs.flat:
    ax.label_outer()

fig.suptitle('Monthly Entries and Exits by Cannabis Product in Washington State')
custom_lines = [
    Line2D([0], [0], color=colors[0], lw=4),
    Line2D([0], [0], color=primary_colors[0], lw=4)
]
ax.legend(
    custom_lines,
    ['Entries', 'Exits'],
    loc='lower left',
    bbox_to_anchor=(0.5, -0.75)
)
plt.show()


#--------------------------------------------------------------------------
# TODO: Build the model: Create a congestion game.
#--------------------------------------------------------------------------

# # 1. Assign t_i to each producer, i=1,...,I

# # 1.a. Build model assuming producers can change products every
# t_i =  pd.Series(np.random.uniform(1, 24, size=N))
# sns.histplot(t_i, kde=True, bins=24)

# # ***Saturday Morning Statistics Teaser***
# # 1.b. Build model assuming producers can change products where
# # t_i is Poisson distributed.
# expected_interval = 12
# t_i = pd.Series(np.random.poisson(lam=expected_interval, size=N))
# sns.histplot(t_i, kde=True, bins=expected_interval * 2)
# print('Sample mean:', t_i.mean())
# print('Sample variance:', t_i.var())

# # Optional: Assign an initial product type to each licensee.

# # 2. Iterate over the time horizon, t=0,...,T
# for t in range(len(T)):

#     # Determine the number of licensees that can change type.
#     choosers = t % t_i == 0

#     # Estimate the number of licensees that will change type in the coming t_i
#     # periods for each product.

#     # Estimate the profit.
#     profits = []
#     n_m = N_m.iloc[t]
#     for product in products:

#         n_m_t = n_m[product]
#         print('Time:', t, 'Product:' , product, 'n_m:', n_m_t)

#         # TODO: Calculate average sales for product m at time t.


#         # Advanced: Estimate expected profits over interval.
#         # for h in range(len(t_i)):
#         #     # Estimate the number of producers of each type for the next t_h periods.

#         # TODO: Forecast average sales by
#         # forecasting number of entries and exits and n_m in each future period
#         # and forecast the total revenue.

#     # Make a decision for each producer that can change!
#     decision = max(profits)

#     # Calculate summary stats for that time period!
#     # Number of exits from each product type.
#     # Number of entries for each product type.


#--------------------------------------------------------------------------
# TODO: Predict:
# - The number of entries for each product type
# - The quantity produced for each product type.
# - The average price for each product type.
# - The expected revenue (profits) for product type.
#--------------------------------------------------------------------------


#--------------------------------------------------------------------------
# TODO: Compare model predictions with empirical observations!
#--------------------------------------------------------------------------

# - Plot predicted entries with actual entries.
# - Plot predicted exits with actual exits.
# - Optional: Compare predicted with actual quantity, price, and revenue.
