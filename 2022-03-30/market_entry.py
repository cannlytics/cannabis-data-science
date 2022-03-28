"""
Market Entry into Cannabis Markets in Washington State
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 3/27/2022
Updated: 3/27/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script studies the choice of licensees to produce various
cannabis product types
using historic Washington State cannabis traceability data.

Data sources:

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd

    - Random Sample of WA Sales Items from 2022-03-16
    https://cannlytics.page.link/cds57

"""
# Standard imports.
from calendar import monthrange
from datetime import timedelta
from dateutil import relativedelta

# External imports.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 24})
colors = sns.color_palette('Set2', n_colors=10)

#--------------------------------------------------------------------------
# Wrangle the sample sales data.
# Random sample of sales: https://cannlytics.page.link/cds57
#--------------------------------------------------------------------------

# Read in the data from where your data lives.
DATA_DIR = '../.datasets'
DATA_FILE = f'{DATA_DIR}/random-sales-items-2022-03-16.csv'
data = pd.read_csv(DATA_FILE, low_memory=False, parse_dates=['date'], index_col=0)

# Restrict the time period.
data = data.loc[
    (data['date'] >= pd.to_datetime('2018-04-01')) &
    (data['date'] < pd.to_datetime('2021-11-01'))
]

# Remove sale outliers.
data = data.loc[data['price_total'] >= 0.25]

# Optional: Read licensees data for quick cross-referencing.
# licensees = pd.read_csv(
#     f'{DATA_DIR}/augmented-washington-state-licensees/augmented-washington-state-licensees.csv',
#     index_col=0,
#     low_memory=False,
# )


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

# Create time series panel data.
# variables = {
#     'price_total': ['mean', sum, 'count'],
#     'price_per_mg_thc': ['mean'],
#     'cannabinoid_d9_thca_percent': ['mean',],
#     'total_cannabinoid_percent': ['mean',],
#     'weight': ['mean', sum],
#     'mme_id': 'nunique',
# }
# producer_panel = data.groupby(['producer_mme_id', pd.Grouper(key='date', freq='M')]).agg(variables)
# cultivator_panel = flower_data.groupby(['producer_mme_id', pd.Grouper(key='date', freq='M')]).agg(variables)
# processor_panel = oil_data.groupby(['producer_mme_id', pd.Grouper(key='date', freq='M')]).agg(variables)
# manufacturer_panel = edible_data.groupby(['producer_mme_id', pd.Grouper(key='date', freq='M')]).agg(variables)
# beverage_producers_panel = beverage_data.groupby(['producer_mme_id', pd.Grouper(key='date', freq='M')]).agg(variables)
# preroller_panel = preroll_data.groupby(['producer_mme_id', pd.Grouper(key='date', freq='M')]).agg(variables)



#--------------------------------------------------------------------------
# Augment the data with entry and exit dates.
#--------------------------------------------------------------------------

def calculate_entry_date(series, date_field='date'):
    """Calculate the date a licensee began operating."""
    return series[date_field].min()


def calculate_exit_date(series, date_field='date'):
    """Calculate the date a licensee ceased operating."""
    return series[date_field].max()


def calculate_duration(series, first='entry_date', last='exit_date'):
    """Calculate the duration a given observation has been operating."""
    first_date = series['entry_date']
    last_date = series['exit_date']
    delta = relativedelta.relativedelta(last_date, first_date)
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
exited = sample.loc[sample['exit_date'] < pd.to_datetime('2021-10-01')]
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
T = pd.date_range(start='2018-04-30', end='2021-10-31')

# Calculate N and n_t for t = 1,..., T.
N = len(producer_data)
n_t = [
        len(producer_data.loc[
            (producer_data['entry_date'] <= t) &
            (producer_data['exit_date'] < t)
        ])
    for t in T
]

# Calculate n_m for m = 1,..., M for t = 1,..., T
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


# TODO: Count the number of entries / exits for each product type over time
# (if the licensee has never produced that product type up to time t)


# cultivator_panel = flower_data.groupby(['producer_mme_id', pd.Grouper(key='date', freq='M')]).agg(variables)
# processor_panel = oil_data.groupby(['producer_mme_id', pd.Grouper(key='date', freq='M')]).agg(variables)
# manufacturer_panel = edible_data.groupby(['producer_mme_id', pd.Grouper(key='date', freq='M')]).agg(variables)
# beverage_producers_panel = beverage_data.groupby(['producer_mme_id', pd.Grouper(key='date', freq='M')]).agg(variables)
# preroller_panel = preroll_data.groupby(['producer_mme_id', pd.Grouper(key='date', freq='M')]).agg(variables)


#--------------------------------------------------------------------------
# Build the model.
#--------------------------------------------------------------------------


# Build model assuming producers can change products every
t = [1, 6, 12, 18]

# Saturday Morning Statistics Teaser:
# Build model assuming producers can change products where
# t_i is Poisson distributed.
expected_interval = 12
draws = pd.Series(np.random.poisson(lam=expected_interval, size=N))
sns.histplot(draws, kde=True, bins=expected_interval * 2)
print('Sample mean:', draws.mean())
print('Sample variance:', draws.var())

# Optional: Assign an initial product type to each licensee.

# 1. Assign t_i to each producer, i=1,...,I

# 2. Iterate over the time horizon

    # Determine the number of licensees that can change type.

    # Estimate the number of licensees that will change type in the coming t_i
    # periods for each product.

    # Estimate the profit.

    # Make a decision for each producer that can change!

    # Calculate summarry stats for that time period!

# Model:


# Predict:
# - The number of entries for each product type
# - The quantity produced for each product type.
# - The average price for each product type.
# - The expected revenue (profits) for product type.



# TODO: Compare model predictions with empirical observations!

# - Plot predicted entries with actual entries.
# - Plot predicted exits with actual exits.
