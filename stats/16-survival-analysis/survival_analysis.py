"""
Survival Analysis of Cannabis Cultivators, Processors, and Retailers in Washington
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 3/2/2022
Updated: 3/12/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script applies survival analysis to study exit from cannabis
markets for various business types, factors, and conditions.

Data sources:

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd

    - Random Sample of WA Sales Items from 2022-03-12
    https://cannlytics.page.link/1e4b8440-3c12-475a-8584-78961b3e5e48

References:

    - Generalized Linear Models: The Piece-Wise Exponential Model
    https://data.princeton.edu/wws509/notes/c7s4

    - LifeLines Packages
    https://lifelines.readthedocs.io/en/latest/Quickstart.html

    - Survival Regressions
    https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html

    - Bayesian Survival Analysis
    https://docs.pymc.io/en/v3/pymc-examples/examples/survival_analysis/survival_analysis.html

    - Censored Data Models
    https://docs.pymc.io/en/v3/pymc-examples/examples/survival_analysis/censored_data.html

"""
# External imports.
import matplotlib.pyplot as plt
import pandas as pd

# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
})


#--------------------------------------------------------------------------
# Read the sample sales data.
# Random sample of sales: https://cannlytics.page.link/1e4b8440-3c12-475a-8584-78961b3e5e48
#--------------------------------------------------------------------------

# Read in the data from where your data lives.
DATA_DIR = 'D:\\leaf-data'
DATA_FILE = f'{DATA_DIR}/samples/random-sales-items-2022-03-12.csv'
data = pd.read_csv(DATA_FILE)

# Minor cleaning.
data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.strftime('%Y-%m')


#------------------------------------------------------------------------------
# Augment the data with duration and exit variables..
# TODO: Explore lifelines.utils to see if there are any tools to help with this.
#------------------------------------------------------------------------------

# Restrict the time period.
data = data.loc[
    (data['date'] >= pd.to_datetime('2020-01-01')) &
    (data['date'] <= pd.to_datetime('2021-10-31'))
]
flower_data = data.loc[data['intermediate_type'] == 'usable_marijuana']
concentate_data = data.loc[data['intermediate_type'] == 'concentrate_for_inhalation']

# Find all businesses that operated.
retailers = list(data['mme_id'].unique())
cultivators = list(flower_data['producer_mme_id'].unique())
processors = list(concentate_data['producer_mme_id'].unique())


def calculate_duration(series, date_field='date'):
    """Calculate the duration a given observation has been operating."""
    try:
        return round((series[date_field].max() - series[date_field].min()).days / 30)
    except:
        return 0


def calculate_exit(series, end_period, date_field='date'):
    """Calculate if a given observation has exited before the end period."""
    try:
        last_obs = series[date_field].max()
        if last_obs >= end_period:
            return 0
        return 1
    except ValueError:
        return 0


# Determine the number of periods a retailer / producer has operated.
durations = []
exits = []
last_period = pd.to_datetime('2021-10-01')
for retailer in retailers:
    retail_data = data.loc[data['mme_id'] == retailer]
    durations.append(calculate_duration(retail_data))
    exits.append(calculate_exit(retail_data, last_period))
panel = pd.DataFrame({
    'mme_id': retailers,
    'duration': durations,
    'exit': exits,
    'type': 'retailer',
})

# Determine the duration and exits for cultivators.
cultivation_durations = pd.Series([
    calculate_duration(flower_data.loc[flower_data['producer_mme_id'] == x])
    for x in cultivators
])
cultivation_exits = pd.Series([
    calculate_exit(flower_data.loc[flower_data['producer_mme_id'] == x], last_period)
    for x in cultivators
])
cultivator_panel = pd.DataFrame({
    'mme_id': cultivators,
    'duration': cultivation_durations,
    'exit': cultivation_exits,
    'type': 'cultivator',
})

# Determine the duration and exits for processors.
processor_durations = pd.Series([
    calculate_duration(concentate_data.loc[concentate_data['producer_mme_id'] == x])
    for x in processors
])
processor_exits = pd.Series([
    calculate_exit(concentate_data.loc[concentate_data['producer_mme_id'] == x], last_period)
    for x in processors
])
processor_panel = pd.DataFrame({
    'mme_id': processors,
    'duration': processor_durations,
    'exit': processor_exits,
    'type': 'processor',
})


#--------------------------------------------------------------------------
# Calculate the Kaplan–Meier estimator.
#--------------------------------------------------------------------------

from lifelines import KaplanMeierFitter

# Fit a Kaplan-Meier estimator.
kmf = KaplanMeierFitter()
kmf.fit(panel['duration'], event_observed=panel['exit'])

# Visualize the Kaplan-Meier estimate of the survival function.
fig, ax = plt.subplots(figsize=(15, 8))
kmf.plot_survival_function(ax=ax, label='Retailer')
plt.title('Survival Rate of Cannabis Retailers in Washington')
plt.xlabel('Months')
plt.ylabel('Estimated Survival Rate')
plt.show()

# TODO: Explore all of the various summary statistics.


#--------------------------------------------------------------------------
# Calculate Kaplan–Meier estimator for each group.
#--------------------------------------------------------------------------

# Create panel data.
groups = pd.concat([panel, cultivator_panel, processor_panel])

# Plot the estimated survival function for each licensee type. 
fig, ax = plt.subplots(figsize=(15, 8))
kmf = KaplanMeierFitter()
for name, grouped_df in groups.groupby('type'):
    kmf.fit(grouped_df['duration'], grouped_df['exit'], label=name.title())
    kmf.plot_survival_function(ax=ax)
plt.title('Survival Function of Cannabis Licensees in Washington')
plt.xlabel('Months')
plt.ylabel('Estimated Survival Rate')
plt.show()


#--------------------------------------------------------------------------
# Calculate the Neslon--Aalen Estimator.
#--------------------------------------------------------------------------

from lifelines import NelsonAalenFitter

# Fit a Neslon--Aalen estimator.
naf = NelsonAalenFitter()
naf.fit(panel['duration'], event_observed=panel['exit'])

# Visualize the Neslon--Aalen estimate of the hazards rate.
fig, ax = plt.subplots(figsize=(15, 8))
naf.plot_hazard(bandwidth=1.0, ax=ax, label='Retailer')
plt.title('Hazards Rate of Cannabis Retailers')
plt.xlabel('Months')
plt.ylabel('Estimated Hazards Rate')
plt.show()

# Plot the estimated hazards rate for each licensee type. 
fig, ax = plt.subplots(figsize=(15, 8))
for name, grouped_df in groups.groupby('type'):
    naf.fit(grouped_df['duration'], grouped_df['exit'], label=name.title())
    naf.plot_hazard(bandwidth=1.0, ax=ax)
plt.title('Hazards Rate of Cannabis Licensees in Washington')
plt.xlabel('Months')
plt.ylabel('Estimated Survival Rate')
plt.show()

# TODO: Explore all of the various summary statistics.


#--------------------------------------------------------------------------
# For next week: Fit a Cox's proportional hazards model.
# See:
#    - Survival Regressions
#    https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html
#
#--------------------------------------------------------------------------


#--------------------------------------------------------------------------
# For next week: Fit a Bayesian Cox's proportional hazard model.
# See:
#     - Bayesian Survival Analysis
#     https://docs.pymc.io/en/v3/pymc-examples/examples/survival_analysis/survival_analysis.html
#
#     - Censored Data Models
#     https://docs.pymc.io/en/v3/pymc-examples/examples/survival_analysis/censored_data.html
#--------------------------------------------------------------------------

