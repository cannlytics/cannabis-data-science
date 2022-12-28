"""
Survival Analysis of Cannabis Cultivators, Processors, and Retailers in Washington
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 3/2/2022
Updated: 3/19/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script applies survival analysis to study exit from the cannabis
industry in Washington State for various business types, factors, and conditions.

Data sources:

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd

    - Random Sample of WA Sales Items from 2022-03-16
    https://cannlytics.page.link/cds57

References:

    - Generalized Linear Models: The Piece-Wise Exponential Model
    https://data.princeton.edu/wws509/notes/c7s4

    - LifeLines Packages
    https://lifelines.readthedocs.io/en/latest/Quickstart.html

    - Survival Regressions
    https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html

    - Bayesian Survival Analysis
    https://docs.pymc.io/en/v3/pymc-examples/examples/survival_analysis/survival_analysis.html

"""
# External imports.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
})


#--------------------------------------------------------------------------
# Read the sample sales data.
# Random sample of sales: https://cannlytics.page.link/cds57
#--------------------------------------------------------------------------

# Read in the data from where your data lives.
DATA_DIR = '../.datasets'
DATA_FILE = f'{DATA_DIR}/random-sales-items-2022-03-16.csv'
data = pd.read_csv(DATA_FILE, low_memory=False)

# Minor cleaning.
data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.strftime('%Y-%m')


#------------------------------------------------------------------------------
# Augment the data with duration and exit variables.
# Optional: Explore lifelines.utils to see if there are any tools to help with this.
# Optional: Rewrite `calculate_duration` and `calculate_exit` to handle groups.
#------------------------------------------------------------------------------


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


def calculate_entry_date(series, date_field='date'):
    """Calculate the date a licensee began operating."""
    return series[date_field].min()


def calculate_exit_date(series, date_field='date'):
    """Calculate the date a licensee ceased operating."""
    return series[date_field].max()


def augment_entry_exit_data(df, last_period, licensees=None, id_field='mme_id'):
    """Augment a dataset with entry_date, exit, exit_date, month_duration variables."""
    if licensees is None:
        licensees = list(data[id_field].unique())
    group = df.groupby(id_field)
    durations = [calculate_duration(df.loc[df[id_field] == x]) for x in licensees]
    exits = [calculate_exit(df.loc[df[id_field] == x], last_period) for x in licensees]
    entry_dates = calculate_entry_date(group)
    exit_dates = calculate_exit_date(group)
    augmented = pd.DataFrame({
        'mme_id': licensees,
        'duration': durations,
        'entry_date': entry_dates,
        'exit_date': exit_dates,
        'exit': exits,
    }, index=licensees)
    augmented.dropna(subset=['mme_id'], inplace=True)
    return augmented


# Restrict the time period.
data = data.loc[
    (data['date'] >= pd.to_datetime('2020-01-01')) &
    (data['date'] <= pd.to_datetime('2021-10-31'))
]

# Identify product type data.
flower_data = data.loc[data['intermediate_type'] == 'usable_marijuana']
concentrate_data = data.loc[data['intermediate_type'] == 'concentrate_for_inhalation']
edible_data = data.loc[
    (data['intermediate_type'] == 'liquid_edible') | 
    (data['intermediate_type'] == 'solid_edible')
]
preroll_data = data.loc[data['intermediate_type'] == 'infused_mix']

# Find all businesses that operated.
retailers = [i for i in list(data['mme_id'].unique()) if pd.notna(i)]
cultivators = [i for i in list(flower_data['producer_mme_id'].unique()) if pd.notna(i)]
processors = [i for i in list(concentrate_data['producer_mme_id'].unique()) if pd.notna(i)]
manufacturers = [i for i in list(edible_data['producer_mme_id'].unique()) if pd.notna(i)]
prerollers = [i for i in list(preroll_data['producer_mme_id'].unique()) if pd.notna(i)]

# Determine the number of periods a cultivator, processor, or retailer has operated.
last_period = pd.to_datetime('2021-10-01')

cultivator_data = augment_entry_exit_data(
    flower_data,
    last_period,
    licensees=cultivators,
    id_field='producer_mme_id'
)
processor_data = augment_entry_exit_data(
    concentrate_data,
    last_period,
    licensees=processors,
    id_field='producer_mme_id'
)
retailer_data = augment_entry_exit_data(
    data,
    last_period,
    licensees=retailers,
    id_field='mme_id'
)
manufacturer_data = augment_entry_exit_data(
    edible_data,
    last_period,
    licensees=manufacturers,
    id_field='producer_mme_id'
)
preroller_data = augment_entry_exit_data(
    preroll_data,
    last_period,
    licensees=prerollers,
    id_field='producer_mme_id'
)

# Assign types.
cultivator_data['type'] = 'cultivator'
processor_data['type'] = 'processor'
retailer_data['type'] = 'retailer'
manufacturer_data['type'] = 'manufacturer'
preroller_data['type'] = 'preroller'


#--------------------------------------------------------------------------
# Augment the data with county variable.
#--------------------------------------------------------------------------

def identify_county(df, id_field='mme_id', county_field='county'):
    """Identify the county of a given dataset."""
    return df.groupby(id_field)[county_field].unique().apply(lambda x: x[0])


# Add county variable.
cultivator_data['county'] = identify_county(flower_data, id_field='producer_mme_id', county_field='producer_county')
processor_data['county'] = identify_county(concentrate_data, id_field='producer_mme_id', county_field='producer_county')
retailer_data['county'] = identify_county(data)
manufacturer_data['county'] = identify_county(edible_data, id_field='producer_mme_id', county_field='producer_county')
preroller_data['county'] = identify_county(preroll_data, id_field='producer_mme_id', county_field='producer_county')


#--------------------------------------------------------------------------
# Augment the data with explanatory variables.
#--------------------------------------------------------------------------


def calculate_avg_thc(df):
    """Calculate the average total THC for a given dataset.
    Source for decarboxylated value conversion factor:
    https://www.conflabs.com/why-0-877/
    """
    thca = df['cannabinoid_d9_thca_percent'].mean()
    thc = df['cannabinoid_d9_thc_percent'].mean()
    return thc + thca.multiply(0.877)


# Calculate average total THC by licensee.
cultivator_data['avg_thc'] = calculate_avg_thc(flower_data.groupby('producer_mme_id'))
processor_data['avg_thc'] = calculate_avg_thc(concentrate_data.groupby('producer_mme_id'))
retailer_data['avg_thc'] = calculate_avg_thc(data.groupby('mme_id'))
manufacturer_data['avg_thc'] = calculate_avg_thc(edible_data.groupby('producer_mme_id'))
preroller_data['avg_thc'] = calculate_avg_thc(preroll_data.groupby('producer_mme_id'))


#--------------------------------------------------------------------------
# Augment the data with market penetration.
#--------------------------------------------------------------------------

def calculate_penetration(series, n, id_field='mme_id'):
    """Calculate retail penetration for a distributor."""
    return round(len(series[id_field].unique()) / n * 100, 2)


# Calculate market penetration.
cultivator_data['penetration'] = [
    calculate_penetration(flower_data.loc[flower_data['producer_mme_id'] == x], len(retailers))
    for x in cultivators
]
processor_data['penetration'] = [
    calculate_penetration(concentrate_data.loc[concentrate_data['producer_mme_id'] == x], len(retailers))
    for x in processors
]
retailer_data['penetration'] = [
    calculate_penetration(
        data.loc[data['mme_id'] == x],
        len(set(cultivators) | set(processors) | set(manufacturers) | set(prerollers)),
        'producer_mme_id'
    )
    for x in retailers
]
manufacturer_data['penetration'] = [
    calculate_penetration(edible_data.loc[edible_data['producer_mme_id'] == x], len(retailers))
    for x in manufacturers
]
preroller_data['penetration'] = [
    calculate_penetration(preroll_data.loc[preroll_data['producer_mme_id'] == x], len(retailers))
    for x in prerollers
]


#--------------------------------------------------------------------------
# Optional: Augment the data with Census explanatory variables.
#--------------------------------------------------------------------------

# Optional: Add other licensee specific variables:
# - median household income
# - percent of people in poverty
# - population per square mile?
# - percent of people between 18 and 65
# - percent of people over 65
# - number of veterans per 100,000
# - median gross rent
# - building permits
# - persons without health insurance (under 65)

# Optional: Estimate total grams sold by licensee.


#--------------------------------------------------------------------------
# Fit a linear regression to begin to look at the data.
#--------------------------------------------------------------------------

import statsmodels.formula.api as sm
formula = 'duration ~ avg_thc + penetration + C(county)'
model = sm.ols(formula, data=preroller_data)
print(model.fit().summary())


#--------------------------------------------------------------------------
# Calculate the Kaplan–Meier estimator.
#--------------------------------------------------------------------------

from lifelines import KaplanMeierFitter

# Fit a Kaplan-Meier estimator.
kmf = KaplanMeierFitter()
kmf.fit(retailer_data['duration'], event_observed=retailer_data['exit'])

# Visualize the Kaplan-Meier estimate of the survival function.
fig, ax = plt.subplots(figsize=(15, 8))
kmf.plot_survival_function(ax=ax, label='Retailer')
plt.title('Survival Rate of Cannabis Retailers in Washington')
plt.xlabel('Months')
plt.ylabel('Estimated Survival Rate')
plt.show()

# Optional: Explore all of the various summary statistics.


#--------------------------------------------------------------------------
# Calculate Kaplan–Meier estimator for each group.
#--------------------------------------------------------------------------

# Create panel data.
groups = pd.concat([
    cultivator_data,
    processor_data,
    retailer_data,
    manufacturer_data,
    preroller_data,
])

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
naf.fit(
    retailer_data['duration'], 
    event_observed=retailer_data['exit']
)

# Visualize the Neslon-Aalen estimate of the hazards rate.
fig, ax = plt.subplots(figsize=(15, 8))
naf.plot_hazard(bandwidth=1.0, ax=ax, label='Retailer')
plt.title('Hazards Rate of Cannabis Retailers')
plt.xlabel('Months')
plt.ylabel('Estimated Hazards Rate')
fig.savefig(
    f'{DATA_DIR}/figures/retailer-hazards-rate.png',
    format='png',
    dpi=96,
    bbox_inches='tight',
    pad_inches=0.75,
    transparent=False,
)
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

# Optional: Explore all of the various summary statistics.


#--------------------------------------------------------------------------
# Fit a Cox's proportional hazards model.
# See:
#    - Survival Regressions
#    https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html
#--------------------------------------------------------------------------

from lifelines import CoxPHFitter

datasets = [
    {'name': 'Cultivators', 'data': cultivator_data},
    {'name': 'Retailers', 'data': retailer_data},
    {'name': 'Processors', 'data': retailer_data},
    # {'name': 'Manufacturers', 'data': retailer_data},
    # {'name': '', 'data': retailer_data},
]

for dataset in datasets:

    # Identify the sample for estimation.
    sample = dataset['data'][[
        'duration',
        'exit',
        'avg_thc',
        'penetration',
        'county',
    ]]

    # Fit a hazards model.
    cph = CoxPHFitter()
    cph.fit(
        sample,
        duration_col='duration',
        event_col='exit',
        formula='penetration' # avg_thc + C(county)
    )
    cph.print_summary()

    # # Make a prediction.
    # X = pd.DataFrame([{
    #     'duration': 18,
    #     'exit': 0,
    #     'avg_thc': 20,
    #     'penetration': 1.5,
    # }])
    # cph.predict_survival_function(X).plot(figsize=(15, 8))
    # plt.title('Predicted Survival Rate Given X')
    # plt.xlabel('Months')
    # plt.ylabel('Estimated Survival Rate')
    # plt.show()
    # median = cph.predict_median(X)
    # print('Predicted median periods survived:', median)

    # # Plot the coefficients.
    # cph.plot()
    # plt.show()

    # # Plot the effect of varying a covariate (avg THC).
    # cph.plot_partial_effects_on_outcome(
    #     covariates=['avg_thc'],
    #     values=[5, 10, 15, 20, 25, 30],
    #     cmap='Greens',
    #     plot_baseline=False,
    #     figsize=(15, 8),
    # )
    # plt.title('The Effect of Avg. THC on the Survival Rate of Cultivators in Washington')
    # plt.xlabel('Months')
    # plt.ylabel('Estimated Survival Rate')
    # plt.show()

    # Plot the effect of varying a covariate (penetration).
    cph.plot_partial_effects_on_outcome(
        covariates=['penetration'],
        values=[0.1, 1, 2, 3, 5, 10, 15, 20],
        cmap='cool_r',
        plot_baseline=False,
        figsize=(15, 8)
    )
    plt.title('The Effect of Market Penetration on the Survival Rate of %s in Washington' % dataset['name'])
    plt.xlabel('Months')
    plt.ylabel('Estimated Survival Rate')
    plt.savefig(
        f'{DATA_DIR}/figures/{dataset["name"].lower()}-survival-function.svg',
        format='svg',
        dpi=96,
        bbox_inches='tight',
        pad_inches=0.75,
        transparent=False,
    )
    plt.show()


#--------------------------------------------------------------------------
# Future work: Explore if number of lab test failures affects the survival rate.
#--------------------------------------------------------------------------
