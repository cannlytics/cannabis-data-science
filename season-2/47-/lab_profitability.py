"""
Predicting Lab Profitability in Washington State
Cannabis Data Science Meetup Group
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 1/10/2022
Updated: 1/12/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: Using data on analyses performed by labs in Washington State,
this script calculates historic performance of each lab and uses analysis
prices to forecast the profitability of each lab over the next 5 years.

Data Sources:

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd?page=1

Resources:

    - Pandas time series / date functionality
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html    

"""
# Standard imports.
import gc
import json
import re
import requests

# External imports.
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import pmdarima as pm
import seaborn as sns
import statsmodels.api as sm

# Internal imports.
from utils import (
    forecast_arima,
    format_millions,
    format_thousands,
)

#------------------------------------------------------------------------------
# Perform housekeeping and define useful functions.
#------------------------------------------------------------------------------

# Define format for all plots.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
})

# Print floats to 2 decimal places.
pd.options.display.float_format = "{:.0f}".format


def sorted_nicely(unsorted_list): 
    """Sort the given iterable in the way that humans expect.
    Credit: Mark Byers <https://stackoverflow.com/a/2669120/5021266>
    License: CC BY-SA 2.5 <https://creativecommons.org/licenses/by-sa/2.5/>
    """ 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(unsorted_list, key = alphanum_key)


#------------------------------------------------------------------------------
# Read in and clean the laboratory data.
#------------------------------------------------------------------------------

# Define lab datasets.
lab_datasets = ['LabResults_0', 'LabResults_1', 'LabResults_2']

# Specify the column types to read.
column_types = {
    'global_id' : 'string',
    # 'mme_id' : 'category',
    # 'type' : 'category',
    # 'intermediate_type' : 'category',
    # 'status' : 'category',
    #'user_id' : 'string',
    #'external_id' : 'string',
    #'inventory_id' : 'string',
    #'testing_status' : 'category',
    #'batch_id' : 'string',
    #'parent_lab_result_id' : 'string',
    #'og_parent_lab_result_id' : 'string',
    #'copied_from_lab_id' : 'string',
    #'lab_user_id' : 'string',
    #'foreign_matter' : 'bool',
    #'moisture_content_percent' : 'float16',
    #'growth_regulators_ppm' : 'float16',
    #'cannabinoid_status' : 'category',
    #'cannabinoid_editor' : 'float32',
    #'cannabinoid_d9_thca_percent': 'float16',
    #'cannabinoid_d9_thca_mg_g' : 'float16',
    #'cannabinoid_d9_thc_percent' : 'float16',
    #'cannabinoid_d9_thc_mg_g' : 'float16',
    #'cannabinoid_d8_thc_percent' : 'float16',
    #'cannabinoid_d8_thc_mg_g' : 'float16',
    #'cannabinoid_cbd_percent' : 'float16',
    #'cannabinoid_cbd_mg_g' : 'float16',
    #'cannabinoid_cbda_percent' : 'float16',
    #'cannabinoid_cbda_mg_g' : 'float16',
    #'cannabinoid_cbdv_percent' : 'float16',
    #'cannabinoid_cbg_percent' : 'float16',
    #'cannabinoid_cbg_mg_g' : 'float16',
    #'terpenoid_pinene_percent' : 'float16',
    #'terpenoid_pinene_mg_g' : 'float16',
    #'microbial_status' : 'category',
    #'microbial_editor' : 'string',
    #'microbial_bile_tolerant_cfu_g' : 'float16',
    #'microbial_pathogenic_e_coli_cfu_g' : 'float16',
    #'microbial_salmonella_cfu_g' : 'float16',
    #'mycotoxin_status' : 'category',
    #'mycotoxin_editor' : 'string',
    #'mycotoxin_aflatoxins_ppb' : 'float16',
    #'mycotoxin_ochratoxin_ppb' : 'float16',
    #'metal_status' : 'category',
    #'metal_editor': 'string',
    #'metal_arsenic_ppm' : 'float16',
    #'metal_cadmium_ppm' : 'float16',
    #'metal_lead_ppm' : 'float16',
    #'metal_mercury_ppm' : 'float16',
    #'pesticide_status' : 'category',
    #'pesticide_editor' : 'string',
    #'pesticide_abamectin_ppm' : 'float16',
    #'pesticide_acequinocyl_ppm' : 'float16',
    #'pesticide_bifenazate_ppm' : 'float16',
    #'pesticide_cyfluthrin_ppm' : 'float16',
    #'pesticide_cypermethrin_ppm' : 'float16',
    #'pesticide_etoxazole_ppm' : 'float16',
    #'pesticide_flonicamid_ppm' : 'float',
    #'pesticide_fludioxonil_ppm' : 'float16',
    #'pesticide_imidacloprid_ppm' : 'float16',
    #'pesticide_myclobutanil_ppm' : 'float16',
    #'pesticide_spinosad_ppm' : 'float16',
    #'pesticide_spirotetramet_ppm' : 'float16',
    #'pesticide_thiamethoxam_ppm' : 'float16',
    #'pesticide_trifloxystrobin_ppm' : 'float16',
    #'solvent_status' : 'category',
    #'solvent_editor' : 'string',
    #'solvent_butanes_ppm' : 'float16',
    #'solvent_heptane_ppm' : 'float16',
    #'solvent_propane_ppm' : 'float16',
    #'notes' : 'float32',
    #'thc_percent' : 'float16',
    #'moisture_content_water_activity_rate' : 'float16',
    #'solvent_acetone_ppm' : 'float16',
    #'solvent_benzene_ppm' : 'float16',
    #'solvent_cyclohexane_ppm' : 'float16',
    #'solvent_chloroform_ppm' : 'float16',
    #'solvent_dichloromethane_ppm' : 'float16',
    #'solvent_ethyl_acetate_ppm' : 'float16',
    #'solvent_hexanes_ppm' : 'float16',
    #'solvent_isopropanol_ppm' : 'float16',
    #'solvent_methanol_ppm' : 'float16',
    #'solvent_pentanes_ppm' : 'float16',
    #'solvent_toluene_ppm' : 'float16',
    #'solvent_xylene_ppm' : 'float16',
    #'pesticide_acephate_ppm' : 'float16',
    #'pesticide_acetamiprid_ppm' : 'float16',
    #'pesticide_aldicarb_ppm' : 'float16',
    #'pesticide_azoxystrobin_ppm' : 'float16',
    #'pesticide_bifenthrin_ppm' : 'float16',
    #'pesticide_boscalid_ppm' : 'float16',
    #'pesticide_carbaryl_ppm' : 'float16',
    #'pesticide_carbofuran_ppm' : 'float16',
    #'pesticide_chlorantraniliprole_ppm' : 'float16'
}

# Specify the date columns.
date_columns = ['created_at']

# Specify all of the columns.
columns = list(column_types.keys()) + date_columns

# Read in the lab result data.
shards = []
for dataset in lab_datasets:
    lab_data = pd.read_csv(
        f'../.datasets/{dataset}.csv',
        sep='\t',
        encoding='utf-16',
        usecols=columns,
        dtype=column_types,
        parse_dates=date_columns,
        # nrows=10000,
        # skipinitialspace=True,
    )
    shards.append(lab_data)

# Aggregate lab data, remove shards to free up memory.
data = pd.concat(shards)
del shards
del lab_data
gc.collect()

# Beginning cleaning the lab data.
data.dropna(subset=['global_id'], inplace=True)
data.index = data['global_id']
data = data.sort_index()

# Define lab ID for each observation.
data['lab_id'] = data['global_id'].map(lambda x: x[x.find('WAL'):x.find('.')])

# Remove attested lab results.
data = data.loc[data.lab_id != '']

# Identify all of the labs.
lab_ids = list(data['lab_id'].unique())

# Sort the alphanumeric lab IDs.
lab_ids = sorted_nicely(lab_ids)

#------------------------------------------------------------------------------
# Read in and clean the licensee data.
#------------------------------------------------------------------------------

# Specify the licensee fields
licensee_column_types = {
    'global_id' : 'string',
    'name': 'string',
    'city': 'string',
    'type': 'string',
    'code': 'string',
}

# Read in the licensee data.
file_name = '../.datasets/Licensees_0.csv'
licensees = pd.read_csv(
    file_name,
    sep='\t',
    encoding='utf-16',
    usecols=list(licensee_column_types.keys()),
    dtype=licensee_column_types,
)


#------------------------------------------------------------------------------
# Create day, month, year variables.
#------------------------------------------------------------------------------

def format_end_of_month(row):
    """Format a row with a 'date' column as an ISO formatted month."""
    month = row['date'].month
    if month < 10:
        month = f'0{month}'
    year = row['date'].year
    day =  row['date'] + MonthEnd(0)
    return f'{year}-{month}-{day.day}'


def format_end_of_year(row):
    """Format a row with a 'date' column as an ISO formatted year."""
    year = row['date'].year
    return f'{year}-12-31'


# Add a time column.
data['date'] = pd.to_datetime(data['created_at'])

# Assign day, month, year variables.
data = data.assign(
    day=data['date'].dt.date,
    month=data.apply(lambda row: format_end_of_month(row), axis=1),
    year=data.apply(lambda row: format_end_of_year(row), axis=1),
)


#------------------------------------------------------------------------------
# Calculate interesting lab summary statistics.
#------------------------------------------------------------------------------

# Identify the number of samples tested by each lab.
stats = {}
total_tests = 0
for lab_id in lab_ids:
    lab_samples = data.loc[data['lab_id'] == lab_id]
    tested_samples = len(lab_samples)
    if tested_samples > 0:
        code = lab_id.replace('WA', '')
        lab_data = licensees.loc[licensees['code'] == code].iloc[0]
        stats[lab_id] = {
            'name': lab_data['name'],
            'city': lab_data['city'],
            'total_samples': tested_samples,
        }
        total_tests += tested_samples

# Calculate the market share for each lab.
lab_stats = pd.DataFrame.from_dict(stats, orient='index')
lab_stats['market_share'] = lab_stats['total_samples'] / total_tests * 100

# Print lab statistics.
statistics = ['name', 'total_samples', 'market_share', 'city']
print(lab_stats[statistics])

# Print by market share.
print(lab_stats.sort_values(by='market_share', ascending=False)[statistics])

#------------------------------------------------------------------------------
# How many analyses are being conducted by each lab on a day-to-day,
# month-to-month, and year-to-year basis?
#------------------------------------------------------------------------------

def plot_samples_by_period(data, column, thousands=False):
    """Plot samples for each lab by a given period."""
    lab_ids = sorted_nicely(list(data['lab_id'].unique()))
    colors = sns.color_palette('tab20', n_colors=len(lab_ids))
    fig, ax = plt.subplots(figsize=(14, 6))
    for count, lab_id in enumerate(lab_ids):
        lab_samples = data.loc[data['lab_id'] == lab_id]
        timeseries = lab_samples.groupby(
            column,
            as_index=False
        ).size()
        timeseries['date'] = pd.to_datetime(timeseries[column])
        timeseries.set_index('date', inplace=True)
        plt.plot(
                timeseries.index,
                timeseries['size'],
                label=lab_id,
                color=colors[count],
                alpha=0.6,
            )
    plt.ylim(0)
    plt.setp(ax.get_yticklabels()[0], visible=False)
    if thousands:
        ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
    plt.title(f'Samples Tested per {column} by Labs in Washington'.title())
    plt.legend(
        ncol=5,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
    )
    plt.savefig(f'figures/samples_tested_per_{column}_wa.png', dpi=300,
                bbox_inches='tight', pad_inches=0.75, transparent=False)
    plt.show()


# Plot daily samples tested by each lab.
plot_samples_by_period(data, 'day')

# Plot monthly samples tested by each lab.
plot_samples_by_period(data, 'month')

# Count yearly samples tested by each lab.
plot_samples_by_period(data, 'year', thousands=True)


#------------------------------------------------------------------------------
# Bonus: Calculate even more lab statistics.
#------------------------------------------------------------------------------

# What is the break down of analyses by sample type? By lab?


# What is the overall failure rate? By lab?


# What is the failure rate by analysis? By lab?


# What is the failure rate day-to-day, month-to-month, and year-to-year? By lab?


#------------------------------------------------------------------------------
# Forecast samples tested by lab.
# How many samples will each lab test in 2022-2026?
#------------------------------------------------------------------------------

# Define forecast horizon and forecast fix effects.
forecast_horizon = pd.date_range(
    start=pd.to_datetime('2021-11-01'),
    end=pd.to_datetime('2027-01-01'),
    freq='M',
)
forecast_month_effects = pd.get_dummies(forecast_horizon.month)

# Create a forecast of samples tested by lab.
forecasts = {}
for lab_id in lab_ids:
    
    # Define the training data.
    training_data = data.loc[
        (data['lab_id'] == lab_id) &
        (data['date'] >= pd.to_datetime('2020-05-31')) &
        (data['date'] <= pd.to_datetime('2021-10-31'))
    ]
    
    # Create the timeseries.
    timeseries = training_data.groupby('month', as_index=False).size()
    timeseries['date'] = pd.to_datetime(timeseries['month'])
    timeseries.set_index('date', inplace=True)

    # Create month fixed effects.
    month_effects = pd.get_dummies(timeseries.index.month)
    month_effects.index = timeseries.index

    # Build a forecasting model.
    try:
        model = pm.auto_arima(
            timeseries['size'],
            X=month_effects,
            start_p=0,
            d=0,
            start_q=0,
            max_p=6,
            max_d=6,
            max_q=6,
            seasonal=True,
            start_P=0,
            D=0,
            start_Q=0,
            max_P=6,
            max_D=6,
            max_Q=6,
            information_criterion='bic',
            alpha=0.2,
        )

        # Forecast.
        forecast, confidence_interval = forecast_arima(
            model,
            forecast_horizon,
            X=forecast_month_effects
        )
        forecasts[lab_id] = forecast
    except:
        pass


# What is the forecast of samples tested by lab on a month-to-month basis?
colors = sns.color_palette('tab20', n_colors=len(lab_ids))
fig, ax = plt.subplots(figsize=(14, 6))
for count, lab_id in enumerate(lab_ids):
    try:
        sample_forecast = forecasts[lab_id]
        sample_forecast.plot(
            label=lab_id,
            color=colors[count],
            linestyle='--'
        )
    except KeyError:
        pass
plt.ylim(0)
plt.setp(ax.get_yticklabels()[0], visible=False)
plt.title('Forecast of Monthly Samples Tested by Labs in Washington')
plt.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.05),)
plt.savefig('figures/samples_tested_per_month_forecast_wa.png', dpi=300,
            bbox_inches='tight', pad_inches=0.75, transparent=False)
plt.show()

# What is the forecast of samples tested by lab on a year-to-year basis?
fig, ax = plt.subplots(figsize=(14, 6))
for count, lab_id in enumerate(lab_ids):
    try:
        sample_forecast = forecasts[lab_id]
        timeseries = sample_forecast.resample('Y').sum()
        timeseries = timeseries.loc[timeseries.index >= pd.to_datetime('2022-01-01')]
        timeseries.plot(
            label=lab_id,
            color=colors[count],
            linestyle='--'
        )
    except KeyError:
        pass
plt.ylim(0)
ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
plt.setp(ax.get_yticklabels()[0], visible=False)
plt.title('Forecast of Yearly Samples Tested by Labs in Washington')
plt.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.05),)
plt.savefig('figures/samples_tested_per_year_forecast_wa.png', dpi=300,
            bbox_inches='tight', pad_inches=0.75, transparent=False)
plt.show()


#------------------------------------------------------------------------------
# What are the prices of analyses in Washington state?
#------------------------------------------------------------------------------

# Get supplementary analysis price data from the Cannlytics API.
labs_url = 'https://cannlytics.com/api/labs'
response = requests.get(labs_url, params={'state': 'WA'})
labs = response.json()['data']

# Convert lab data to a DataFrame.
labs_data = pd.DataFrame(labs)
labs_data.index = labs_data['license']

# Print lab prices
labs_data.sort_values(by='panel_price', ascending=False, inplace=True)
print(labs_data['panel_price'])


#------------------------------------------------------------------------------
# Forecast lab revenue.
# What is the expected revenue per lab?
#------------------------------------------------------------------------------

# What is the forecast of revenue by lab on a month-to-month basis?
colors = sns.color_palette('tab20', n_colors=len(lab_ids))
fig, ax = plt.subplots(figsize=(14, 6))
for count, lab_id in enumerate(lab_ids):
    try:
        sample_forecast = forecasts[lab_id]
        code = lab_id.replace('WA', '')
        price = labs_data.loc[code]['panel_price']
        revenue = sample_forecast * price
        revenue.plot(
            label=lab_id,
            color=colors[count],
            linestyle='--'
        )
    except KeyError:
        pass
plt.ylim(0)
plt.setp(ax.get_yticklabels()[0], visible=False)
ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
plt.title('Forecast of Monthly Revenue by Labs in Washington')
plt.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.05),)
plt.savefig('figures/monthly_revenue_per_lab_forecast_wa.png', dpi=300,
            bbox_inches='tight', pad_inches=0.75, transparent=False)
plt.show()

# What is the forecast of revenue by lab on a year-to-year basis?
fig, ax = plt.subplots(figsize=(14, 6))
for count, lab_id in enumerate(lab_ids):
    try:
        sample_forecast = forecasts[lab_id]
        timeseries = sample_forecast.resample('Y').sum()
        timeseries = timeseries.loc[
            timeseries.index >= pd.to_datetime('2022-01-01')
        ]
        code = lab_id.replace('WA', '')
        price = labs_data.loc[code]['panel_price']
        revenue = timeseries * price
        revenue.plot(
            label=lab_id,
            color=colors[count],
            linestyle='--'
        )
    except KeyError:
        pass
plt.ylim(0)
ax.yaxis.set_major_formatter(FuncFormatter(format_millions))
plt.setp(ax.get_yticklabels()[0], visible=False)
plt.title('Forecast of Annual Revenue by Labs in Washington')
plt.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.05),)
plt.savefig('figures/annual_revenue_per_lab_forecast_wa.png', dpi=300,
            bbox_inches='tight', pad_inches=0.75, transparent=False)
plt.show()

# What is a 5 year projection of revenue by lab?
# five_year_revenue_forecast = {}
# five_year_total_samples = {}
five_year_forecasts = {}
for count, lab_id in enumerate(lab_ids):
    try:
        sample_forecast = forecasts[lab_id]
        timeseries = sample_forecast.resample('Y').sum()
        timeseries = timeseries.loc[
            timeseries.index >= pd.to_datetime('2022-01-01')
        ]
        code = lab_id.replace('WA', '')
        price = labs_data.loc[code]['panel_price']
        revenue = timeseries * price
        total_revenue = revenue.sum()
        total_samples = timeseries.sum()
        five_year_forecasts[lab_id] = {
            'revenue': revenue.sum(),
            'formatted_revenue': format_millions(total_revenue),
            'total_samples': total_samples,
            'formatted_total_samples': format_thousands(total_samples),
            'price': price,
        }
    except KeyError:
        pass

# Print 5 year forecasted revenue for each lab.
five_year_forecast_data = pd.DataFrame.from_dict(five_year_forecasts, orient='index')
five_year_forecast_data.sort_values(by='revenue', ascending=False, inplace=True)
print('5 Year Projected Revenue')
print(five_year_forecast_data[['formatted_revenue', 'formatted_total_samples', 'price']])

#------------------------------------------------------------------------------
# Future work: What are the prices of required instruments?
#------------------------------------------------------------------------------

# # What are the costs of instruments needed for each analysis?
# with open('data/instrument_data.json') as file:
#     instrument_data = json.load(file)

# # What is the total capital cost to open a lab?
# total_fixed_costs = 0
# for instrument in instrument_data:
#     total_fixed_costs += instrument['price']

# formatted_cost = format_millions(total_fixed_costs)
# print('Total cost to open a lab ≈  ${}'.format(formatted_cost))


#------------------------------------------------------------------------------
# Forecast lab profit.
# What is the maximum average variable cost per sample to make a profit?
#------------------------------------------------------------------------------

# Specify total fixed costs
total_fixed_costs = 5_000_000

# Create a forecast of profits over 5 years for each lab assuming that each lab
# needs to buy all brand new instruments and that each instrument has a life
# span of 5 years with straight-line depreciation.
five_year_forecast_data['max_variable_cost_per_sample'] = \
    (five_year_forecast_data['revenue'] - total_fixed_costs) \
    / five_year_forecast_data['total_samples']


# What is the maximum variable cost that a lab will need to stay under to make a
# profit? (On a day-to-day, month-to-month, and year-to-year basis).
print(five_year_forecast_data[['formatted_revenue', 'total_samples',
                               'price', 'max_variable_cost_per_sample']])


#------------------------------------------------------------------------------
# Optional: Quick analysis of forecast revenue and price.
#------------------------------------------------------------------------------

# Only look at labs with projected revenue.
revenue_data = five_year_forecast_data.loc[
    five_year_forecast_data['revenue'] > 0
]

# Plot regression of revenue on price.
fig, ax = plt.subplots(figsize=(14, 6))
ax = sns.regplot(y='revenue', x='price', data=revenue_data, scatter_kws={'s':220})
plt.ylim(0)
ax.yaxis.set_major_formatter(FuncFormatter(format_millions))
plt.setp(ax.get_yticklabels()[0], visible=False)
plt.title('Forecast of 5 Year Revenue to Price for Labs in Washington')
plt.savefig('figures/regression_revenue_price_wa.png', dpi=300,
            bbox_inches='tight', pad_inches=0.75, transparent=False)
plt.show()

# Plot regression of revenue on max variable cost.
fig, ax = plt.subplots(figsize=(14, 6))
ax = sns.regplot(y='revenue', x='max_variable_cost_per_sample', data=revenue_data, scatter_kws={'s':220})
plt.ylim(0)
plt.xlim(0)
ax.yaxis.set_major_formatter(FuncFormatter(format_millions))
plt.setp(ax.get_yticklabels()[0], visible=False)
plt.title('Forecast of 5 Year Revenue to VC for Labs in Washington')
plt.savefig('figures/regression_revenue_price_wa.png', dpi=300,
            bbox_inches='tight', pad_inches=0.75, transparent=False)
plt.show()

# Plot regression of total samples on max variable cost.
fig, ax = plt.subplots(figsize=(14, 6))
ax = sns.regplot(y='total_samples', x='max_variable_cost_per_sample', data=revenue_data, scatter_kws={'s':220})
plt.ylim(0)
plt.xlim(0)
ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
plt.setp(ax.get_yticklabels()[0], visible=False)
plt.title('Forecast of Total Samples to VC for Labs in Washington')
plt.savefig('figures/regression_revenue_price_wa.png', dpi=300,
            bbox_inches='tight', pad_inches=0.75, transparent=False)
plt.show()

# Plot regression of max variable cost on total samples.
fig, ax = plt.subplots(figsize=(14, 6))
ax = sns.regplot(y='max_variable_cost_per_sample', x='total_samples', data=revenue_data, scatter_kws={'s':220})
plt.ylim(0)
plt.xlim(0)
ax.xaxis.set_major_formatter(FuncFormatter(format_thousands))
plt.setp(ax.get_yticklabels()[0], visible=False)
plt.title('Forecast of Total Samples to VC for Labs in Washington')
plt.savefig('figures/regression_revenue_price_wa.png', dpi=300,
            bbox_inches='tight', pad_inches=0.75, transparent=False)
plt.show()
