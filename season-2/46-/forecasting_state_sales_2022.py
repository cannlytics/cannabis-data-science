"""
Predicting Cannabis Sales with Panel Data
Cannabis Data Science Meetup Group
Copyright (c) 2021-2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 12/15/2021
Updated: 1/5/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script uses a structural regression model to
forecast cannabis sales for each state in 2022.

Note: This is the first draft of this forecasting script and it
likely needs a major refactor. Please feel free to improve upon
this script and make a pull request for your changes:
    
    https://github.com/cannlytics/cannabis-data-science 

See https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
"""
# Standard imports.
import json
from typing import Any, List, Optional

# External imports.
from dateutil import relativedelta
from dotenv import dotenv_values
from fredapi import Fred
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

# Internal imports.
from utils import (
    format_billions,
    format_millions,
)

#------------------------------------------------------------------------------
# Define some useful functions.
#------------------------------------------------------------------------------

def get_state_population(
        api_key: str,
        state: str,
        obs_start: Optional[Any] = None,
        obs_end: Optional[Any] = None,
) -> List[int]:
    """Get a given state's population from the Fed Fred API."""
    fred = Fred(api_key=api_key)
    population = fred.get_series(f'{state}POP', obs_start, obs_end)
    return [int(x * 1000) for x in population.values]


def months_elapsed(start, end):
    """Calculate the months elapsed between two datetimes,
    returning 0 if a negative time span.
    """
    diff = relativedelta.relativedelta(end, start)
    time_span = diff.months + diff.years * 12
    return time_span if time_span > 0 else 0


#------------------------------------------------------------------------------
# Do a little housekeeping.
#------------------------------------------------------------------------------

# Define format for all plots.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
})

# Read your Fed FRED API key.
# See https://fred.stlouisfed.org/docs/api/api_key.html
config = dotenv_values('../.env')
fred_api_key = config['FRED_API_KEY']


#------------------------------------------------------------------------------
# Read the data.
#------------------------------------------------------------------------------

# Read the collected sales data from an Excel workbook.
panel_data = pd.read_excel('./data/state_cannabis_data.xlsx')
panel_data['time'] = pd.to_datetime(panel_data['date'])

# Read state-specific data from JSON.
with open('data/state_data.json') as file:
    state_data = json.load(file)


#------------------------------------------------------------------------------
# Create supplementary variables.
#------------------------------------------------------------------------------

# Combine panel and state data.
panel_data['adult_use'] = 0
panel_data['medicinal'] = 1
panel_data['months_adult_use'] = 0
panel_data['months_medicinal'] = 0
states_with_data = panel_data['state'].unique()
for state_with_data in states_with_data:
    for state in state_data:
        if state['state'] == state_with_data:
            
            # Create adult-use dummy variable where
            # 1 if adult-use and 0 otherwise.
            panel_data.loc[
                panel_data['state'] == state['state'],
                'adult_use'
            ] = 1 if state['adult_use'] else 0
            
            # Find when / if each state permitted adult-use and medicinal.
            adult_use_start = pd.to_datetime(state.get('adult_use_permitted'))
            medicinal_start = pd.to_datetime(state.get('medicinal_permitted'))
            
            # Assign months of medicinal and adult-use.
            data = panel_data.loc[panel_data['state'] == state['state']]
            for index, values in data.iterrows():
                months_adult_use = months_elapsed(adult_use_start, values['time'])
                months_medicinal = months_elapsed(medicinal_start, values['time'])
                panel_data.at[index, 'months_adult_use'] = months_adult_use
                panel_data.at[index, 'months_medicinal'] = months_medicinal

# Create month fixed effect dummy variables, excluding 1 month for comparison.
month_effects = pd.get_dummies(panel_data['time'].dt.month)
del month_effects[1]


#------------------------------------------------------------------------------
# Estimate a structural regression.
# sales = b0 + b1.population + b2.adult_use + b3.months_medicinal
#         + b4.months_adult_use * adult_use + b5.D(month)
#------------------------------------------------------------------------------

# Regress sales on months permitted, adult-use status, population,
# and a month fixed effect.
Y = panel_data['total_sales']
X = pd.concat([
    panel_data['population'],
    panel_data['adult_use'],
    panel_data['months_medicinal'],
    panel_data['months_adult_use'],
    month_effects,
], axis=1)
X = sm.add_constant(X)
regression = sm.OLS(Y, X).fit()
print(regression.summary())

# Define estimated parameters.
b0 = regression.params['const']
b1 = regression.params['population']
b2 = regression.params['adult_use']
b3 = regression.params['months_medicinal']
b4 = regression.params['months_adult_use']
b5_15 = regression.params[-11:].values

# Optional: Interpret parameters.


#------------------------------------------------------------------------------
# Use the estimated regression to forecast sales in all states in 2022.
#------------------------------------------------------------------------------

# Define forecasting horizon
dates = pd.date_range('2022-01-01', '2023-01-01', freq='M')

# Estimate future sales, state by state.
forecasts = []
for state in state_data:
    
    # Define static variables.
    abv = state['state']
    adult_use = 1 if state['adult_use'] else 0
    adult_use_start = pd.to_datetime(state.get('adult_use_permitted'))
    medicinal_start = pd.to_datetime(state.get('medicinal_permitted'))
    population = get_state_population(fred_api_key, abv, '2020-01-01')[-1]

    # Forecast sales at each date with the regression model.
    for date in dates:

        # Only forecast sales in states with medicinal or adult-use.
        if state['medicinal'] or state['adult_use']:
            
            # Calcuate months of medicinal and adult-use.
            months_medicinal = months_elapsed(medicinal_start, date)
            # if state['adult_use'] or state.get('adult_use_permitted'):
            try:
                adult_use_permitted = adult_use_start >= date
            except TypeError:
                adult_use_permitted = False
            if adult_use_permitted:
                months_adult_use = months_elapsed(adult_use_start, date)
            else:
                months_adult_use = 0
            
            # Identify the month effect.
            if date.month == 1:
                month_effect = 0
            else:
                month_effect = b5_15[date.month - 2]

            # Create a forecast with the specified model.
            forecast = b0 + b1 * population + b2 * adult_use + \
                       b3 * months_medicinal + b4 * months_adult_use + \
                       month_effect
            
        # Indicate that the state has not begun sales.
        else:
            forecast = 0
            
        # Record the forecast for the state.
        state_forecast = {
            'state': abv,
            'date': date,
            'sales_forecast': forecast,
        }
        forecasts.append(state_forecast)

# Turn state forecasts into a DataFrame.
state_forecasts = pd.DataFrame(forecasts)


#------------------------------------------------------------------------------
# Visualize the 2022 forecasts.
#------------------------------------------------------------------------------

# Add a time index.
state_forecasts.index = pd.to_datetime(state_forecasts.date)

# Plot forecasts.
fig, ax = plt.subplots(figsize=(16, 8))
colors = sns.color_palette('Set2', n_colors=len(state_data))
for i in range(len(state_data)):
    abv = state_data[i]['state']
    series = state_forecasts.loc[
        state_forecasts['state'] == abv
    ]['sales_forecast']
    plt.plot(
        series.index,
        series,
        color=colors[i],
        label=abv,
    )
yaxis_format = FuncFormatter(format_millions)
ax.yaxis.set_major_formatter(yaxis_format)
plt.ylim(0)
plt.title('Forecast of Monthly Cannabis Sales by State')
plt.legend(ncol=4, loc='upper left')
plt.savefig(
    'figures/2022_state_cannabis_sales_forecast.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.75,
    transparent=False,
)
plt.show()


#------------------------------------------------------------------------------
# Analyze the 2022 forecasts.
#------------------------------------------------------------------------------

# Calculate total expected sales in each state and in the U.S. in 2022.
print('-----------------------------------------')
print('Predicted cannabis sales by state in 2022')
print('-----------------------------------------')
predictions_2022 = []
for state in state_data:
    abv = state['state']
    state_predictions = state_forecasts.loc[
        state_forecasts['state'] == abv
    ]
    total_sales = state_predictions['sales_forecast'].sum()
    if total_sales <= 0:
        total_sales = 0
    if total_sales >= 1_000_000_000:
        formatted_sales = format_billions(total_sales)
    else:
        formatted_sales = format_millions(total_sales)
    if total_sales:
        print(f'{abv}: {formatted_sales}')
    predictions_2022.append({
        'state': abv,
        'forecast': total_sales
    })

total_us_sales = state_forecasts['sales_forecast'].sum()
print('--------------------------------------------')
print('Predicted total U.S. cannabis sales in 2022:')
print(format_billions(total_us_sales))
print('--------------------------------------------')

# Print in descending order.
predictions_2022 = sorted(
    predictions_2022,
    key = lambda i: i['forecast'],
    reverse=True
)
print('-----------------------------------------')
print('Predicted cannabis sales by state in 2022')
print('-----------------------------------------')
for prediction in predictions_2022:
    total_sales = prediction['forecast']
    if total_sales >= 1_000_000_000:
        formatted_sales = format_billions(total_sales)
    else:
        formatted_sales = format_millions(total_sales)
    if total_sales:
        print(prediction['state'], '≈', formatted_sales)
