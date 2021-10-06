"""
Market Analysis for Massachusetts Data | Cannabis Data Science

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 10/6/2021
Updated: 10/6/2021
License: MIT License <https://opensource.org/licenses/MIT>

Data Sources:

    MA Cannabis Control Commission
    - Retail Sales by Date and Product Type: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/xwf2-j7g9
    - Approved Massachusetts Licensees: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/hmwt-yiqy
    - Average Monthly Price per Ounce for Adult-Use Cannabis: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/rqtv-uenj
    - Plant Activity and Volume: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/j3q7-3usu
    - Weekly sales by product type: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/87rp-xn9v

    Fed Fred
    - MA Gross Domestic Product: https://fred.stlouisfed.org/series/MANQGSP
    - MA Civilian Labor Force: https://fred.stlouisfed.org/series/MALF
    - MA All Employees: https://fred.stlouisfed.org/series/MANA
    - MA Avg. Weekly Wage: https://fred.stlouisfed.org/series/LES1252881600Q
    - MA Minimum Wage: https://fred.stlouisfed.org/series/STTMINWGMA
    - MA Population: https://fred.stlouisfed.org/series/MAPOP
"""
from dotenv import dotenv_values
from fredapi import Fred
import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table

def end_of_period_timeseries(df, period='M'):
    """Convert a DataFrame from beginning-of-the-period to
    end-of-the-period timeseries.
    Args:
        df (DataFrame): The DataFrame to adjust timestamps.
        period (str): The period of the time series, monthly "M" by default.
    Returns:
        (DataFrame): The adjusted DataFrame, with end-of-the-month timestamps.
    """
    df.index = df.index.to_period(period).to_timestamp(period)
    return df


def reverse_dataframe(df):
    """Reverse the ordering of a DataFrame.
    Args:
        df (DataFrame): A DataFrame to re-order.
    Returns:
        (DataFrame): The re-ordered DataFrame.
    """
    return df[::-1].reset_index(drop=True)


#--------------------------------------------------------------------------
# Get MA public cannabis data.
#--------------------------------------------------------------------------

# Setup Socrata API, get the App Token, and define the headers.
config = dotenv_values('../.env')
app_token = config.get('APP_TOKEN', None)
headers = {'X-App-Token': app_token}
base = 'https://opendata.mass-cannabis-control.com/resource'

# Get production stats (total employees, total plants, etc.) j3q7-3usu
url = f'{base}/j3q7-3usu.json'
params = {'$limit': 2000, '$order': 'activitysummarydate DESC'}
response = requests.get(url,  headers=headers, params=params)
production = pd.DataFrame(response.json(), dtype=float)
production = reverse_dataframe(production)
variables = [
    'activitysummarydate',
    'total_plantimmaturecount',
    'total_planttrackedcount',
    'total_plantfloweringcount',
    'total_plantvegetativecount',
    'total_plantdestroyedcount',
    'total_plantharvestedcount',
    'total_plantcount',
    'salestotal',
    'total_active_harvestcount',
    'total_active_packagecount',
    'total_plantbatchcount',
    'total_activeproducts',
    'total_activestrains',
    'total_employees'
]

# Calculate sales difference.
production['sales'] = production['salestotal'].diff()

# FIX: Fix outlier that appears to have an extra 0.
outlier = production.loc[production.sales >= 10000000]
production.at[outlier.index, 'sales'] = 0

# FIX: Remove negative values.
negatives = production.loc[production.sales < 0]
production.at[negatives.index, 'sales'] = 0

# Aggregate daily production data into monthly and quarterly averages.
production['date'] = pd.to_datetime(production['activitysummarydate'])
production.set_index('date', inplace=True)
monthly_avg_production = production.resample('M').mean()
quarterly_avg_production = production.resample('Q').mean()
monthly_total_production = production.resample('M').sum()
quarterly_total_production = production.resample('Q').sum()

# Get licensees data.
url = f'{base}/hmwt-yiqy.json'
params = {'$limit': 10000,  '$order': 'app_create_date DESC'}
response = requests.get(url,  headers=headers, params=params)
licensees = pd.DataFrame(response.json(), dtype=float)

# Get the monthly average price per ounce.
url = f'{base}/rqtv-uenj.json'
params = {'$limit': 10000, '$order': 'date DESC'}
response = requests.get(url,  headers=headers, params=params)
prices = pd.DataFrame(response.json(), dtype=float)
prices = reverse_dataframe(prices)
prices.set_index('date', inplace=True)

# Calculate the average price per specific quantity.
price_per_gram = prices.avg_1oz.astype(float).divide(28).round(2)
price_per_teenth = prices.avg_1oz.astype(float).divide(16).round(2)
price_per_eighth = prices.avg_1oz.astype(float).divide(8).round(2)
price_per_quarter = prices.avg_1oz.astype(float).divide(4).round(2)

#--------------------------------------------------------------------------
# Get supplemental data from FRED (Federal Reserve Economic Data).
#--------------------------------------------------------------------------

# Initialize Fred client.
config = dotenv_values('../.env')
fred = Fred(api_key=config.get('FRED_API_KEY'))

# Find the observation time start.
start = production.activitysummarydate.min()
observation_start = start.split('T')[0]

# Get the civilian labor force in Massachusetts.
labor_force = fred.get_series('MALF', observation_start=observation_start)
labor_force.index = labor_force.index.to_period('M').to_timestamp('M')

# Get total employees in MA.
total_ma_employees = fred.get_series('MANA', observation_start=observation_start)
total_ma_employees = end_of_period_timeseries(total_ma_employees)
total_ma_employees = total_ma_employees.multiply(1000) # Thousands of people

# Get MA population (conjecturing that population remains constant in 2021).
population = fred.get_series('MAPOP', observation_start=observation_start)
population = end_of_period_timeseries(population, 'Y')
population = population.multiply(1000) # Thousands of people
new_row = pd.DataFrame([population[-1]], index=[pd.to_datetime('2021-12-31')])
population = pd.concat([population, pd.DataFrame(new_row)], ignore_index=False)

# Get Average Weekly Earnings of All Employees: Total Private in Massachusetts.
avg_weekly_wage = fred.get_series('SMU25000000500000011SA', observation_start=observation_start)
avg_weekly_wage = end_of_period_timeseries(avg_weekly_wage)

# Get average weekly hours worked in MA.
avg_weekly_hours = fred.get_series('SMU25000000500000002SA', observation_start=observation_start)
avg_weekly_hours = end_of_period_timeseries(avg_weekly_hours)
avg_monthly_hours = avg_weekly_hours.resample('M').sum()

#--------------------------------------------------------------------------
# Calculate (estimate) GDP from the cannabis industry in MA.
#--------------------------------------------------------------------------

# Calculate GDP from consumption (sales).
cannabis_gdp = production['sales']
quarterly_cannabis_gdp = cannabis_gdp.resample('Q').sum()
quarterly_cannabis_gdp = quarterly_cannabis_gdp.divide(1000000) # Millions of dollars
ma_gdp = fred.get_series('MANQGSP', observation_start=observation_start)
ma_gdp.index = ma_gdp.index.to_period('Q').to_timestamp('Q')

# Calculate cannabis percent of GDP.
cannabis_percent_of_gdp = (quarterly_cannabis_gdp / ma_gdp) * 100

# Calculate percent of annual GDP.
annual_cannabis_gdp = cannabis_gdp.resample('Y').sum()
annual_cannabis_gdp = annual_cannabis_gdp.divide(1000000) # Millions of dollars
annual_ma_gdp = ma_gdp.resample('Y').sum()
annual_percent_of_gdp = annual_cannabis_gdp / annual_ma_gdp * 100

# Calculate (Estimate) the cannabis GDP per Capita in MA.
gdp_per_capita = (annual_cannabis_gdp / population[0]) * 1000000

#--------------------------------------------------------------------------
# Estimate the competitive wage in MA.
#--------------------------------------------------------------------------

# Define economic variables.
Y = monthly_total_production['sales']
K = monthly_avg_production['total_planttrackedcount']
L = (monthly_avg_production['total_employees'] * avg_monthly_hours * 4)

# Exclude missing observations.
missing_obs = [
    pd.to_datetime('2018-10-31'),
    pd.to_datetime('2020-04-30'),
    pd.to_datetime('2021-09-30'),
    pd.to_datetime('2021-10-31'),
]
Y = Y[~Y.index.isin(missing_obs)]
K = K[~K.index.isin(missing_obs)]
L = L[~L.index.isin(missing_obs)]

# Estimate the economic model.
ln_x = np.column_stack([np.log(K), np.log(L)])
ln_x = np.asarray(sm.add_constant(ln_x))
ln_y = np.log(Y)
wage_model = sm.OLS(ln_y, ln_x).fit()
print(wage_model.summary())

# Present the estimated model.
Table, Summary, Labels = summary_table(wage_model, alpha=0.05)
Predictions = Summary[:, 2]
CI_Lower, CI_Upper = Summary[:, 4:6].T
PI_Lower, PI_Upper = Summary[:, 6:8].T

# Estimate the competitive wage.
beta = wage_model.params[-1]
beta_bounds = wage_model.conf_int().iloc[1]
wage = beta * (Y / L)
min_wage = beta_bounds[0] * (Y / L)
max_wage = beta_bounds[1] * (Y / L)

# Plot the competitive wage.
wage.plot()
min_wage.plot()
max_wage.plot()

#--------------------------------------------------------------------------
# Estimate the competitive interest rate in MA.
#--------------------------------------------------------------------------

# Estimate the interest rate.
alpha = wage_model.params[1]
alpha_bounds = wage_model.conf_int().iloc[2]
rate_of_return = alpha * (Y / K)
min_rate_of_return = alpha_bounds[0] * (Y / K)
max_rate_of_return = alpha_bounds[1] * (Y / K)

# Plot the interest rate.
rate_of_return.plot()
min_rate_of_return.plot()
max_rate_of_return.plot()

#--------------------------------------------------------------------------
# Estimate the inflation rate in MA.
#--------------------------------------------------------------------------

inflation = (prices.avg_1oz - prices.avg_1oz.shift()) / prices.avg_1oz.shift()
inflation.plot()
