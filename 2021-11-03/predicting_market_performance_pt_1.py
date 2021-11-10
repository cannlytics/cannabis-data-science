"""
Applying Box-Jenkins Forecasting Methodology
to Predict Massachusetts Cannabis Data
Copyright (c) 2021 Cannlytics and the Cannabis Data Science Meetup Group

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 10/6/2021
Updated: 11/3/2021
License: MIT License <https://opensource.org/licenses/MIT>

References:
    
    - Time Series forecasting using Auto ARIMA in Python
    https://towardsdatascience.com/time-series-forecasting-using-auto-arima-in-python-bb83e49210cd

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
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import pmdarima as pm
import requests
import seaborn as sns

# Internal imports
from utils import (
    forecast_arima,
    format_millions,
    reverse_dataframe,
    set_training_period,
)


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

# Get the products.
url = f'{base}/xwf2-j7g9.json'
params = {'$limit': 10000, '$order': 'saledate DESC'}
response = requests.get(url,  headers=headers, params=params)
products = pd.DataFrame(response.json(), dtype=float)
products = reverse_dataframe(products)
products.set_index('saledate', inplace=True)

# Plot sales by product type.
product_types = list(products.productcategoryname.unique())
for product_type in product_types:
    print(product_type)
    products.loc[products.productcategoryname == product_type].dollartotal.plot()


#--------------------------------------------------------------------------
# Estimate sales, plants, employees in 2021 and 2022,
#--------------------------------------------------------------------------


# Specifiy training time periods.
train_start = '2020-06-01'
train_end = '2021-10-25'

# Create weekly series.
weekly_sales = production.sales.resample('W-SUN').sum()
weekly_plants = production.total_planttrackedcount.resample('W-SUN').mean()
weekly_employees = production.total_employees.resample('W-SUN').mean()

# Define forecast horizon.
forecast_horizon = pd.date_range(
    pd.to_datetime(train_end),
    periods=60,
    freq='w'
)

# Create month fixed effects (dummy variables),
# excluding 1 month (January) for comparison.
month_effects = pd.get_dummies(weekly_sales.index.month)
month_effects.index = weekly_sales.index
month_effects = set_training_period(month_effects, train_start, train_end)
forecast_month_effects = pd.get_dummies(forecast_horizon.month)
del month_effects[1]
try:
    del forecast_month_effects[1]
except:
    pass

# Estimate sales forecasting model.
model = pm.auto_arima(
    set_training_period(weekly_sales, train_start, train_end),
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
print(model.summary())

# Make sales forecasts.
sales_forecast, sales_conf = forecast_arima(model, forecast_horizon, X=month_effects)


#--------------------------------------------------------------------------
# Visualize the forecasts.
#--------------------------------------------------------------------------

# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams['font.family'] = 'Times New Roman'
palette = sns.color_palette('tab10')
primary_color = palette[0]
secondary_color = palette[-1]

# Plot sales forecast.
fig, ax = plt.subplots(figsize=(15, 5))
weekly_sales[-25:-1].plot(ax=ax, color=primary_color, label='Historic')
sales_forecast.plot(ax=ax, color=secondary_color, style='--', label='Forecast')
plt.fill_between(
    sales_forecast.index,
    sales_conf[:, 0],
    sales_conf[:, 1],
    alpha=0.1,
    color=secondary_color,
)
plt.legend(loc='lower left', fontsize=18)
plt.title('Massachusetts Cannabis Sales Forecast', fontsize=24, pad=10)
yaxis_format = FuncFormatter(format_millions)
ax.yaxis.set_major_formatter(yaxis_format)
plt.setp(ax.get_yticklabels()[0], visible=False)
plt.xlabel('')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()


#--------------------------------------------------------------------------
# Estimate sales per retialer, plants per cultivator,
# and employees per licensee.
#--------------------------------------------------------------------------

# Find total retailers and cultivators.
retailers = licensees.loc[licensees.license_type == 'Marijuana Retailer']
cultivators = licensees.loc[licensees.license_type == 'Marijuana Cultivator']
total_retailers = len(retailers)
total_cultivators = len(cultivators)
total_licensees = len(licensees)

# Create total licensees series.
production['total_retailers'] = 0
production['total_cultivators'] = 0
production['total_licensees'] = 0
for index, _ in production.iterrows():
    timestamp = index.isoformat()
    production.at[index, 'total_retailers'] = len(licensees.loc[
        (licensees.license_type == 'Marijuana Retailer') &
        (licensees.app_create_date <= timestamp)
    ])
    production.at[index, 'total_cultivators'] = len(licensees.loc[
        (licensees.license_type == 'Marijuana Cultivator') &
        (licensees.app_create_date <= timestamp)
    ])
    production.at[index, 'total_licensees'] = len(licensees.loc[
        (licensees.app_create_date <= timestamp)
    ])

# Create weekly averages.
weekly_total_retailers = production['total_retailers'].resample('W-SUN').mean()
weekly_total_cultivators = production['total_cultivators'].resample('W-SUN').mean()
weekly_total_licensees = production['total_licensees'].resample('W-SUN').mean()

# Plot sales per retailer.
sales_per_retailer = weekly_sales / weekly_total_retailers
sales_per_retailer.plot()
plt.show()

# Plot plants per cultivator.
plants_per_cultivator = weekly_plants / weekly_total_cultivators
plants_per_cultivator.plot()
plt.show()

# Plot employees per licensee.
employees_per_license = weekly_employees / weekly_total_licensees
employees_per_license.plot()
plt.show()

#--------------------------------------------------------------------------
# Forecast sales, plants grown, and employees using Box-Jenkins methodology.
# Optional: Also forecast total retailers, total cultivators, and total licensees.
# Optional: Attempt to forecast with daily series with day-of-the-week fixed effects.
# Attempt to forecast with weekly series with month fixed effects.
#--------------------------------------------------------------------------

# Estimate plants forecasting model and make plant forecasts.
model = pm.auto_arima(
    weekly_plants[73:-1],
    X=month_effects[73:-1],
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
    # m=12,
)
print(model.summary())
plants_forecast, plants_conf = model.predict(
    n_periods=len(forecast_horizon),
    return_conf_int=True,
    X=forecast_month_effects,
)
plants_forecast = pd.Series(plants_forecast)
plants_forecast.index = forecast_horizon
weekly_plants[73:-1].plot()
plants_forecast.plot()
plt.show()

# Estimate employees forecasting model and make employees forecasts.
model = pm.auto_arima(
    weekly_employees[73:-1],
    X=month_effects[73:-1],
    start_p=0,
    d=1,
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
    # m=12,
)
print(model.summary())
employees_forecast, plants_conf = model.predict(
    n_periods=len(forecast_horizon),
    return_conf_int=True,
    X=forecast_month_effects,
)
employees_forecast = pd.Series(employees_forecast)
employees_forecast.index = forecast_horizon
weekly_employees[73:-1].plot()
employees_forecast.plot()
plt.show()

# TODO: Forecast total retailers.


# TODO: Forecast total cultivators.


# TODO: Forecast total licensees.


# TODO: Predict total sales per retailer in 2022.


# TODO: Predict total plants per cultivator in 2022.


# TODO: Predict total employees per licensee in 2022.



#--------------------------------------------------------------------------
# Optional: Estimate a production function with the forecasted values
# and calculate the estimated competitive wage and interest rate,
# getting supplemental data from FRED (Federal Reserve Economic Data).
#--------------------------------------------------------------------------

# # Initialize Fred client.
# config = dotenv_values('../.env')
# fred = Fred(api_key=config.get('FRED_API_KEY'))

# # Find the observation time start.
# observation_start = production.index.min().isoformat()

# # # Optional: Get MA population (conjecturing that population remains constant in 2021).
# # population = fred.get_series('MAPOP', observation_start=observation_start)
# # population = end_of_period_timeseries(population, 'Y')
# # population = population.multiply(1000) # Thousands of people
# # new_row = pd.DataFrame([population[-1]], index=[pd.to_datetime('2021-12-31')])
# # population = pd.concat([population, pd.DataFrame(new_row)], ignore_index=False)

# # Get average weekly hours worked in MA.
# avg_weekly_hours = fred.get_series('SMU25000000500000002SA', observation_start=observation_start)
# avg_weekly_hours = end_of_period_timeseries(avg_weekly_hours)
# avg_monthly_hours = avg_weekly_hours.resample('M').sum()

#--------------------------------------------------------------------------
# Optional: Estimate historic competitive wages and interest rates.
#--------------------------------------------------------------------------

# # Define variables.
# Y = weekly_sales
# K = weekly_plants
# L = weekly_employees * avg_weekly_hours

# # Exclude missing observations.
# missing_sales = Y.loc[Y == 0].index
# Y = Y[~Y.index.isin(missing_sales)]
# K = K[~K.index.isin(missing_sales)]
# L = L[~L.index.isin(missing_sales)]

# # Restrict time frame.
# Y = Y[Y.index >= pd.to_datetime('2020-06-01')]
# K = K[K.index >= pd.to_datetime('2020-06-01')]
# L = L[L.index >= pd.to_datetime('2020-06-01')]

# # Define per labor variables.
# y = Y / L
# k = K / L

# # Estimate alpha.
# ln_y = np.log(y)
# ln_k = np.log(k)
# ln_x = np.asarray(sm.add_constant(ln_k))
# regression = sm.OLS(ln_y, ln_x).fit()
# print(regression.summary())


#--------------------------------------------------------------------------
# Optional: Predict future competitive wages and interest rates
# either using estimated historic production function parameters
# or by estimating a new production function for the future observations.
#--------------------------------------------------------------------------

