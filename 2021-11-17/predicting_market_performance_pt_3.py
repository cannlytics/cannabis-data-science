"""
Applying Box-Jenkins Forecasting Methodology
to Predict Massachusetts Cannabis Data
Copyright (c) 2021 Cannlytics and the Cannabis Data Science Meetup Group

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 10/6/2021
Updated: 11/17/2021
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
    - Average Hourly Earnings of All Employees: Total Private in Massachusetts
      https://fred.stlouisfed.org/series/SMU25000000500000003SA
    - MA Minimum Wage Rate: https://fred.stlouisfed.org/series/STTMINWGMA
"""
from dotenv import dotenv_values
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import pmdarima as pm
import requests
import seaborn as sns
import statsmodels.api as sm

# Internal imports
from utils import (
    end_of_period_timeseries,
    forecast_arima,
    format_millions,
    format_thousands,
    reverse_dataframe,
    set_training_period,
)


CAPITAL = 'total_plantfloweringcount'
# CAPITAL = 'total_planttrackedcount'


#--------------------------------------------------------------------------
# Get all MA public cannabis data.
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
product_types = list(products.productcategoryname.unique())

#--------------------------------------------------------------------------
# Create novel statistics.
# Estimate historic sales per retialer, plants per cultivator,
# and employees per licensee.
#--------------------------------------------------------------------------

# Create weekly series.
weekly_sales = production.sales.resample('W-SUN').sum()
weekly_plants = production[CAPITAL].resample('W-SUN').mean()
weekly_employees = production.total_employees.resample('W-SUN').mean()

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

# Estimate sales per retailer.
sales_per_retailer = weekly_sales / weekly_total_retailers
sales_per_retailer.plot()
plt.show()

# Estimate plants per cultivator.
plants_per_cultivator = weekly_plants / weekly_total_cultivators
plants_per_cultivator.plot()
plt.show()

# Estimate employees per licensee.
employees_per_license = weekly_employees / weekly_total_licensees
employees_per_license.plot()
plt.show()

#--------------------------------------------------------------------------
# Estimate sales, plants grown, and employees,
# total retailers, total cultivators, and total licensees
# in 2021 and 2022 using Box-Jenkins (ARIMA) methodology.
# Month fixed effects are used.
# Optional: Attempt to forecast with daily series with
# day-of-the-week fixed effects.
#--------------------------------------------------------------------------

# Specifiy training time periods.
train_start = '2020-06-01'
train_end = '2021-11-15'
forecast_end = '2023-01-01'

# Define forecast horizon.
forecast_horizon = pd.date_range(
    start=pd.to_datetime(train_end),
    end=pd.to_datetime(forecast_end),
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

# Forecast sales.
sales_model = pm.auto_arima(
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
sales_forecast, sales_conf = forecast_arima(
    sales_model,
    forecast_horizon,
    X=forecast_month_effects
)

# Forecast total plants.
plants_model = pm.auto_arima(
    set_training_period(weekly_plants, train_start, train_end),
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
plants_forecast, plants_conf = forecast_arima(
    plants_model,
    forecast_horizon,
    X=forecast_month_effects
)

# Forecast total employees.
employees_model = pm.auto_arima(
    set_training_period(weekly_employees, train_start, train_end),
    X=month_effects,
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
)
employees_forecast, employees_conf = forecast_arima(
    employees_model,
    forecast_horizon,
    X=forecast_month_effects
)

# Forecast total retailers.
retailers_model = pm.auto_arima(
    set_training_period(weekly_total_retailers, train_start, train_end),
    X=month_effects,
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
total_retailers_forecast, total_retailers_conf = forecast_arima(
    retailers_model,
    forecast_horizon,
    X=forecast_month_effects
)

# Forecast total cultivators.
cultivators_model = pm.auto_arima(
    set_training_period(weekly_total_cultivators, train_start, train_end),
    X=month_effects,
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
total_cultivators_forecast, total_cultivators_conf = forecast_arima(
    cultivators_model,
    forecast_horizon,
    X=forecast_month_effects
)

# Forecast total licensees.
licensees_model = pm.auto_arima(
    set_training_period(weekly_total_licensees, train_start, train_end),
    X=month_effects,
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
total_licensees_forecast, total_licensees_conf = forecast_arima(
    licensees_model,
    forecast_horizon,
    X=forecast_month_effects
)

# Predict total sales per retailer in 2022.
# TODO: Figure out how to estimate confidence bounds?
forecast_sales_per_retailer =  sales_forecast / total_retailers_forecast

# Predict total plants per cultivator in 2022.
forecast_plants_per_cultivator =  plants_forecast / total_cultivators_forecast

# Predict total employees per licensee in 2022.
forecast_employees_per_license =  employees_forecast / total_licensees_forecast

# TODO: Calculate totals for 2022.


#--------------------------------------------------------------------------
# Visualize the forecasts as 2x3 time series plots.
# Sales | Plants
# Retailers | Cultivators
# Sales per Retailer | Plants per Cultivator
#--------------------------------------------------------------------------

def plot_forecast(
        ax,
        forecast,
        historic=None,
        conf=None,
        title=None,
        color=None,
        formatter=None,
):
    """Plot a time series forecast.
    Args:
        ax (): The axes on which to plot the forecast.
        forecast (Series): The forecast to plot.
        historic (Series): Optional historic time series to plot.
        conf (Array): An optional 2xN array of lower and upper confidence
            bounds for the forecast series.
        title (str): An optional title to place above the chart.
        color: (str): An optional color hex code.
        formatter (func): An optional formatting function for the Y axis.
    """
    forecast.plot(color=color, style='--', label='Forecast')
    if conf is not None:
        plt.fill_between(
            forecast.index,
            conf[:, 0],
            conf[:, 1],
            alpha=0.1,
            color=color,
        )
    if historic is not None:
        historic.plot(color=color, label='Historic')
    if title is not None:
        plt.title(title, fontsize=24, pad=10)
    if formatter is not None:
        yaxis_format = FuncFormatter(formatter)
        ax.yaxis.set_major_formatter(yaxis_format)
    plt.gca().set(ylim=0)
    plt.setp(ax.get_yticklabels()[0], visible=False)
    plt.xlabel('')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams['font.family'] = 'Times New Roman'
palette = sns.color_palette('tab10')

# Plot all series.
fig = plt.figure(figsize=(40, 25))

# Plot sales.
ax1 = plt.subplot(3, 2, 1)
plot_forecast(
        ax1,
        sales_forecast,
        historic=weekly_sales,
        conf=sales_conf,
        title='Cannabis Sales',
        color=palette[0],
        formatter=format_millions,
)

# Plot plants.
ax2 = plt.subplot(3, 2, 2)
plot_forecast(
        ax2,
        plants_forecast,
        historic=weekly_plants,
        conf=plants_conf,
        title='Cannabis Plants',
        color=palette[-1],
        formatter=format_thousands,
)

# Plot retailers.
ax3= plt.subplot(3, 2, 3)
plot_forecast(
        ax3,
        total_retailers_forecast,
        historic=weekly_total_retailers,
        conf=total_retailers_conf,
        title='Cannabis Retailers',
        color=palette[3],
        # formatter=format_thousands,
)

# Plot cultivators.
ax4= plt.subplot(3, 2, 4)
plot_forecast(
        ax4,
        total_cultivators_forecast,
        historic=weekly_total_cultivators,
        conf=total_cultivators_conf,
        title='Cannabis Cultivators',
        color=palette[2],
        # formatter=format_thousands,
)

# Plot average sales per retailer.
ax5= plt.subplot(3, 2, 5)
plot_forecast(
        ax5,
        forecast_sales_per_retailer,
        historic=sales_per_retailer,
        # conf=total_cultivators_conf,
        title='Average Cannabis Sales per Retailer',
        color=palette[4],
        formatter=format_thousands,
)

# Plot average plants per cultivator.
ax6 = plt.subplot(3, 2, 6)
plot_forecast(
        ax6,
        forecast_plants_per_cultivator,
        historic=plants_per_cultivator,
        # conf=total_cultivators_conf,
        title='Average Cannabis Plants per Cultivator',
        color=palette[5],
        # formatter=format_thousands,
)

# Add figure notes.
notes = """Data: Cannabis sales, total tracked plants, and licensees data from October 15, 2018 through October 26, 2021.
Data Source: MA Cannabis Control Commission.
Notes: Box-Jenkins methodology is used for forecasting. Massachusetts cannabis markets were closed from 3/26/2021 through 5/24/2020.
Forecasts were made with data post 5/25/2020."""
plt.figtext(0.05, -0.01, notes, ha='left', fontsize=16)

# Add a title above all of the subplots.
fig.suptitle(
    'Massachusetts Historic and Predicted Cannabis Market Performance',
    fontsize=40
)

# Save the figure.
plt.margins(1, 1)
plt.savefig(
    'figures/ma_market_forecast.png',
    dpi=96,
    bbox_inches='tight',
    pad_inches=0.75,
    transparent=False,
)

# Show the figure.
plt.show()

# Optional: Show a legend?
# plt.legend(loc='lower left', fontsize=18)

# Optional: Find a clever way to display 2022 forecast totals (curly braces?)

#--------------------------------------------------------------------------
# Optional: Estimate a production function with the forecasted values
# and calculate the estimated competitive wage and interest rate,
# getting supplemental data from FRED (Federal Reserve Economic Data).
#--------------------------------------------------------------------------

# Initialize Fred client.
from fredapi import Fred
config = dotenv_values('../.env')
fred = Fred(api_key=config.get('FRED_API_KEY'))

# Find the observation time start.
observation_start = production.index.min().isoformat()

# Get average weekly hours worked in MA.
avg_weekly_hours = fred.get_series('SMU25000000500000002SA', observation_start=observation_start)
avg_weekly_hours = end_of_period_timeseries(avg_weekly_hours)
avg_weekly_hours = avg_weekly_hours.resample('W-Sun').ffill().iloc[:-1]

# Get Average Hourly Earnings of All Employees: Total Private in Massachusetts
# SMU25000000500000003SA
avg_earnings = fred.get_series('SMU25000000500000003SA', observation_start=observation_start)
avg_earnings = end_of_period_timeseries(avg_earnings)
avg_earnings = avg_earnings.resample('W-Sun').ffill().iloc[:-1]

# Get the state minimum wage rate for Massachusetts.
min_wage = fred.get_series('STTMINWGMA', observation_start=observation_start)
min_wage = end_of_period_timeseries(min_wage)
min_wage = min_wage.resample('W-Sun').ffill().iloc[:-1]

#--------------------------------------------------------------------------
# Optional: Estimate historic competitive wages and interest rates.
#--------------------------------------------------------------------------

# Define variables.
Y = weekly_sales
K = weekly_plants
L = weekly_employees * avg_weekly_hours

# Restrict time frame.
Y = Y.loc[
    (Y.index >= pd.to_datetime('2019-01-01')) &
    (Y.index <= pd.to_datetime('2021-09-30'))
]
K = K.loc[
    (K.index >= pd.to_datetime('2019-01-01')) &
    (K.index <= pd.to_datetime('2021-09-30'))
]
L = L.loc[
    (L.index >= pd.to_datetime('2019-01-01')) &
    (L.index <= pd.to_datetime('2021-09-30'))
]

# Exclude missing observations.
missing_sales = Y.loc[Y == 0].index
Y = Y[~Y.index.isin(missing_sales)]
K = K[~K.index.isin(missing_sales)]
L = L[~L.index.isin(missing_sales)]

# Estimate alpha and beta.
# The typical is 0.3 for alpha and 0.7 for beta for most industries.
ln_y = np.log(Y)
ln_k = np.log(K)
ln_l = np.log(L)
X = pd.concat([ln_k, ln_l], axis=1)
X = sm.add_constant(X)
regression = sm.OLS(ln_y, X).fit()
print(regression.summary())
alpha = regression.params[CAPITAL]
beta = regression.params[0]

# Estimate historic competitive wage.
wage = beta * Y / L
wage.plot()
plt.show()

# Estimate historic competitive interest rate.
interest_rates = alpha * Y / K
interest_rates.plot()
plt.show()

#--------------------------------------------------------------------------
# Predict the future competitive wages and interest rates
# either using estimated historic production function parameters
# or by estimating a new production function for the future observations.
#--------------------------------------------------------------------------

# Set forecast time index for the wage series.
wage_month_effects = month_effects.loc[
    (month_effects.index >= wage.index.min()) &
    (month_effects.index <= wage.index.max())
]
wage_forecast_horizon = pd.date_range(
    start=wage.index.max(),
    end=pd.to_datetime(forecast_end),
    freq='w'
)
wage_forecast_month_effects = pd.get_dummies(wage_forecast_horizon.month)
try:
    del wage_forecast_month_effects[1]
except:
    pass

# Atheoretical forecasting.
wage_model = pm.auto_arima(
    set_training_period(wage, train_start, train_end),
    X=wage_month_effects,
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
wage_forecast, wage_conf = forecast_arima(
    wage_model,
    wage_forecast_horizon,
    X=wage_forecast_month_effects
)

# Plot the wage and wage forecast.
wage.plot()
wage_forecast.plot()
wages_2022 = wage_forecast.loc[wage_forecast.index >= pd.to_datetime('2022-01-01')]
print('Forecast average wage for 2022:', round(wages_2022.mean(), 2))

fig = plt.figure(figsize=(10, 5))
ax1 = plt.subplot(1, 1, 1)
plot_forecast(
        ax1,
        wage_forecast,
        historic=wage,
        conf=wage_conf,
        title='Competitive Wage Forecast',
        color=palette[0],
)
plt.show()

# Plot all wages together.
fig = plt.figure(figsize=(10, 5))
avg_earnings.plot()
min_wage.plot()
wage.plot()
wage_forecast.plot()
plt.fill_between(
    wage_forecast.index,
    wage_conf[:, 0],
    wage_conf[:, 1],
    alpha=0.1,
    color='green',
)
plt.ylim(0)
plt.show()

# TODO: Atheoretical forecast of the rate of return of plants.


#--------------------------------------------------------------------------
# Look at labor share of income.
#--------------------------------------------------------------------------

# Historic labor share of income. Notice, this is beta!
gross_wages = wage * \
              avg_weekly_hours.loc[
                  (avg_weekly_hours.index >= wage.index.min()) &
                  (avg_weekly_hours.index <= wage.index.max())
              ] * \
              weekly_employees.loc[
                  (weekly_employees.index >= wage.index.min()) &
                  (weekly_employees.index <= wage.index.max())
              ]
labor_share_of_income = gross_wages / weekly_sales.loc[
                  (weekly_sales.index >= wage.index.min()) &
                  (weekly_sales.index <= wage.index.max())
              ]
labor_share_of_income.plot()
plt.show()
assert round(labor_share_of_income.mean(), 4) == round(beta, 4)

#--------------------------------------------------------------------------
# Theoretical forecasting.
#--------------------------------------------------------------------------

# Estimate future competitive wage with regression params and
# predicted sales and labor.
L_forecast = employees_forecast * avg_weekly_hours.iloc[-1]
theory_wage_forecast = beta * sales_forecast / (L_forecast)
theory_wage_forecast.plot()
plt.show()

# Plot theoretical with atheoretical wage forecast.
fig = plt.figure(figsize=(10, 5))
avg_earnings.plot()
min_wage.plot()
wage.plot()
wage_forecast.plot()
plt.fill_between(
    wage_forecast.index,
    wage_conf[:, 0],
    wage_conf[:, 1],
    alpha=0.1,
    color='green',
)
theory_wage_forecast.plot()
plt.ylim(0)
plt.show()

# Estimate future competitive interest rate with regression params and
# predicted sales and plants.
interest_rates.plot()
interest_rates_forecast = alpha * sales_forecast / plants_forecast
interest_rates_forecast.plot()
plt.show()
