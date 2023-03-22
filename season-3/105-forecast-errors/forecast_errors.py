"""
Measuring Forecast Errors | Cannabis Data Science
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 3/22/2023
Updated: 3/22/2023
License: MIT License <https://opensource.org/licenses/MIT>
"""
# External imports:
from cannlytics.data.opendata import OpenData
import matplotlib.pyplot as plt
import pandas as pd
import pmdarima as pm

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 21,
})


#------------------------------------------------------------------------------
# Get the data.
#------------------------------------------------------------------------------

# Initialize a CCC Data Catalog client.
ccc = OpenData()

# Get licensees data and stats.
licensees: pd.DataFrame = ccc.get_licensees()
licensees_approved = ccc.get_licensees('approved')
licensees_pending = ccc.get_licensees('pending')
demographics = ccc.get_licensees('demographics')
application_stats = ccc.get_licensees('application-stats')

# FIXME: This function raises an `APIError`.
# under_review_stats = ccc.get_licensees('under-review-stats')

# Get retail stats.
sales_stats: pd.DataFrame = ccc.get_retail('sales-stats')
sales_weekly: pd.DataFrame = ccc.get_retail('sales-weekly')
prices: pd.DataFrame = ccc.get_retail('price-per-ounce')

# Get agent stats.
gender_stats = ccc.get_agents('gender-stats')
ethnicity_stats = ccc.get_agents('ethnicity-stats')

# Get medical stats.
medical: pd.DataFrame = ccc.get_medical()

# Get cultivation data.
plants: pd.DataFrame = ccc.get_plants()

# Get sales data.
sales: pd.DataFrame = ccc.get_sales()


#------------------------------------------------------------------------------
# Look the data.
#------------------------------------------------------------------------------

# Format prices in MA.
prices['date'] = pd.to_datetime(prices['solddate'], format='%m/%d/%Y')
prices.sort_values(by='date', inplace=True)
prices.set_index('date', inplace=True)

# Look at prices in MA.
prices['price_per_gram'] = pd.to_numeric(prices['averageretailpricegm'])
prices['price_per_gram'].plot(figsize=(10, 6))
plt.xlabel('\nData Source: MA CCC', fontsize=16)
plt.ylabel('Price ($ per gram)')
plt.title('Avg. Price per Gram of Cannabis in MA')
plt.show()

# Format flower sales.
sales['date'] = pd.to_datetime(sales['saledate'], format='%Y-%m-%d')
flower_sales = sales.loc[sales['productcategoryname'] == 'Buds']
flower_sales.sort_values(by='date', inplace=True)
flower_sales.set_index('date', inplace=True)

# Look at weekly flower sales.
flower_sales['sales'] = pd.to_numeric(flower_sales['totalprice'])
weekly = flower_sales['sales'].resample('W').sum()
weekly.div(1_000_000).plot(figsize=(10, 6))
plt.xlabel('\nData Source: MA CCC', fontsize=16)
plt.ylabel('Millions of Dollars ($)')
plt.title('Weekly Cannabis Flower Sales in MA')
plt.show()

# Look at monthly flower sales.
monthly = flower_sales['sales'].resample('M').sum()
monthly.div(1_000_000).plot(figsize=(10, 6))
plt.xlabel('\nData Source: MA CCC', fontsize=16)
plt.ylabel('Millions of Dollars ($)')
plt.title('Monthly Cannabis Flower Sales in MA')
plt.show()


#------------------------------------------------------------------------------
# Model the data.
#------------------------------------------------------------------------------

# Format total monthly cannabis sales.
total_sales = sales.groupby('saledate')['totalprice'].sum()
total_sales.sort_values(inplace=True)
total_monthly = total_sales.resample('M').sum()

# Read in prior forecasts.
priors = pd.read_excel('ma-forecasts-2021-10-27.xlsx')

# Format prior forecasts.
priors.set_index('date', inplace=True)
monthly_prior = priors['total_sales_forecast']

# Visualize prior forecasts against actual.
monthly_prior.div(1_000_000).plot(figsize=(10, 6), label='Predicted')
total_monthly.div(1_000_000).plot(figsize=(10, 6), label='Actual', style='.', ms=12)
plt.xlabel('\nData Source: MA CCC and Saturday Morning Statistics by Cannlytics', fontsize=16)
plt.ylabel('Millions of Dollars ($)')
plt.title('Predicted to Actual Total Cannabis Sales in MA')
plt.legend()
plt.show()


#------------------------------------------------------------------------------
# Use the model to get results.
#------------------------------------------------------------------------------

# Define timeframe for forecasting.
start = pd.to_datetime('2021-01-01')
end = pd.to_datetime('2022-12-31')
forecast_horizon = pd.date_range(
    pd.to_datetime('2023-01-01'),
    periods=52,
    freq='w'
)


# === Forecast prices. ===

# Format average weekly price.
avg_weekly_price = prices['price_per_gram'].resample('W').mean()
avg_weekly_price = avg_weekly_price.loc[start: end]

# Add month fixed effects.
excluded_month = 1
month_effects = pd.get_dummies(avg_weekly_price.index.month)
month_effects.index = avg_weekly_price.index

# Build a forecasting model.
prices_model = pm.auto_arima(
    avg_weekly_price,
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

# Forecast prices.
x_hat = pd.get_dummies(forecast_horizon.month)
prices_forecast, conf = prices_model.predict(
    n_periods=52,
    return_conf_int=True,
    X=x_hat,
)
prices_forecast = pd.Series(prices_forecast)
prices_forecast.index = pd.date_range(
    pd.to_datetime('2023-01-01'),
    periods=52,
    freq='w'
)

# Save the forecasts!
prices_forecast.to_excel('ma-prices-forecast-2023-03-22.xlsx')


# === Forecast sales. ===

# Format total weekly cannabis sales.
total_weekly = total_sales.resample('W').sum()
total_weekly = total_weekly.loc[start: end]

# Add month fixed effects.
excluded_month = 1
month_effects = pd.get_dummies(total_weekly.index.month)
month_effects.index = total_weekly.index

# Build a forecasting model.
sales_model = pm.auto_arima(
    total_weekly,
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

# Forecast sales.
x_hat = pd.get_dummies(forecast_horizon.month)
sales_forecast, conf = sales_model.predict(
    n_periods=52,
    return_conf_int=True,
    X=x_hat,
)
sales_forecast = pd.Series(sales_forecast)
sales_forecast.index = pd.date_range(
    pd.to_datetime('2023-01-01'),
    periods=52,
    freq='w'
)

# Save the forecasts!
sales_forecast.to_excel('ma-total-sales-forecast-2023-03-22.xlsx')


#------------------------------------------------------------------------------
# Analyze the results.
#------------------------------------------------------------------------------

# Visualize forecast alongside historic prices.
prices_forecast.plot(figsize=(10, 6), label='Predicted')
avg_weekly_price.plot(figsize=(10, 6), label='Actual', style='.', ms=12)
plt.xlabel('\nData Source: MA CCC and Cannabis Data Science by Cannlytics', fontsize=16)
plt.ylabel('Price ($)')
plt.title('Predicted Price per Gram of Cannabis Sales in MA')
plt.legend()
plt.show()

# Visualize predicted sales alongside historic sales.
sales_forecast.div(1_000_000).plot(figsize=(10, 6), label='Predicted')
total_weekly.div(1_000_000).plot(figsize=(10, 6), label='Actual', style='.', ms=12)
plt.xlabel('\nData Source: MA CCC and Cannabis Data Science by Cannlytics', fontsize=16)
plt.ylabel('Millions of Dollars ($)')
plt.title('Predicted Total Weekly Cannabis Sales in MA')
plt.legend(loc='lower right')
plt.show()

# What is the total sales predicted in 2023?
total_2023_sales = sales_forecast.sum()
millions = round(total_2023_sales / 1_000_000, 1)
print(f'Estimated 2023 MA cannabis sales: ${millions} million')
