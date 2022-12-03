"""
Estimate the Cannabis Demand Curve in Massachusetts
Cannabis Data Science Meetup Group
Saturday Morning Statistics
Copyright (c) 2021 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 12/11/2021
Updated: 12/11/2021
License: MIT License <https://opensource.org/licenses/MIT>

Objective:
    
    Attempt to identify the demand curve for cannabis in
    Massachusetts through instrumental variable models.
    
Data Sources:
    
    MA Cannabis Control Commission

    - Average Monthly Price per Ounce for Adult-Use Cannabis: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/rqtv-uenj
    - Approved Massachusetts Licensees: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/hmwt-yiqy
    - Plant Activity and Volume: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/j3q7-3usu
    - Retail Sales by Date and Product Type: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/xwf2-j7g9
    
    U.S. Bureau of Labor Statistics

    - Average Price: Electricity per Kilowatt-Hour in Boston-Cambridge-Newton, MA-NH:    
    https://fred.stlouisfed.org/series/APUS11A72610

"""
# Standard imports.
from typing import Optional, Union

# External imports.
from dotenv import dotenv_values
from fredapi import Fred
from linearmodels import IV2SLS
from linearmodels.iv.results import compare
import matplotlib.pyplot as plt
import pandas as pd
import requests
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import abline_plot

# Internal imports.
from utils import end_of_period_timeseries


def get_socrata_data(
        app_token: str,
        base: str,
        dataset_id: str,
        limit: Optional[int] = None,
        order_by: Optional[str] = None,
        timeseries: Optional[Union[bool, str]] = False,
) -> pd.DataFrame:
    """Get a dataset from the Socrata API.
    Args:
        app_token (str): Your secret Socrata API app token.
        base (str): The Socrata base API url.
        dataset_id (str): The Socrata dataset ID.
        limit (int): A limit to place on the number of observations.
        order_by (str): What key field, if any, to order the data.
        timeseries (bool): Whether or not the data is a timeseries.
            If a string is passed, then it is used to identify the time column.
    Returns:
        (DataFrame): The Socrata data as a Pandas DataFrame.
    """
    headers = {'X-App-Token': app_token}
    params = {'$limit': 10000,  '$order': order_by}
    url = f'{base}/{dataset_id}.json'
    response = requests.get(url,  headers=headers, params=params)
    data = pd.DataFrame(response.json(), dtype=float)
    if timeseries:
        if isinstance(timeseries, str):
            key = timeseries
        else:
            key = order_by.split(' ')[0]
        data.index = pd.to_datetime(data[key])
        data = data.sort_index()
    return data


def licensees_timeseries(
        data: pd.DataFrame,
        operational_key: str,
        license_type: Optional[str] = None,
        license_type_key: Optional[str] = 'license_type',
) -> pd.Series:
    """Create a count of licensees that could be operating over time
    based on when the licensee was issued their license, etc.
    Args:
        data (DataFrame): The licensees data.
        opertaional_key (str): The column used to identify when a licensee
            was operating. The column should be a datetime series.
        license_type (str): An optional license type to filter by.
        license_type_key (str): The column name where any optional license type
            is specified.
    Returns:
        (Series): The total count of licensees that could be operational over
            time. Filtered by `license_type` if specified.
    """
    series = []
    if license_type:
        subset = data.loc[licensees[license_type_key] == license_type]
    else:
        subset = data.copy()
    subset[operational_key] = pd.to_datetime(subset[operational_key])
    subset = subset[subset[operational_key].notnull()]
    start = subset[operational_key].min()
    end = subset[operational_key].max()
    time_index = pd.date_range(start, end)
    for index in time_index:
        count = len(subset.loc[subset[operational_key] <= index])
        series.append(count)
    timeseries = pd.Series(series)
    timeseries.index = time_index
    return timeseries


#--------------------------------------------------------------------------
# Get MA public cannabis data.
#--------------------------------------------------------------------------

# Define dataset IDs.
dataset_ids = {
    'licensees': 'hmwt-yiqy',
    'prices': 'rqtv-uenj',
    'production': 'j3q7-3usu',
    'products': 'xwf2-j7g9',
}

# Setup Socrata API, setting the App Token for request headers.
config = dotenv_values('../.env')
app_token = config.get('APP_TOKEN', None)
base = 'https://opendata.mass-cannabis-control.com/resource'

# Get licensees data.
licensees = get_socrata_data(
        app_token,
        base,
        dataset_ids['licensees'],
        limit=10000,
        order_by='app_create_date DESC',
)

# Get production stats (total employees, total plants, etc.).
production = get_socrata_data(
        app_token,
        base,
        dataset_ids['production'],
        limit=2000,
        order_by='activitysummarydate DESC',
        timeseries=True,
)

# Get the monthly average price per ounce.
prices = get_socrata_data(
        app_token,
        base,
        dataset_ids['prices'],
        limit=10000,
        order_by='date DESC',
        timeseries=True,
)

# Get sales by product type.
products = get_socrata_data(
        app_token,
        base,
        dataset_ids['products'],
        limit=10000,
        order_by='saledate DESC',
        timeseries=True,
)

#--------------------------------------------------------------------------
# Calculate retail statistics.
#--------------------------------------------------------------------------

# Calculate retail sales, coding outliers and negatives as 0.
production['sales'] = production['salestotal'].diff()
outlier = production.loc[production.sales >= 10000000]
production.at[outlier.index, 'sales'] = 0
negatives = production.loc[production.sales < 0]
production.at[negatives.index, 'sales'] = 0

# Create aggregate sales series.
monthly_sales = production.sales.resample('M').sum()

# Create aggregate flower sales series.
flower_sales = products.loc[products['productcategoryname'] == 'Buds']
monthly_flower_sales = flower_sales['dollartotal'].resample('M').sum()

# Create total licensees series.
total_licensees = licensees_timeseries(
        licensees,
        operational_key='cnb_dt_of_final_licensure',
)
total_retailers = licensees_timeseries(
        licensees,
        operational_key='cnb_dt_of_final_licensure',
        license_type='Marijuana Retailer',
)
total_cultivators = licensees_timeseries(
        licensees,
        operational_key='cnb_dt_of_final_licensure',
        license_type='Marijuana Cultivator',
)

# Create aggregate total licensees series.
monthly_total_retailers = total_retailers.resample('M').mean()
monthly_total_cultivators = total_cultivators.resample('M').mean()
monthly_total_licensees = total_licensees.resample('M').mean()

# Create aggregate total employees series.
monthly_employees = production.total_employees.resample('M').mean()

# Calculate the average price per specific quantity. (Bonus)
price_per_gram = prices.avg_1oz.astype(float).divide(28).round(2)
price_per_teenth = prices.avg_1oz.astype(float).divide(16).round(2)
price_per_eighth = prices.avg_1oz.astype(float).divide(8).round(2)
price_per_quarter = prices.avg_1oz.astype(float).divide(4).round(2)

#--------------------------------------------------------------------------
# Get instrumental variables from Fed FRED.
#--------------------------------------------------------------------------

# Initialize FRED API client.
fred_api_key = config.get('FRED_API_KEY')
fred = Fred(api_key=fred_api_key)

# Get electricity prices.
observation_start = production.index.min().isoformat()
energy_prices = fred.get_series('APUS11A72610', observation_start=observation_start)
energy_prices = end_of_period_timeseries(energy_prices)

#--------------------------------------------------------------------------
# Look at average prices and estimated quantity in MA over time.
#--------------------------------------------------------------------------

# Plot prices over time.
flower_price = end_of_period_timeseries(prices['avg_1oz'])
flower_price.plot(title='Average Price per Ounce of Cannabis Flower in MA')
plt.show()

# Estimated the quantity of flower sold.
flower_quantity = monthly_flower_sales / flower_price
flower_quantity.plot(title='Ounces of Cannabis Flower Sold in MA')
plt.show()

#--------------------------------------------------------------------------
# Attempt to identify the demand curve without an instrument.
#--------------------------------------------------------------------------

# Specifiy stats.
# Excluding the period markets were close 2020-04 through 2020-05.
series = {
    'price': flower_price,
    'quantity': flower_quantity,
    'retailers': monthly_total_retailers,
    'change_retailers': monthly_total_retailers.diff(),
    'cultivators': monthly_total_cultivators,
    'change_cultivators': monthly_total_cultivators.diff(),
    'total_licensees': monthly_total_licensees,
    'change_total_licensees': monthly_total_licensees.diff(),
    'employees': monthly_employees,
    'change_employees': monthly_employees.diff(),
    'energy_price': energy_prices,
}
stats = pd.concat(series.values(), axis=1)
stats.columns = series.keys()
stats = stats.loc[(stats.index >= pd.to_datetime('2019-01-01')) &
                  (stats.index <= pd.to_datetime('2021-10-31'))]
stats = stats.loc[(stats.index <= pd.to_datetime('2020-04-01')) |
                  (stats.index >= pd.to_datetime('2020-06-01'))]

# Run a regression of "price" on "quantity".
Y = stats['price']
x = stats['quantity']
X = sm.add_constant(x)
regression = sm.OLS(Y, X).fit()
print(regression.summary())

# Interpret the relationship.
# FIXME: The estimates do not look right...
# https://en.wikipedia.org/wiki/Price_elasticity_of_demand
# b_0 = regression.params.values[0]
# beta = regression.params.values[1]
# percent_change_q = .01
# change_p = beta * percent_change_q
# change_q = x * (1 + percent_change_q) - x
# y_hat = b_0 + beta * x
# elasticity_of_demand = (change_q / change_p) * (y_hat / x)
# print(elasticity_of_demand)

# Visualize the regression.
stats['Price per Ounce'] = Y
stats['Quantity (Ounces)'] = x
ax = stats.plot(
    x='Quantity (Ounces)',
    y='Price per Ounce',
    kind='scatter'
)
abline_plot(
    model_results=regression,
    ax=ax,
)
plt.title("Naive Attempt to Estimate MA Demand \n"
          "Not Accounting for the Identification Problem")
plt.show()

#--------------------------------------------------------------------------
# Attempt to identify the demand curve with an instrument.
#--------------------------------------------------------------------------

# Specifiy an instrument.
z = 'employees'
instrument = stats[z]

# Run the first-stage regression.
Z = sm.add_constant(instrument)
first_stage_regression = sm.OLS(x, Z).fit()
print(first_stage_regression.summary())

# Fit an OLS regression, Z should not be significant.
model_ols = IV2SLS.from_formula(f'price ~ 1 + quantity + {z}', stats)
ols_regression = model_ols.fit() 

# Estimate the second-stage by hand.
stats['x_hat'] = first_stage_regression.predict()
model_tsls = IV2SLS.from_formula('price ~ 1 + x_hat', stats)
tsls_regression = model_tsls.fit()

# Fit an IV two-stage LS regression.
model_iv = IV2SLS.from_formula(f'price ~ 1 + [quantity ~ {z}]', stats)
iv_regression = model_iv.fit()

# Compare OLS and IV2SLS regression estimates.
models = {
        'OLS': ols_regression,
        'IV': iv_regression,
        '2SLS': tsls_regression,
}
print(compare(models))

#--------------------------------------------------------------------------
# Perform diagnostics for proper IV specification.
# TODO: Research interpretation of each test.
#--------------------------------------------------------------------------

# # Durbin's test of exogeneity.
# print(iv_regression.durbin())

# # Wu-Hausman test of exogeneity.
# print(iv_regression.wu_hausman())

# # Wooldridge's regression test of exogeneity.
# print(iv_regression.wooldridge_regression)

# # Wooldridge's score test of exogeneity.
# print(iv_regression.wooldridge_score)

# # Wooldridge's score test of overidentification.
# print(iv_regression.wooldridge_overid)

# # Sargan's test of overidentification.
# print(iv_regression.sargan)

# # First Stage Diagnostics
# print(iv_regression.first_stage)
