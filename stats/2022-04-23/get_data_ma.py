"""
Get Cannabis Data from Massachusetts
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 4/23/2022
Updated: 4/23/2022
License: MIT License <https://opensource.org/licenses/MIT>

Data Sources:

    - MA Cannabis Control Commission

"""
# External imports.
from dotenv import dotenv_values
from fredapi import Fred
import pandas as pd


def calculate_stats_ma(licensees, sales, env_file='../.env'):
    """Calculate cannabis statistics in Illinois.
    Args:
        data_dir (str): A directory for the data to live.
        filename (str): A filename, without extension, for the data.
    Returns:
        (DataFrame): Returns the sales data.
    """

    # Estimate number of operating retailers per day.
    sales['total_retailers'] = 0
    for index, row in sales.iterrows():
        timestamp = row['sale_period'].isoformat()
        sales.at[index, 'total_retailers'] = len(licensees.loc[
            (licensees.license_type == 'Marijuana Retailer') &
            (licensees['cnb_dt_of_final_licensure'] <= timestamp)
        ])

    # Set the index.
    sales.index = sales['sale_period']

    # Get the Illinois population data.
    config = dotenv_values(env_file)
    fred_api_key = config.get('FRED_API_KEY')
    fred = Fred(api_key=fred_api_key)
    observation_start = sales.index.min().isoformat()
    population = fred.get_series('MAPOP', observation_start=observation_start)
    population = population.multiply(1000) # thousands of people

    # Conjecture that the population remains constant in 2022.
    # Future work: Make this dynamically add any missing years.
    new_row = pd.DataFrame([population[-1]], index=[pd.to_datetime('2022-12-31')])
    population = pd.concat([population, pd.DataFrame(new_row)], ignore_index=False)

    # Project monthly population.
    monthly_population = population.resample('M').mean().pad()
    monthly_population = monthly_population.loc[monthly_population.index <= sales.index.max()]

    # Calculate retailers per capita.
    capita = monthly_population / 100_000
    monthly_total_retailers = sales['total_retailers'].resample('M').mean()
    retailers_per_capita = monthly_total_retailers / capita[0]

    # Calculate sales per retailer.
    monthly_sales = sales['gross_sales'].resample('M').sum()
    sales_per_retailer = monthly_sales / monthly_total_retailers

    # Format and return the statistics.
    stats = pd.concat([retailers_per_capita, sales_per_retailer], axis=1)
    stats.columns = ['retailers_per_capita', 'sales_per_retailer']
    stats = stats.loc[stats.index >= pd.to_datetime('2018-11-30')]
    return stats
