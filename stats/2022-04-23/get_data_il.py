"""
Get Cannabis Data from Illinois
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 4/23/2022
Updated: 4/23/2022
License: MIT License <https://opensource.org/licenses/MIT>

Data Sources:

    - Illinois Cannabis Data
    https://www.idfpr.com/profs/adultusecan.asp

    - Illinois Retailers
    https://www.idfpr.com/LicenseLookup/AdultUseDispensaries.pdf

"""
# Standard imports.
from datetime import datetime
import os

# External imports.
from dotenv import dotenv_values
from fredapi import Fred
import pandas as pd
import pdfplumber
import requests

# Internal imports.
from utils import end_of_period_timeseries


def month_year_to_date(x):
    """Create a time index and only keep rows that start with a
    month name."""
    try:
        return datetime.strptime(x.replace('.0', ''), '%B %Y')
    except:
        return pd.NaT


def get_retailers_il(data_dir, filename):
    """Get retailers operating in Illinois.
    Args:
        data_dir (str): A directory for the data to live.
        filename (str): A filename, without extension, for the data.
    Returns:
        (DataFrame): Returns the retailer data.
    """

    # Download the licensees PDF.
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    licensees_url = 'https://www.idfpr.com/LicenseLookup/AdultUseDispensaries.pdf'
    filename = os.path.join(data_dir, filename + '.pdf')
    response = requests.get(licensees_url)
    with open(filename, 'wb') as f:
        f.write(response.content)

    # Read the licensees PDF.
    pdf = pdfplumber.open(filename)

    # Get all of the table data.
    table_data = []
    for page in pdf.pages:
        table = page.extract_table()
        table_data += table

    # Remove the header.
    table_data = table_data[4:]

    # Remove missing cells.
    table_data = [list(filter(None, x)) for x in table_data]

    # Create a DataFrame from the table data.
    licensee_columns = [
        'organization',
        'trade_name',
        'address',
        'medical',
        'license_issue_date',
        'license_number',
    ]
    licensees = pd.DataFrame(table_data, columns=licensee_columns)

    # Clean the names.
    licensees['organization'] = licensees['organization'].str.replace('\n', '')
    licensees['trade_name'] = licensees['trade_name'].str.replace('\n', '',)
    licensees['trade_name'] = licensees['trade_name'].str.replace('*', '', regex=False)

    # Convert issue date to a datetime.
    licensees['license_issue_date'] = pd.to_datetime(licensees['license_issue_date'])

    # Separate address into 'street', 'city', 'state', 'zip_code', 'phone_number'.
    # Note: This could probably be done more elegantly and it's not perfect.
    streets, cities, states, zip_codes, phone_numbers = [], [], [], [], []
    for _, row in licensees.iterrows():
        parts = row.address.split(' \n')
        streets.append(parts[0])
        phone_numbers.append(parts[-1])
        locales = parts[1]
        city_locales = locales.split(', ')
        state_locales = city_locales[-1].split(' ')
        cities.append(city_locales[0])
        states.append(state_locales[0])
        zip_codes.append(state_locales[-1])
    licensees['street'] = pd.Series(streets)
    licensees['city'] = pd.Series(cities)
    licensees['state'] = pd.Series(states)
    licensees['zip_code'] = pd.Series(zip_codes)
    licensees['phone_number'] = pd.Series(phone_numbers)
    licensees['address'] = licensees['address'].str.replace('\n', '')

    # Set the index as the license number.
    licensees.index = licensees['license_number']

    # Map medical status to True / False.
    licensees = licensees.replace({'medical': {'Yes': True, 'No': False}})

    # Save the licensees data.
    output_file = os.path.join(data_dir, filename + '.xlsx')
    licensees.to_excel(output_file, sheet_name='Data')
    return licensees


def get_sales_il(data_dir, filename, url):
    """Get cannabis sales data in Illinois.
    Args:
        data_dir (str): A directory for the data to live.
        filename (str): A filename, without extension, for the data.
    Returns:
        (DataFrame): Returns the sales data.
    """

    # TODO: Get the url programatically.

    # Download the sales data PDF.
    raw_file = os.path.join(data_dir, filename + '.pdf')
    response = requests.get(url)
    with open(raw_file, 'wb') as f:
        f.write(response.content)

    # Read the sales data PDF.
    pdf = pdfplumber.open(raw_file)

    # Get all of the table data.
    table_data = []
    for page in pdf.pages:

        # Get all of the tables on the page.
        tables = page.find_tables()
        for table in tables:
            data = table.extract()
            table_data += data

    # Add the year to each observation, assuming reverse chronological order
    # starting at the beginning year, 2020, and adding a year at each beginning
    # of year.
    year = 2020
    for row in reversed(table_data):
        row.append(year)
        if row[0] == 'January':
            year += 1

    # Create a DataFrame from the table data.
    sales_columns = [
        'month',
        'items_sold',
        'in_state_sales',
        'out_of_state_sales',
        'total_sales',
        'year',
    ]
    sales_data = pd.DataFrame(table_data, columns=sales_columns)

    # Set the time index.
    dates = sales_data.month.map(str) + ' ' + sales_data.year.map(str)
    dates = dates.apply(month_year_to_date)
    sales_data.index = dates
    sales_data = sales_data.loc[sales_data.index.notnull()]
    sales_data.sort_index(inplace=True)

    # Convert string columns to numeric, handling dollar signs.
    # FIXME: One `out_of_state_sales` has 2 decimal places.
    sales_data[sales_data.columns[1:]] = sales_data[sales_data.columns[1:]] \
        .replace('[\$,]', '', regex=True).astype(float, errors='ignore')

    # Set the index as the end of the month.
    sales_data = end_of_period_timeseries(sales_data)

    # Save the sales data.
    output_file = os.path.join(data_dir, filename + '.xlsx')
    sales_data.to_excel(output_file, sheet_name='Data')
    return sales_data


def calculate_stats_il(licensees, sales_data, env_file='../.env'):
    """Calculate cannabis statistics in Illinois.
    Args:
        data_dir (str): A directory for the data to live.
        filename (str): A filename, without extension, for the data.
    Returns:
        (DataFrame): Returns the sales data.
    """

    # Create total retailers by month series.
    total_retailers = []
    for index, _ in sales_data.iterrows():
        licensed_retailers = licensees.loc[licensees['license_issue_date'] <= index]
        count = len(licensed_retailers)
        total_retailers.append(count)
    sales_data['total_retailers'] = pd.Series(total_retailers, index=sales_data.index)

    # Get the Illinois population data.
    config = dotenv_values(env_file)
    fred_api_key = config.get('FRED_API_KEY')
    fred = Fred(api_key=fred_api_key)
    observation_start = sales_data.index.min().isoformat()
    population = fred.get_series('ILPOP', observation_start=observation_start)
    population = population.multiply(1000) # thousands of people

    # Conjecture that the population remains constant in 2022.
    # Future work: Make this dynamically add any missing years.
    new_row = pd.DataFrame([population[-1]], index=[pd.to_datetime('2022-12-31')])
    population = pd.concat([population, pd.DataFrame(new_row)], ignore_index=False)

    # Project monthly population.
    monthly_population = population.resample('M').mean().pad()
    monthly_population = monthly_population.loc[monthly_population.index <= sales_data.index.max()]

    # Calculate retailers per capita.
    capita = monthly_population / 100_000
    retailers_per_capita = sales_data['total_retailers'] / capita[0]

    # Calculate sales per retailer.
    sales_per_retailer = sales_data['total_sales'] / sales_data['total_retailers']

    # Format and return the statistics.
    stats = pd.concat([retailers_per_capita, sales_per_retailer], axis=1)
    stats.columns = ['retailers_per_capita', 'sales_per_retailer']
    return stats


if __name__ == '__main__':

    # Get retail data.
    data_dir = 'D:\\data\\illinois'
    filename = 'retailers_il_test'
    retailers_il = get_retailers_il(data_dir, filename)

    # Get sales data.
    data_dir = 'D:\\data\\illinois'
    filename = 'sales_il_test'
    url = 'https://www.idfpr.com/Forms/AUC/IDFPR%20monthly%20adult%20use%20cannabis%20sales_4_4_22.pdf'
    sales_data = get_sales_il(data_dir, filename, url)

    # Calculate statistics.
    stats = calculate_stats_il(retailers_il, sales_data)
    stats.to_excel(f'D:\\data\\illinois\\stats_il_test.xlsx')
