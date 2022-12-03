"""
New Jersey Forecast
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 4/22/2022
Updated: 4/23/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script creates a forecast of sales in New Jersey
given available data.

Data Sources:

    - New Jersey Retail Licensees
    https://data.nj.gov/Reference-Data/New-Jersey-Cannabis-Dispensary-List/p3ry-ipie

    - New Jersey Approved Licensees
    'https://www.nj.gov/cannabis/businesses/recreational/license-awardees/'

    - New Jersey Medicinal Licensees
    https://www.nj.gov/cannabis/businesses/medicinal/

Setup:

    1. pip install cannlytics

"""
# Internal imports.
from datetime import datetime
import os
import requests
from time import sleep
from urllib.parse import urljoin

# External imports.
from bs4 import BeautifulSoup
from dotenv import dotenv_values
import matplotlib.pyplot as plt
import pandas as pd
import pdfplumber
import seaborn as sns


#-----------------------------------------------------------------------
# Setup
#-----------------------------------------------------------------------

# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 24})
colors = sns.color_palette('Set2', n_colors=10)
primary_colors = sns.color_palette('husl', n_colors=10)

# Define where your data lives.
DATA_DIR = 'D:\\data\\new-jersey'
DATE = datetime.now().isoformat().split('T')[0]


#-----------------------------------------------------------------------
# Access retail licensee data through the Open Data API.
# Socrata endpoint: https://data.nj.gov/api/odata/v4/p3ry-ipie
#-----------------------------------------------------------------------

# Get retail licensees.
base = 'https://data.nj.gov/resource'
endpoint = 'p3ry-ipie'
url = f'{base}/{endpoint}.json'
response = requests.get(url)
retailers = pd.DataFrame(response.json())


#-----------------------------------------------------------------------
# Download all licensee PDFs.
#-----------------------------------------------------------------------

def download_all_pdfs(destination, url, pause=0.2):
    """Download all PDFs on a given web page to a given folder.
    Args:
        destination (str): The folder for the PDFs.
        url (str): The URL of the web page.
        pause (float): An optional pause to wait between downloads, 0.2s by default.
    Author: SIM https://stackoverflow.com/a/54618327
    License: CC BY-SA 4.0 https://creativecommons.org/licenses/by-sa/4.0/
    """
    if not os.path.exists(destination):
        os.mkdir(destination)
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    for link in soup.select("a[href$='.pdf']"):
        filename = os.path.join(destination, link['href'].split('/')[-1])
        with open(filename, 'wb') as f:
            f.write(requests.get(urljoin(url,link['href'])).content)
        sleep(pause)

# Download all permitted licensee PDFs.
# url = 'https://www.nj.gov/cannabis/businesses/recreational/permitted/'
# download_all_pdfs(DATA_DIR + '\\pdfs', url)

# Download all approved licensee PDFs.
# url = 'https://www.nj.gov/cannabis/businesses/recreational/license-awardees/'
# download_all_pdfs(DATA_DIR + '\\pdfs\\approved', url)


#-----------------------------------------------------------------------
# Example PDF parsing.
#-----------------------------------------------------------------------

# Read the licensees PDF.
filename = f'{DATA_DIR}\\pdfs\\approved\\1002_GRC NJ LLC_Final FAD Conditional License Approval Letter.pdf'
pdf = pdfplumber.open(filename)

# Get the license number from the filename.
license_number = filename.split('\\')[-1].split('_')[0]

# Get the time the license was issued.
license_issued = pd.to_datetime(pdf.metadata['SourceModified'].split('D:')[-1])

# Get the license type from the first paragraph.
p0 = pdf.pages[0]
crop = p0.within_bbox((0, 360, p0.width, 420))
text = crop.extract_text()
license_types = ['Cultivator', 'Manufacturer']
for license_type in license_types:
    if license_type in text:
        licensee_type = license_type
        break

# Get all table data.
table_data = []
for page in pdf.pages:
    tables = page.find_tables()
    for table in tables:
        data = table.extract()
        table_data += data


#-----------------------------------------------------------------------
# All PDF parsing
#-----------------------------------------------------------------------

# Extract data programatically for each PDF.
directory = 'D:\\data\\new-jersey\\pdfs\\approved'
approved = []
license_types = ['Cultivator', 'Manufacturer']
fields = ['name', 'license_holder', 'license_email', 'address']
redacted = ['license_holder', 'license_email']
for file in os.listdir(directory):

    # Collect licensee's data.
    licensee = {}

    # Get the license number from the filename.
    licensee['license_number'] = file.split('_')[0]

    # Open the PDF.
    filepath = os.path.join(directory, file)
    pdf = pdfplumber.open(filepath)

    # Get the time the license was issued.
    licensee['license_issued'] = pd.to_datetime(pdf.metadata['SourceModified'].split('D:')[-1])

    # Get the license type from the first paragraph.
    p0 = pdf.pages[0]
    crop = p0.within_bbox((0, 300, p0.width, 485))
    text = crop.extract_text()
    for license_type in license_types:
        if license_type in text:
            licensee['license_type'] = license_type
            break

    # Get licensee name and address.
    table_data = []
    for page in pdf.pages:
        tables = page.find_tables()
        for table in tables:
            data = table.extract()
            table_data += data

    # Record fields in the table.
    for i, field in enumerate(fields):
        if field not in redacted:
            licensee[field] = table_data[i][0]

    # Record the producer.
    approved.append(licensee)
    print('Recorded license', licensee['license_number'])

# Format producer data as a DataFrame.
producers = pd.DataFrame(approved)

# Count the number of different license types.
licensee_counts = producers['license_type'].value_counts()

# Visualize the breakdown of licensees.
fig, ax = plt.subplots(figsize=(8, 8))
licensee_counts.plot(
    kind='bar',
    ax=ax,
    # autopct='%1.1f%%',
    # startangle=90,
    # shadow=False,
    legend=False,
    fontsize=21,
)

# fig.set_facecolor('silver')
plt.ylabel('')
plt.title('Number of Cannabis Producers in New Jersey')
plt.show()

# Add retail counts!
licensee_counts['Retail'] = len(retailers)
licensee_counts['Total'] = len(retailers) + licensee_counts['Cultivator'] + licensee_counts['Manufacturer']

# Visualize licensee breakdown including retail.
fig, ax = plt.subplots(figsize=(12, 8))
licensee_counts.plot(
    kind='bar',
    ax=ax,
    # autopct='%1.1f%%',
    # startangle=90,
    # shadow=False,
    legend=False,
    fontsize=21,
)
# fig.set_facecolor('silver')
plt.ylabel('')
plt.title('Number of Cannabis Licensees in New Jersey')
plt.show()


#-----------------------------------------------------------------------
# Geocode licensees.
#-----------------------------------------------------------------------

# Geocode all New Jersey licensees!
from googlemaps import Client


def geocode_addresses(df, api_key=None, address_field='address'):
    """Geocode addresses in a dataframe.
    Args:
        df (DataFrame): A DataFrame containing the addresses to geocode.
        api_key (str): A Google Maps API key.
        address_field (str): The address field to use in geocoding,
            `address` by default (optional).
    Returns:
        (DataFrame): Returns the DataFrame with geocoded latitudes and longitudes.
    """
    gmaps = Client(key=api_key)
    for index, item in df.iterrows():
        address = item[address_field]
        geocode_result = gmaps.geocode(address)
        if geocode_result:
            location = geocode_result[0]['geometry']['location']
            df.at[index, 'latitude'] = location['lat']
            df.at[index, 'longitude'] = location['lng']
            for info in geocode_result[0]['address_components']:
                key = info['types'][0]

                # FIXME: Also add state!

                if key == 'administrative_area_level_2':
                    county = info['long_name'].replace(' County', '')
                    df.at[index, 'county'] = county
    return df


# Read Google Maps API key.
config = dotenv_values('../.env')
api_key = config['GOOGLE_MAPS_API_KEY']

# Geocode producers.
producers = geocode_addresses(producers, api_key=api_key)

# Visualize the breakdown of licensees by county.
county_counts = producers['county'].value_counts()
fig, ax = plt.subplots(figsize=(8, 8))
county_counts.plot(
    kind='bar',
    ax=ax,
    legend = False,
    fontsize=18,
)
plt.title('Number of Cannabis Producers in New Jersey Counties')
plt.show()


# Future work: Visualize the number of out of state licensees.


#-----------------------------------------------------------------------
# Future work: Augment with additional public data from the
# Census and Fed FRED.
#-----------------------------------------------------------------------

# Future work: Augment with Census data.


# Future work: Augment with Fed FRED data.



#-----------------------------------------------------------------------
# Collect data from other states to form a Bayesian prior.
#-----------------------------------------------------------------------

# Get retailers per capita and sales per retailer in MA.
from get_data_ma import calculate_stats_ma
from opendata import CCC

# Get sales and licensees from the CCC open data API.
ccc = CCC()
sales_ma = ccc.get_retail('sales-weekly')
licensees_ma = ccc.get_licensees('approved')
retailers_ma = licensees_ma.loc[licensees_ma['license_type'] == 'Marijuana Retailer']

# Estimate number of operating retailers per day.
stats_ma = calculate_stats_ma(licensees_ma, sales_ma)
stats_ma.iloc[-1, stats_ma.columns.get_loc('retailers_per_capita')] = stats_ma.iloc[-2]['retailers_per_capita']
stats_ma.to_excel(f'D:\\data\\massachusetts\\stats_ma_{DATE}.xlsx')

#-----------------------------------------------------------------------

# Get retailers per capita and sales per retailer in IL.
from get_data_il import calculate_stats_il, get_retailers_il, get_sales_il

retailers_il = get_retailers_il('D:\\data\\illinois', f'retailers_il_{DATE}')

# Get sales data.
url = 'https://www.idfpr.com/Forms/AUC/IDFPR%20monthly%20adult%20use%20cannabis%20sales_4_4_22.pdf'
sales_data = get_sales_il('D:\\data\\illinois', f'sales_il_{DATE}', url)

# Calculate statistics.
stats_il = calculate_stats_il(retailers_il, sales_data)
stats_il.to_excel(f'D:\\data\\illinois\\stats_il_{DATE}.xlsx')


#-----------------------------------------------------------------------
# Statistics and forecasting.
#-----------------------------------------------------------------------

import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import abline_plot

from utils import format_thousands, format_millions


# Estimate cultivators per capita and compare to other states.
stats_il['t'] = np.arange(0, len(stats_il))
stats_ma['t'] = np.arange(0, len(stats_ma))
stats_il['state'] = 'IL'
stats_ma['state'] = 'MA'
stats = pd.concat([stats_il, stats_ma])
stats.reset_index(inplace=True)
stats.rename(columns={'index': 'date'}, inplace=True)

# Run a regression of sales per retailer on retailers per 100,000 adults.
Y = stats['sales_per_retailer']
X = stats[['t', 'retailers_per_capita']]
X = sm.add_constant(X)
regression = sm.OLS(Y, X).fit()
print(regression.summary())

# Interpret the relationship.
beta = regression.params.retailers_per_capita
statement = """If retailers per 100,000 adults increases by 1,
then everything else held constant one would expect
revenue per retailer to change by {}.
""".format(format_thousands(beta))
print(statement)

# Visualize the correlation.
Y = stats['sales_per_retailer']
X = stats[['retailers_per_capita']]
X = sm.add_constant(X)
corr = sm.OLS(Y, X).fit()
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(
    data=stats,
    x='retailers_per_capita',
    y='sales_per_retailer',
    hue='state',
    ax=ax,
)
abline_plot(
    model_results=corr,
    ax=ax,
)
plt.title("""Relationship between sales per retailer and
retailers per 100,000""", fontsize=24, pad=20)
plt.show()


#-----------------------------------------------------------------------
# Estimate sales in New Jersey
#-----------------------------------------------------------------------

from dotenv import dotenv_values
from fredapi import Fred

# Get New Jersey population data.
# Conjecture that the population remains constant in 2022.
# Future work: Make this dynamically add any missing years.
config = dotenv_values('../.env')
fred_api_key = config.get('FRED_API_KEY')
fred = Fred(api_key=fred_api_key)
observation_start = sales_data.index.min().isoformat()
population = fred.get_series('NJPOP', observation_start=observation_start)
population = population.multiply(1000) # thousands of people

# Calculate retailers per capita.
current_population = population.iloc[-1]
capita = current_population / 100_000
number_of_retailers = len(retailers)
retailers_per_capita = number_of_retailers / capita

# Predict monthly sales per retailer and sales in New Jersey,
# assuming that retailers and population remains constant.
# Future work: Assume population changes linearly.
# Future work: Incorporate an estimation of potential retail entrants
# given any application data.
april = {'const': 1, 't': 0, 'retailers_per_capita': retailers_per_capita}
april_sales_per_retailer = regression.predict(pd.Series(april))
april_sales = april_sales_per_retailer * number_of_retailers

# Predict sales for each month.
predictions = []
horizon = pd.date_range('2022-04-01', '2023-01-01', freq='M')
for i, date in enumerate(horizon):
    obs = {'const': 1, 't': i, 'retailers_per_capita': retailers_per_capita}
    prediction = regression.predict(pd.Series(obs)).iloc[0]
    total = prediction * number_of_retailers
    predictions.append({
        'sales_per_retailer_forecast': prediction,
        'total_sales_forecast': total,
    })

# Compile forecasts.
forecast = pd.DataFrame(predictions, index=horizon)

# Predict total sales in New Jersey in 2022!
forecast_sales_2022 = forecast['total_sales_forecast'].sum()
forecast_sales_per_retailer_2022 = forecast['sales_per_retailer_forecast'].sum()
forecast_avg_sales_per_retailer_2022 = forecast['sales_per_retailer_forecast'].mean()
print('Forecasts')
print('---------')
print(
    'Average monthly sales per retailer in New Jersey in 2022:',
    format_millions(forecast_avg_sales_per_retailer_2022)
)
print(
    'Sales per retailer in New Jersey in 2022:',
    format_millions(forecast_sales_per_retailer_2022)
)
print(
    'Cannabis sales in New Jersey in 2022:',
    format_millions(forecast_sales_2022)
)


#-----------------------------------------------------------------------
# Future work
#-----------------------------------------------------------------------

# Future work: Poisson regression to predict count per county based on various factors.


# Optional: See if rent helps predict where cannabis licensees locate.


# Optional: Estimate the amount of reduced alcohol and tobacco consumption.


#-----------------------------------------------------------------------
# Analyze fees.
#-----------------------------------------------------------------------

# Future work: Fee calculator
license_types = {
   
}
fees = {

}


#-----------------------------------------------------------------------
# Future work: Make statistics accessible through the Cannlytics API.
#-----------------------------------------------------------------------

# Save statistics to local storage.
# producers.to_excel(f'{DATA_DIR}\\stats\\producers-{DATE}.xlsx')

# Future work: Upload statistics to cloud database.

# Future work: Retrieve statistics through the Cannlytics API.
