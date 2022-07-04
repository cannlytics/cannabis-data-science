"""
Get Cannabis Data for Oklahoma | Cannlytics

Author: Keegan Skeate
Contact: <keegan@cannlytics.com>
Created: 5/27/2021
Updated: 5/30/2021
License: MIT License <https://opensource.org/licenses/MIT>
Data Sources:
    - Medical Marijuana Excise Tax: https://oklahomastate.opengov.com/transparency#/33894/accountType=revenues&embed=n&breakdown=types&currentYearAmount=cumulative&currentYearPeriod=months&graph=bar&legendSort=desc&month=5&proration=false&saved_view=105742&selection=A49C34CEBF1D01A1738CB89828C9274D&projections=null&projectionType=null&highlighting=null&highlightingVariance=null&year=2021&selectedDataSetIndex=null&fiscal_start=earliest&fiscal_end=latest
    - List of Licensed Businesses: https://oklahoma.gov/omma/businesses/list-of-businesses.html
Resources:
    - [SQ788](https://www.sos.ok.gov/documents/questions/788.pdf)
    - [How to Extract Text from a PDF](https://stackoverflow.com/questions/34837707/how-to-extract-text-from-a-pdf-file/63518022#63518022)
"""
# Standard imports
from datetime import datetime
from dotenv import dotenv_values
import os
import sys
import time
from urllib.parse import urljoin

# External imports
from bs4 import BeautifulSoup
import fitz # pymupdf
from fredapi import Fred
import pandas as pd
import requests
import statsmodels.api as sm

# Internal imports
import sys
sys.path.append('../../../')
from cannlytics import firebase # pylint: disable=import-error


STATE = 'OK'

COLUMNS = [
    'name',
    'trade_name',
    'license_number',
    'email',
    'phone',
    'city',
    'zip_code',
    'county',
]

COUNTIES = [
    {'name': 'Adair', 'population_source_code': 'OKADAI1POP'},
    {'name': 'Alfalfa', 'population_source_code': 'OKALFA3POP'},
    {'name': 'Atoka', 'population_source_code': 'OKATOK5POP'},
    {'name': 'Beaver', 'population_source_code': 'OKBEAV7POP'},
    {'name': 'Beckham', 'population_source_code': 'OKBECK9POP'},
    {'name': 'Blaine', 'population_source_code': 'OKBLAI1POP'},
    {'name': 'Bryan', 'population_source_code': 'OKBRYA3POP'},
    {'name': 'Caddo', 'population_source_code': 'OKCADD5POP'},
    {'name': 'Canadian', 'population_source_code': 'OKCANA7POP'},
    {'name': 'Carter', 'population_source_code': 'OKCART9POP'},
    {'name': 'Cherokee', 'population_source_code': 'OKCHER1POP'},
    {'name': 'Choctaw', 'population_source_code': 'OKCHOC3POP'},
    {'name': 'Cimarron', 'population_source_code': 'OKCIMA5POP'},
    {'name': 'Cleveland', 'population_source_code': 'OKCLEV7POP'},
    {'name': 'Coal', 'population_source_code': 'OKCOAL9POP'},
    {'name': 'Comanche', 'population_source_code': 'OKCOMA2POP'},
    {'name': 'Cotton', 'population_source_code': 'OKCOTT3POP'},
    {'name': 'Craig', 'population_source_code': 'OKCRAI5POP'},
    {'name': 'Creek', 'population_source_code': 'OKCREE5POP'},
    {'name': 'Custer', 'population_source_code': 'OKCUST9POP'},
    {'name': 'Delaware', 'population_source_code': 'OKDELA1POP'},
    {'name': 'Dewey', 'population_source_code': 'OKDEWE3POP'},
    {'name': 'Ellis', 'population_source_code': 'OKELLI5POP'},
    {'name': 'Garfield', 'population_source_code': 'OKGARF7POP'},
    {'name': 'Garvin', 'population_source_code': 'OKGARV9POP'},
    {'name': 'Grady', 'population_source_code': 'OKGRAD1POP'},
    {'name': 'Grant', 'population_source_code': 'OKGRAN3POP'},
    {'name': 'Greer', 'population_source_code': 'OKGREE5POP'},
    {'name': 'Harmon', 'population_source_code': 'OKHARM7POP'},
    {'name': 'Harper', 'population_source_code': 'OKHARP9POP'},
    {'name': 'Haskell', 'population_source_code': 'OKHASK1POP'},
    {'name': 'Hughes', 'population_source_code': 'OKHUGH3POP'},
    {'name': 'Jackson', 'population_source_code': 'OKJACK5POP'},
    {'name': 'Jefferson', 'population_source_code': 'OKJEFF7POP'},
    {'name': 'Johnston', 'population_source_code': 'OKJOHN9POP'},
    {'name': 'Kay', 'population_source_code': 'OKKAYC1POP'},
    {'name': 'Kingfisher', 'population_source_code': 'OKKING3POP'},
    {'name': 'Kiowa', 'population_source_code': 'OKKIOW5POP'},
    {'name': 'Latimer', 'population_source_code': 'OKLATI7POP'},
    {'name': 'Le Flore', 'population_source_code': 'OKLEFL9POP'},
    {'name': 'Lincoln', 'population_source_code': 'OKLINC1POP'},
    {'name': 'Logan', 'population_source_code': 'OKLOGA3POP'},
    {'name': 'Love', 'population_source_code': 'OKLOVE5POP'},
    {'name': 'Major', 'population_source_code': 'OKMAJO3POP'},
    {'name': 'Marshall', 'population_source_code': 'OKMARS5POP'},
    {'name': 'Mayes', 'population_source_code': 'OKMAYE7POP'},
    {'name': 'McClain', 'population_source_code': 'OKMCCL7POP'},
    {'name': 'McCurtain', 'population_source_code': 'OKMCCU9POP'},
    {'name': 'McIntosh', 'population_source_code': 'OKMCIN1POP'},
    {'name': 'Murray', 'population_source_code': 'OKMURR9POP'},
    {'name': 'Muskogee', 'population_source_code': 'OKMUSK0POP'},
    {'name': 'Noble', 'population_source_code': 'OKNOBL3POP'},
    {'name': 'Nowata', 'population_source_code': 'OKNOWA5POP'},
    {'name': 'Okfuskee', 'population_source_code': 'OKOKFU7POP'},
    {'name': 'Oklahoma', 'population_source_code': 'OKOKLA9POP'},
    {'name': 'Okmulgee', 'population_source_code': 'OKOKMU1POP'},
    {'name': 'Osage', 'population_source_code': 'OKOSAG3POP'},
    {'name': 'Ottawa', 'population_source_code': 'OKOTTA5POP'},
    {'name': 'Pawnee', 'population_source_code': 'OKPAWN7POP'},
    {'name': 'Payne', 'population_source_code': 'OKPAYN0POP'},
    {'name': 'Pittsburg', 'population_source_code': 'OKPITT1POP'},
    {'name': 'Pontotoc', 'population_source_code': 'OKPONT3POP'},
    {'name': 'Pottawatomie', 'population_source_code': 'OKPOTT0POP'},
    {'name': 'Pushmataha', 'population_source_code': 'OKPUSH7POP'},
    {'name': 'Roger Mills', 'population_source_code': 'OKROGE9POP'},
    {'name': 'Rogers', 'population_source_code': 'OKROGE1POP'},
    {'name': 'Seminole', 'population_source_code': 'OKSEMI3POP'},
    {'name': 'Sequoyah', 'population_source_code': 'OKSEQU5POP'},
    {'name': 'Stephens', 'population_source_code': 'OKSTEP7POP'},
    {'name': 'Texas', 'population_source_code': 'OKTEXA9POP'},
    {'name': 'Tillman', 'population_source_code': 'OKTILL1POP'},
    {'name': 'Tulsa', 'population_source_code': 'OKTULS3POP'},
    {'name': 'Wagoner', 'population_source_code': 'OKWAGO5POP'},
    {'name': 'Washington', 'population_source_code': 'OKWASH7POP'},
    {'name': 'Washita', 'population_source_code': 'OKWASH9POP'},
    {'name': 'Woods', 'population_source_code': 'OKWOOD1POP'},
    {'name': 'Woodward', 'population_source_code': 'OKWOOD3POP'},
]
COUNTY_NAMES = [county['name'].upper() for county in COUNTIES]

EXCLUDE = [
    '',
    'OMMA.ok.gov',
    'Page 518 of 518',
    'Oklahoma Medical Marijuana Authority',
    'Licensed Growers | May 26, 2021',
    'NAME',
    'ZIP',
    'PHONE',
    'CITY',
    'LICENSE No.',
    'EMAIL',
    'COUNTY',
]

EXLUDE_PARTS = [
    'Page ',
    'Licensed ',
    'LIST OF ',
    'As of '
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
}

LICENSEE_LISTS = [
    'https://oklahoma.gov/omma/businesses/list-of-businesses.html',
    'https://oklahoma.gov/omma/businesses/list-of-licensed-businesses/licensed-laboratories.html',
    'https://oklahoma.gov/omma/businesses/list-of-licensed-businesses/licensed-waste-disposal-facility.html',
]

LICENSE_TYPES = {
    'growers': 'producer',
    'processors': 'processor',
    'dispensaries': 'dispensary',
    'transporter': 'transporter',
    'waste_disposal': 'waste_disposal',
    'laboratory': 'lab',
}

MONTHS = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September': 9,
    'October': 10,
    'November': 11,
    'December' : 12,
}

TAX_RATE = 7


def download_website_pdfs(url, destination):
    """Download all PDFs from a given website to a given folder.
    Args:
        url (str): The website that contains the PDFs.
        destination (str): The destination for the PDFS.
    Returns:
        files (list): A list of where all PDFs were downloaded.
    """
    files = []
    if not os.path.exists(destination): os.mkdir(destination)
    response = requests.get(url)
    soup= BeautifulSoup(response.text, 'html.parser')     
    for link in soup.select('a[href$=".pdf"]'):
        file_name = os.path.join(destination, link['href'].split('/')[-1])
        files.append(file_name)
        with open(file_name, 'wb') as f:
            response = requests.get(urljoin(url, link['href']), headers=HEADERS)
            f.write(response.content)
            time.sleep(1)
    return files


def parse_licensee_records(file_name):
    """Parse OMMA licensee records from a given licensee list.
    Args:
        url (str): The website that contains the PDFs.
        destination (str): The destination for the PDFS.
    Returns:
        records (list(list)): A list of lists of values for each licensee.
    """
    records = []
    with fitz.open(file_name) as doc:
        for page in doc:
            page_text = page.getText()
            values = page_text.replace(' \n \n', ' \n').split(' \n')
            row = []
            for value in values:
                if value in EXCLUDE or any(x in value for x in EXLUDE_PARTS):
                    continue
                row.append(value.title())
                if len(row) == 8:
                    records.append(row)
                    row = []
    return records


def clean_licensee_records(records):
    """Clean a list of lists of licensee records into a DataFrame.
    Args:
        records (list(list)): A list of a list of record values.
    Returns:
        data (DataFrame): A DataFrame of licensee records.
    """
    data = pd.DataFrame(records, columns=COLUMNS)
    data['trade_name'] = data['trade_name'].str.replace('Trade Name: ', '')
    data['email'] = data['email'].str.lower()
    data['license_number'] = data['license_number'].str.upper()
    return data


def get_state_current_population(state):
    """Get a given state's latest population from the Fed Fred API,
    getting the number in 1000's and returning the absolute value."""
    config = dotenv_values('../../../.env')
    fred = Fred(api_key=config['FRED_API_KEY'])
    state_code = state.upper()
    population = fred.get_series(f'{state_code}POP')
    return population.iloc[-1] * 1000


def get_licensees_ok():
    """Get cannabis licensees in Oklahoma.
    Download all licensee lists to a new folder in the directory.
    Parse and clean all licensee data.
    """
    date = datetime.now().isoformat()[:10]
    licensee_folder = os.path.join('.datasets', date)
    licensee_files = []
    for url in LICENSEE_LISTS:
        files = download_website_pdfs(url, licensee_folder)
        licensee_files += files
    frames = []
    for file_name in licensee_files:
        license_type = file_name.split('omma_')[1].replace('_list.pdf', '')
        records = parse_licensee_records(file_name)
        data = clean_licensee_records(records)
        data['license_type'] = LICENSE_TYPES.get(license_type)
        frames.append(data)
    licensees = pd.concat(frames)
    licensees.to_excel(f'.datasets/licensees_OK_{date}.xlsx')
    return licensees


def get_licensees_by_county_ok(licensees, counties):
    """Get data on licensees by county in Oklahoma.
    Calculate licensees per capita for each county in Oklahoma
    by finding all licensees in a given county,
    getting the population of that county,
    and calculating the licensees per capita in that county.
    """
    config = dotenv_values('../../../.env')
    fred = Fred(api_key=config['FRED_API_KEY'])
    county_data = {}
    for county in counties:
        county_name = county['name']
        fred_code = county['population_source_code']
        county_licensees = licensees.loc[licensees.county == county_name]
        county_population_data = fred.get_series(fred_code)
        county_population = int(county_population_data.iloc[-1] * 1000)
        county_licensees_per_capita = len(county_licensees) / county_population
        population_date = county_population_data.index[-1].isoformat()[:10]
        entry = {
            'population': f'{county_population:,}',
            'population_source_code': fred_code,
            'population_source': f'https://fred.stlouisfed.org/series/{fred_code}',
            'licensees_per_capita': county_licensees_per_capita,
            'population_at': population_date,
            'total_licensees': len(county_licensees)
        }
        county_data[county_name] = {**county, ** entry}
    return county_data


def get_sales_data_ok():
    """Get cannabis sales data in Oklahoma."""
    # Read Oklahoma tax data.
    # Downloaded from:
    # https://oklahomastate.opengov.com/transparency#/33894/accountType=revenues&embed=n&breakdown=types&currentYearAmount=cumulative&currentYearPeriod=months&graph=bar&legendSort=desc&month=5&proration=false&saved_view=105742&selection=A49C34CEBF1D01A1738CB89828C9274D&projections=null&projectionType=null&highlighting=null&highlightingVariance=null&year=2021&selectedDataSetIndex=null&fiscal_start=earliest&fiscal_end=latest
    date = datetime.now().isoformat()[:10]
    file_name = f'Oklahoma Data Snapshot {date}'
    ext = 'csv'
    raw_excise_tax_data = pd.read_csv(f'.datasets/{file_name}.{ext}', skiprows=4)

    # Format excise tax.
    excise_tax = []
    raw_excise_tax = raw_excise_tax_data.iloc[1].to_dict()
    for key, value in raw_excise_tax.items():
        if 'Actual' in key:
            timestring = key.replace(' Actual', '')
            parts = timestring.replace(' ', '-').split('-')
            month_name = parts[0]
            # FIXME: Format year dynamically.
            month = MONTHS[month_name]
            if month < 6:
                year = '2021'
            else:
                year = '2020'
            excise_tax.append({
                'date': f'{year}-{month}',
                'excise_tax': int(value.strip().replace(',', '')),
            })
    
    # Set a date index.
    revenue_data = pd.DataFrame(excise_tax)
    revenue_data = revenue_data.set_index(pd.to_datetime(revenue_data['date']))
    
    # Calculate the total revenue.
    revenue_data['revenue'] = revenue_data['excise_tax'].diff() * 100 / TAX_RATE
    revenue_data['revenue'].iloc[0] = revenue_data['excise_tax'][0] * 100 / TAX_RATE
    revenue_data.revenue.plot()

    # Add a time index.
    revenue_data['t'] = range(0, len(revenue_data))

    # Save the revenue data.
    revenue_data.to_excel(f'.datasets/revenue_OK_{date}.xlsx')

    # Calculate rate of growth (trend).
    # Run a regression of total revenue on time, where t = 0, 1, ... T.
    model = sm.formula.ols(formula='revenue ~ t', data=revenue_data)
    regression = model.fit()
    print(regression.summary())

    # Plot the trend with total revenue.
    revenue_data['trend'] = regression.predict()
    revenue_data[['revenue', 'trend']].plot()

    # Calculate estimated revenue per dispensary per month.
    dispensaries = licensees.loc[licensees.license_type == 'dispensaries']
    number_of_dispensaries = len(dispensaries)
    revenue_data['revenue_per_dispensary'] = revenue_data['revenue'] / number_of_dispensaries


def get_cannabis_data_ok():
    """Get cannabis data from Oklahoma."""

    return {}


def upload_licensees(data, state):
    """Upload cannabis licensees."""
    for key, values in data.iterrows():
        values['state'] = state
        key = values['license_number']
        ref = f'public/data/licensees/{key}'
        firebase.update_document(ref, values.to_dict())


if __name__ == '__main__':
    
    # Initialize Firebase.
    config = dotenv_values('../../../.env')
    credentials = config['GOOGLE_APPLICATION_CREDENTIALS']
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials
    firebase.initialize_firebase()

    # Get all licensees.
    print('Getting licensees...')
    licensees = get_licensees_ok()
    print('Found data for %i licensees.' % len(licensees))

    # Upload all licensees.
    print('Uploading licensee data...')
    upload_licensees(licensees, STATE)
    print('Uploaded data for %i licensees.' % len(licensees))

    # Get the state's population.
    print('Getting state data...')
    population = get_state_current_population(STATE)

    # Update state data.
    state_key = STATE.lower()
    total_licensees = len(licensees)
    licensees_per_capita = total_licensees / population
    firebase.update_document(f'public/data/state_data/{state_key}', {
        'id': state_key,
        'population': f'{int(population):,}',
        'population_source_code': f'{STATE}POP',
        'population_source': f'https://fred.stlouisfed.org/series/{STATE}POP',
        'total_licensees': f'{total_licensees:,}',
        'licensees_per_capita': f'{licensees_per_capita:,}'
    })
    print('Uploaded state data.')

    # Get licensees, population, and licensees per capita for each county.
    county_data = get_licensees_by_county_ok(licensees, COUNTIES)
    
    # Rank counties by licensees_per_capita.
    county_dataframe = pd.DataFrame.from_dict(county_data, orient='index')
    county_dataframe['licensees_per_capita_rank'] = county_dataframe.licensees_per_capita.rank()
    
    # Upload county data.
    for county_name, values in county_dataframe.iterrows():
        key = county_name.lower().replace(' ', '-')
        ref = f'public/data/state_data/ok/county_data/{key}'
        firebase.update_document(ref, values)
        print('Uploaded data for county', county_name)

    # FIXME: Automate the download of the sales data!!!

    # TODO: Upload the revenue data.

    print('Finished uploading data for', STATE)
