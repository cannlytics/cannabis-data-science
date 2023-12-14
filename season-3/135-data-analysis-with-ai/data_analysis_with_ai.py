"""
Data Analysis with AI - TerpLife Labs Tests Analysis
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 11/7/2023
Updated: 11/22/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# Standard imports:
from datetime import datetime
import json
import os

# External imports:
from googlemaps import Client
from openai import OpenAI
from dotenv import dotenv_values
import matplotlib.pyplot as plt
import pandas as pd


# === Initialization ===

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})

# Initialize OpenAI client.
env_file = '../../.env'
os.environ['OPENAI_API_KEY'] = dotenv_values(env_file)['OPENAI_API_KEY']
openai_api_key = os.environ['OPENAI_API_KEY']
client = OpenAI()


# === Get the data ===

# Read TerpLife Labs COA data.
results = pd.read_excel('data/terplife-labs-coa-data-2023-11-21.xlsx')
print('Read {} rows of data.'.format(len(results)))

# Assign date columns.
results['date'] = pd.to_datetime(results['date_tested'])
results['month_year'] = results['date'].dt.to_period('M')
print('Starting date: {}'.format(results['date'].min()))
print('Ending date: {}'.format(results['date'].max()))


# === Augment the data. ===

# Identify all unique cultivation facilities.
locations = list(results['producer_address'].unique())

# Geocode licenses.
googlemaps_api_key = dotenv_values(env_file)['GOOGLE_MAPS_API_KEY']
gmaps = Client(key=googlemaps_api_key)
address_coordinates = {}
for address in locations:
    geocode_result = gmaps.geocode(address)
    location = geocode_result[0]['geometry']['location']
    address_coordinates[address] = (location['lat'], location['lng'])
    
# Add latitude and longitude to the dataframe
results['producer_latitude'] = results['producer_address'].apply(lambda x: address_coordinates[x][0] if x in address_coordinates else None)
results['producer_longitude'] = results['producer_address'].apply(lambda x: address_coordinates[x][1] if x in address_coordinates else None)


# === Analyze the time data. ===

# Look at time between date_produced and date_tested.
results['date_produced'] = pd.to_datetime(results['date_produced'])
time_diff = (results['date'] - results['date_produced']).dt.days
plt.figure(figsize=(12, 8))
time_diff.hist(bins=100)
plt.xlabel('Days')
plt.ylabel('Count')
plt.title('Time from Batch Creation to Batch Test in Florida')
plt.legend()
plt.tight_layout()
plt.savefig(f'figures/terplife-batch-to-test-time.png', bbox_inches='tight', dpi=300)
plt.show()

# Find all the unique space-time coordinates.
unique_dates_by_coordinate = results.groupby(['producer_latitude', 'producer_longitude'])['date_produced'].unique()

# Future work: Augment weather data.


# === Analyze the production data. ===

# Clean batch sizes.
results['product_size'] = results['product_size'].str.replace('[gG,]', '', regex=True).astype(float)
results['batch_size'] = results['batch_size'].str.replace('[gG,]', '', regex=True).astype(float)

# Look at the average size of flower batches.
flower = results[results['product_type'] == 'Flower Inhalable']
flower['batch_size'].div(28).hist(bins=100)
plt.xlabel('Batch Size (Ounces)')
plt.ylabel('Count')
plt.title('Average Size of Flower Batches in Florida')
plt.legend()
plt.tight_layout()
plt.savefig(f'figures/terplife-flower-batch-size.png', bbox_inches='tight', dpi=300)
plt.show()

# Calculate total flower produced.
total_flower = flower.loc[flower['date_produced'] >= pd.to_datetime('2023-01-01')]['batch_size'].sum()
tons = round(total_flower * 0.035274 * 3.125e-5, 1)
print(f'Total flower produced: {tons} tons.')


# === Analyze the strain data. ===

# Find the top 20 strains by amount produced.
batch_size_by_strain = results.groupby('strain_name')['batch_size'].sum()
top_20_strains = batch_size_by_strain.nlargest(21)
plt.figure(figsize=(12, 8))
plt.barh(top_20_strains.index[1:], top_20_strains.values[1:], color='skyblue')
plt.xlabel('Total Produced (Grams)')
plt.title('Top 20 Produced Strains')
plt.tight_layout()
plt.savefig(f'figures/terplife-total-produced-by-strain.png', bbox_inches='tight', dpi=300)
plt.show()


# === Chemical analysis. ===

# Look at total THC in flower.
flower = results[results['product_type'] == 'Flower Inhalable']
flower['total_thc'].hist(bins=100)
plt.ylabel('Count')
plt.xlabel('Percent')
plt.title('Total THC of Flower Tested by TerpLife Life Labs in Florida')
plt.tight_layout()
plt.savefig(f'figures/terplife-total-thc.png', bbox_inches='tight', dpi=300)
plt.show()

# Look at farnesene in flower.
flower['trans_beta_farnesene'].hist(bins=100)
plt.ylabel('Count')
plt.xlabel('Percent')
plt.title('Farnesene Concentrations in Flower Tested by TerpLife Life Labs in Florida')
plt.tight_layout()
plt.savefig(f'figures/terplife-farnesene.png', bbox_inches='tight', dpi=300)
plt.show()

concentrate = results[results['product_type'] == 'Derivative Inhalable']
concentrate['trans_beta_farnesene'].hist(bins=100)
plt.ylabel('Count')
plt.xlabel('Percent')
plt.title('Farnesene Concentrations in Concentrates Tested by TerpLife Life Labs in Florida')
plt.tight_layout()
plt.savefig(f'figures/terplife-farnesene-conc.png', bbox_inches='tight', dpi=300)
plt.show()

# === Given a lab's website (or name), get prices for analyses. ===

# TODO: Find cost per product (price per batch / products per batch)



# === Given an image, identify if it is a COA, product label, or receipt. ===



# === Describe a product image. ===



# === Parse licensee data block by block. ===



# === Parse strain names from blocks of text. ===



# === Parse known analyses / analytes from a COA. ===



# === Determine if a compound is a cannabinoid, terpene, or other. ===

def identify_analysis(compound: str) -> dict:
    """Identify if a compound is a cannabinoid, terpene, or other."""
    print('\nIdentify analysis for:', compound)
    start = datetime.now()
    prompt = 'Is the following compound a terpene or a cannabinoid or neither?'
    prompt += f'\n\nCompound: {compound}'
    instructions = 'Return your analysis as JSON, e.g. {"analysis": "terpenes"}'
    messages = [
        {'role': 'system', 'content': instructions},
        {'role': 'user', 'content': prompt},
    ]
    completion = client.chat.completions.create(
        model='gpt-4-1106-preview',
        messages=messages,
        max_tokens=200,
        temperature=0.0,
        user='cannlytics',
    )
    end = datetime.now()
    usage = completion.model_dump()['usage']
    cost = 0.01 / 1_000 * usage['prompt_tokens'] + 0.03 / 1_000 * usage['completion_tokens']
    time = end - start
    print('Cost:', cost)
    print('Time:', time)
    content = completion.choices[0].message.content
    extracted_json = content.lstrip('```json\n').rstrip('\n```')
    extracted_data = json.loads(extracted_json)
    print('Extracted:', extracted_data)
    return extracted_data, cost


# Identify various compounds.
analysis, cost = identify_analysis('3-Carene')
analysis, cost = identify_analysis('d9 - Tetrahydrocannabinolic acid (THCA)')
analysis, cost = identify_analysis('Cannabidivarin (CBDV)')
analysis, cost = identify_analysis('Farnesene')
analysis, cost = identify_analysis('Selenium')
