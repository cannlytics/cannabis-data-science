"""
Analyze Results from MI PRR
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 10/26/2023
Updated: 10/26/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# External imports:
from datetime import datetime
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib import cm
import numpy as np
import pandas as pd
import re
import seaborn as sns
from scipy import stats


# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})


def save_figure(filename, dpi=300, bbox_inches='tight'):
    """Save a figure to the figures directory."""
    plt.savefig(f'figures/{filename}', bbox_inches=bbox_inches, dpi=dpi)



def extract_strain_name(product_name):
    """Extract the strain name from the product name."""
    name = str(product_name)
    name = name.replace('_', ' ')
    # TODO: Remove first word if it's a number.

    # strain_name = re.split(r' - | \| | _ | x | – | — |:|\(|\)|/', name)[0]
    # FIXME: Remove product type
    # Testing Sample
    # Finished Flower
    # Bulk Bud
    # Bulk
    # Trim
    # Harvest Batch
    # .rstrip('.')
    # Trim/Shake
    # (BUDS) 
    # (SHAKE)
    # (Shake/Trim)
    # Dry Flower
    # Kief
    # Cuttings
    # Preroll Material
    # Biomass 
    # Harvested
    # Popcorn
    strain_name = strain_name.split('Buds')[0].strip()
    strain_name = strain_name.split('Bulk')[0].strip()
    strain_name = strain_name.split('Flower')[0].strip()
    strain_name = strain_name.split('Pre-Roll')[0].strip()

    # TODO: Remove #00

    return strain_name


# === Get the data ===

# Read MA lab results.
mi_results = pd.read_excel('./data/mi-prr-lab-results-2023-02-17.xlsx')
ct_results = pd.read_excel('./data/ct-lab-results-2023-10-27.xlsx')
wa_results = pd.read_csv('./data/wa-lab-results-latest.csv')


# === Clean the data ===

# Rename certain columns.
mi_results = mi_results.rename(columns={
    'ProductName': 'product_name',
    'ProductCategory': 'product_type',
    'TestType': 'test_type',
    'Quantity': 'total_thc',
    'Licensee': 'lab',
    'TestPerformedDate': 'date_tested',
    'Comment': 'notes',
    'Med AU': 'medical',
})

# Add a date column.
mi_results['date'] = pd.to_datetime(mi_results['date_tested'])
mi_results['month_year'] = mi_results['date'].dt.to_period('M')

# Exclude outliers.
sample = mi_results.loc[
    (mi_results['total_thc'] > 0) &
    (mi_results['total_thc'] < 100) &
    (mi_results['product_type'] == 'Flower')
]
print('Number of samples:', len(sample))


# === Augment the data. ===

# Augment strain names.
sample['strain_name'] = sample['product_name'].apply(extract_strain_name)
print(sample.sample(10)['strain_name'])


# === Analyze the data. ===

# Add strain first observed at column.
sample['strain_first_observed_at'] = sample.groupby('strain_name')['date'].transform('min')

# Count number of strains tested by month.
sample['month_year'] = sample['month_year'].dt.to_timestamp()
strains_tested_by_month = sample.groupby('month_year').size().reset_index(name='count')
plt.figure(figsize=(10, 5))
plt.plot(strains_tested_by_month['month_year'], strains_tested_by_month['count'], marker='o')
plt.title('Number of Strains Tested By Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# Number of new strains per month.
sample['is_new_strain'] = sample['date'] == sample['strain_first_observed_at']
new_strains_per_month = sample[sample['is_new_strain']].groupby('month_year').size().reset_index(name='count')
plt.figure(figsize=(10, 5))
plt.plot(new_strains_per_month['month_year'], new_strains_per_month['count'], marker='o', color='orange')
plt.title('Number of New Strains Per Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# FIXME:  Number of strains and new strains by adult use and medical.
strains_by_use = sample.groupby(['medical', 'is_new_strain']).size().unstack().fillna(0)
strains_by_use.plot(kind='bar', stacked=True, figsize=(10, 5))
plt.title('Number of Strains and New Strains by Adult Use and Medical')
plt.xlabel('Use Type')
plt.ylabel('Count')
plt.show()



# === WA strains analysis ===

wa_strains = list(wa_results.strain_name.unique())
wa_strains.sort()


# TODO: Parse strain name with AI.
# - Try to find cheapest model
# - Limit costs
# - Limit products if necessary
# - Save prompt, response, model, cost, time
import openai
import os
from dotenv import dotenv_values


# Keep track of costs and time.
total_costs = 0
total_time = 0

# Initialize OpenAI.
config = dotenv_values('../../.env')
openai_api_key = config['OPENAI_API_KEY']
openai_api_key = os.environ['OPENAI_API_KEY']

# Configuration.
model = 'gpt-4'
max_tokens = 500
temperature = 0.0
user = 'cannlytics'
identification_prompt = 'Given the following cannabis product name or text, what is the strain(s) of cannabis, if any? Return your answer as JSON, e.g. when given "Jack Herer 1g Flower" return {"strain_name": "Jack Herer"}.'

# Format the prompt.
strain_name = 'F: Lili Koi R: Rainbow Beltz'
prompt = f'Text: {strain_name}\n\nStrains:'
messages = [
    {'role': 'system', 'content': identification_prompt},
    {'role': 'user', 'content': prompt}
]

# Ask GPT to identify strain name.
print('Prompting model:', model)
start = datetime.now()
response = openai.ChatCompletion.create(
    model=model,
    messages=messages,
    max_tokens=max_tokens,
    temperature=temperature,
    user=user,
)
end = datetime.now()
try:
    content = response['choices'][0]['message']['content']
    predicted_strain = json.loads(content)['strain_name']
except:
    predicted_strain = 'None'
print('Predicted strain:', predicted_strain)
total_tokens = response['usage']['total_tokens']
print('Total tokens:', total_tokens)
cost = response['usage']['total_tokens'] / 1000 * 0.0001
time_elapsed = end - start
print('Time elapsed:', time_elapsed)
total_costs += cost
total_time += time_elapsed.total_seconds()



# TODO: GIF or video of strain abundance over time.




# TODO: Compare product diversity in various states:
# - CT
# - MI
# - WA


wa_strains = pd.Series(wa_results.strain_name.unique())
wa_strains.sort_values(inplace=True)
