"""
Cannabis Consumption in Washington State
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 3/2/2022
Updated: 3/8/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script analyzes yields in Washington State.

Data sources:

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd

    - Random Sample of WA Sales Items from 2022-02-16
    https://cannlytics.page.link/cds53

Notes:

    Install SpaCy:
        1. pip install spacy
        2. python -m spacy download en_core_web_sm
           python -m spacy download en_core_web_trf

"""
# Standard imports.
import re

# External imports.
import matplotlib.pyplot as plt
import pandas as pd

# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
})


#--------------------------------------------------------------------------
# Read the sample sales data.
# Random sample of sales: https://cannlytics.page.link/cds53
#--------------------------------------------------------------------------

# Specify where your data lives.
DATA_DIR = 'D:\\leaf-data'

# Read in the data.
DATA_FILE = f'{DATA_DIR}/samples/random-sales-items-2022-02-16.csv'
data = pd.read_csv(DATA_FILE)

# Bonus: Read in the data if you're a Saturday Morning Statistics attendee.
# DATA_FILE = f'{DATA_DIR}/samples/random-sales-items-2022-03-05.csv'
# data = pd.read_csv(DATA_FILE)


#------------------------------------------------------------------------------
# Get the retail city and county.
# Licensees data: https://cannlytics.com/data/market/augmented-washington-state-licensees
# Augment yourself: https://github.com/cannlytics/cannabis-data-science/blob/main/2022-01-26/geocode_licensees.py
#------------------------------------------------------------------------------

# Augment fields from the licensees data.
licensee_fields = {
    'global_id': 'string',
    'city': 'string',
    'county': 'string',
}
licensees = pd.read_csv(
    f'{DATA_DIR}/augmented/augmented-washington-state-licensees.csv',
    usecols=list(licensee_fields.keys()),
    dtype=licensee_fields,
)
data = pd.merge(
    left=data,
    right=licensees,
    how='left',
    left_on='mme_id',
    right_on='global_id',
)
data.drop(['global_id'], axis=1, inplace=True, errors='ignore')
print('Augmented the sales data with city and county.')


#--------------------------------------------------------------------------
# Clean the data.
#--------------------------------------------------------------------------

# Determine wholesale vs retail transactions.
# Note: It is necessary to find out if there are any medical sales and
# exclude any medical sales from taxes.
data = data.loc[data['sale_type'] != 'wholesale']

# Drop observations with negative prices and prices in the upper quantile.
data = data.loc[data.price_total > 0]
data = data[data.price_total < data.price_total.quantile(.95)]

# Add a date column.
data['date'] = pd.to_datetime(data['created_at'])
data['day'] = data['date'].dt.date

# Add quarter (e.g. 2022Q1) field.
data['quarter'] = data.date.dt.to_period('Q')

# Clean the city name and county.
data['city'] = data['city'].str.title()
data['county'] = data['county'].str.title().str.replace(' County', '')


#--------------------------------------------------------------------------
# Natural language processing to determine quantity and unit of measure.
#--------------------------------------------------------------------------

import spacy
from spacy import displacy
from spacy.matcher import Matcher 

# Create natural language processing client.
nlp = spacy.load('en_core_web_sm') # Or en_core_web_trf for accuracy.

# Optional: Inspect the pipeline.
print(nlp.pipe_names)

# Examples:
# doc = nlp('(Dabulous - 1 g )')
# doc = nlp('Blue Cookies Jar 1 g')
# displacy.render(doc, style='ent')

# Create a random sample for development.
sample_size = 20
sample = data.sample(sample_size, random_state=420)

# Look at named entities in product names.
for index, values in sample.iterrows():
    try:
        doc = nlp(values['product_name'])
        displacy.render(doc, style='ent')
    except ValueError:
        pass

# Print identified quantities.
identified = 0
for index, values in sample.iterrows():
    try:
        doc = nlp(values['product_name'])
        for entity in doc.ents:
            if entity.label_ == 'QUANTITY':
                print('Text:', entity.text)
                identified += 1
    except ValueError:
        pass
identification_rate = identified / sample_size
print('Identified: %.0f%%' % round(identification_rate * 100))

# Example patterns.
pattern = [{'LOWER': 'g'}]
# pattern = [{'LOWER': {'IN': ['g', 'gram', 'grams']}}]
# pattern = [
#     {'LIKE_NUM': True},
#     {'LOWER': {'IN': ['g', 'gram', 'grams']}}
# ]
# pattern = [
#     {'LIKE_NUM': True},
#     {'LOWER': {'IN': ['x', 'pk', 'pack']}}
# ]

# Create a matching mechanism.
matcher = Matcher(nlp.vocab)
matcher.add('identify_weight', [pattern])

# Match namespaces in the sample.
for index, values in sample.iterrows():
    try:
        doc = nlp(values['product_name'])
        matches = matcher(doc)
        for match_id, start, end in matches:
            displacy.render(doc, style='ent')
            print('Match found:', doc[start:end].text)
    except ValueError:
        pass

# Add custom pipeline component(s).
patterns = [
    {
        'label': 'MULTIPLIER',
        'pattern': [
            {'LIKE_NUM': True},
            {'LOWER': {'IN': [
                '(2)', 'x', 'pk', 'pack', 'packs'
            ]}}
        ],
    },
    {
        'label': 'QUANTITY',
        'pattern': [
            {'LIKE_NUM': True},
            {'LOWER': {'IN': [
                'g', 'gr', 'gram', 'grams', 'teenth', 'sixteenth', 'eighth',
                'quarter', 'ounce', 'ounces', 'oz', 'pound', 'lb', 'mg', 'kg',
                'milligram', 'milligrams', 'kilogram', 'kilograms',
                '1\/8 oz'
            ]}}
        ],
    },
]
try:
    ruler = nlp.add_pipe('entity_ruler', before='ner')
except ValueError:
    nlp.remove_pipe('entity_ruler')
    ruler = nlp.add_pipe('entity_ruler', before='ner')
ruler.add_patterns(patterns)

# Look at named entities in product names after customization.
for index, values in sample.iterrows():
    try:
        doc = nlp(values['product_name'])
        displacy.render(doc, style='ent')
    except ValueError:
        pass

# Print identified quantities after customization.
identified = 0
for index, values in sample.iterrows():
    try:
        doc = nlp(values['product_name'])
        for entity in doc.ents:
            if entity.label_ == 'QUANTITY':
                print('Text:', entity.text)
                identified += 1
    except ValueError:
        pass
identification_rate = identified / sample_size
print('Identified: %.0f%%' % round(identification_rate * 100))


def split_on_letter(s):
    """
    Credit: C_Z_ https://stackoverflow.com/a/35610194
    License: CC-BY-SA-3.0 https://creativecommons.org/licenses/by-sa/3.0/
    """
    match = re.compile("[^\W\d]").search(s)
    return [s[:match.start()], s[match.start():]]


# Parse quantity and uom now once the quantity string is found.
sample['parsed_uom'] = 'ea'
sample['parsed_quantity'] = 1
for index, values in sample.iterrows():
    try:
        doc = nlp(values['product_name'])
        for entity in doc.ents:
            if entity.label_ == 'QUANTITY':
                identified += 1
                texts = split_on_letter(entity.text.replace(' ', ''))
                sample.loc[index, 'parsed_quantity'] = float(texts[0])
                sample.loc[index, 'parsed_uom'] = texts[1]
    except ValueError:
        pass
# print(sample[['parsed_quantity', 'parsed_uom']])

# Look at the sample with weights.
sample_with_weights = sample.loc[
    (sample['parsed_uom'] != 'ea') &
    (sample['total_cannabinoid_percent'] > 0)
]

# Calculate weight in milligrams.
# print(list(sample_with_weights['parsed_uom'].unique()))
milligrams_per_unit = {
    'g': 1000,
    'gr': 1000,
    'gram': 1000,
    'grams': 1000,
    'teenth': 1771.845,
    'sixteenth': 1771.845,
    'eighth': 3543.69,
    '1\/8 oz': 3543.69,
    'quarter': 7087.381,
    'ounce': 28_349.52,
    'ounces': 28_349.52,
    'oz': 28_349.52,
    'pound': 28_349.52 * 16,
    'lb': 28_349.52 * 16,
    'mg': 1,
    'milligram': 1,
    'milligrams': 1,
    'kg': 100_000,
    'kilogram': 100_000,
    'kilograms': 100_000,
}


def calculate_milligrams(row):
    """Calculate the milligram weight of a given observation."""
    mg = milligrams_per_unit.get(row['parsed_uom'], 0)
    return mg * row['parsed_quantity']


# Assign weight in mg.
milligrams = sample_with_weights.apply(lambda x: calculate_milligrams(x), axis=1)
sample_with_weights = sample_with_weights.assign(weights=milligrams)


def calculate_price_per_mg_thc(row):
    """Calculate the price per milligram of THC for a given observation.
    TODO: Source for carboxyl conversion factor.
    """
    try:
        thca = row['cannabinoid_d9_thca_percent'] * 0.877
    except TypeError:
        thca = 0
    thc_mg = row['weight'] * (row['cannabinoid_d9_thc_percent'] + thca * 0.01)
    # TODO: Apply any multiplier!
    return row['price_total'] / thc_mg


# Assign price per mg of THC.
price_per_mg_thc = sample_with_weights.apply(lambda x: calculate_price_per_mg_thc(x), axis=1)
sample_with_weights = sample_with_weights.assign(price_per_mg_thc=price_per_mg_thc)


#--------------------------------------------------------------------------
# TODO: Parse quantity and unit of measure for all of the data.
#--------------------------------------------------------------------------

# 2 Pk .75 gr Pre-Roll
# 01g
# common_weights = {
#     '1g': {'qty': 1, 'uom': 'gm'},
#     '1.0g': {'qty': 1, 'uom': 'gm'},
#     '1.0 g': {'qty': 1, 'uom': 'gm'},
#  .75 gr
#     '.75g': {'qty': 0.75, 'uom': 'gm'},
#     '.8g': {'qty': 0.8, 'uom': 'gm'},
#     '1.5g': {'qty': 1, 'uom': 'gm'},
#     '2g': {'qty': 2, 'uom': 'gm'},
#     '2.5g': {'qty': 2.5, 'uom': 'gm'},
#     '3.5g': {'qty': 3.5, 'uom': 'gm'},
#     '3.50 grams': {'qty': 3.5, 'uom': 'gm'},
#     '3.5 gram': {'qty': 3.5, 'uom': 'gm'},
#     'eighth': {'qty': 3.5, 'uom': 'gm'},
#     '7.5g': {'qty': 7.5, 'uom': 'gm'},
#     # 10PK=100MG | 10pack 100mgTHC
#     # Joints .5g (2)
#     # 0.5g x 2
#     # 1\/8 oz
# }
# common_multipliers = {
#     'x 2': 2,
#     # '(2)': 2,
#     '2 Pk': 2,
#     '2pack': 2,
#     '3pk': 3,
#     '7 x ': 7,
#     '10-pack': 10,
# }

# Ratios:
# 5:1


#--------------------------------------------------------------------------
# TODO: Calculate price per total cannabinoids ($/mg)
#--------------------------------------------------------------------------

# Look at price per mg of THC (or total cannabinoids) in different sample types.


# Identify the time period for analysis.
# start = '2021-01-01'
# end = '2021-10-31'
# data = data.loc[
#     (data['date'] >= pd.to_datetime(start)) &
#     (data['date'] <= pd.to_datetime(end))
# ]
# print('Data cleaned and limited to the period of analysis.')
# print('{:,} observations.'.format(len(data)))

# # ARCH and GARCH.
# sample_type = 'concentrate_for_inhalation'
# sample_data = data.loc[data.intermediate_type == sample_type]
# daily_data = sample_data.groupby('day')

# avg_price = daily_data.mean()['price_total']
# avg_price.index = pd.to_datetime(avg_price.index)

# # Estimate the total tax paid by month.
# monthly_avg_price = avg_price.groupby(pd.Grouper(freq='M')).mean()
# monthly_avg_price.plot()

# # Estimate the total tax paid by month.
# std_price = daily_data.std()['price_total']
# std_price.index = pd.to_datetime(std_price.index)
# monthly_std_price = std_price.groupby(pd.Grouper(freq='M')).mean()
# monthly_std_price.plot()


#--------------------------------------------------------------------------
# Yield appears to be the name of the game. Does the amount a producer produces
# affect the number of periods a producer has operated or if a producer has exited.
#--------------------------------------------------------------------------

# 1. Measure yields.


# 2. Determine when producers are operating.


# import pymc3 as pm
# Bayesian linear regression.
# X, y = linear_training_data()
# with pm.Model() as linear_model:
#     parameter_belief = pm.Normal('weights', mu=0, sigma=1)
#     variance_belief = pm.Gamma('noise', alpha=2, beta=1)
#     y_observed = pm.Normal(
#         'y_observed',
#         mu=X @ parameter_belief,
#         sigma=variance_belief,
#         observed=y,
#     )
#     prior = pm.sample_prior_predictive()
#     posterior = pm.sample()
#     posterior_pred = pm.sample_posterior_predictive(posterior)
