"""
Cannabis Consumption in Washington State
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 3/2/2022
Updated: 3/11/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description: This script parses quantities and units of measure from the
product name of sales items in the historic Washington State traceability data.

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
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import seaborn as sns
import spacy
from spacy import displacy
from spacy.matcher import Matcher


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


#--------------------------------------------------------------------------
# Clean the data.
#--------------------------------------------------------------------------

# Determine wholesale vs retail transactions.
# Note: It is necessary to find out if there are any medical sales and
# exclude any medical sales from taxes.
data = data.loc[data['sale_type'] != 'wholesale']

# Drop observations with negative prices and prices in the upper quantile.
data = data.loc[
    (data.price_total > 0) &
    (data.price_total < data.price_total.quantile(.95))
]


#--------------------------------------------------------------------------
# Exploratory analysis: Using natural language processing to determine
# quantity and unit of measure for a sample for the data.
#--------------------------------------------------------------------------

# Create natural language processing client.
# Use `en_core_web_sm` for speed and `en_core_web_trf` for accuracy.
# For a blank model, use spacy.blank('en')
nlp = spacy.load('en_core_web_sm')

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
pattern = [{'LOWER': {'IN': ['g', 'gram', 'grams']}}]
pattern = [
    {'LIKE_NUM': True},
    {'LOWER': {'IN': ['g', 'gram', 'grams']}}
]
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
            # displacy.render(doc, style='ent')
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


#--------------------------------------------------------------------------
# Natural language processing functions.
#--------------------------------------------------------------------------

# Calculate weight in milligrams.
# TODO: Add more keys!
# print(list(sample_with_weights['parsed_uom'].unique()))
MILLIGRAMS_PER_UNIT = {
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


def calculate_milligrams(row, quantity_field='parsed_quantity', units_field='parsed_uom'):
    """Calculate the milligram weight of a given observation."""
    mg = MILLIGRAMS_PER_UNIT.get(row[units_field].lower(), 0)
    return mg * row[quantity_field]


def calculate_price_per_mg_thc(row):
    """Calculate the price per milligram of THC for a given observation.
    Source for decarboxylated value conversion factor:
    https://www.conflabs.com/why-0-877/
    """
    try:
        thca = row['cannabinoid_d9_thca_percent'] * 0.877
    except TypeError:
        thca = 0
    thc_mg = row['weight'] * (row['cannabinoid_d9_thc_percent'] + thca) * 0.01
    # TODO: Apply any multiplier!
    try:
        return row['price_total'] / thc_mg
    except ZeroDivisionError:
        return 0


def split_on_letter(string):
    """Split a string at the first letter.
    Credit: C_Z_ https://stackoverflow.com/a/35610194
    License: CC-BY-SA-3.0 https://creativecommons.org/licenses/by-sa/3.0/
    """
    match = re.compile("[^\W\d]").search(string)
    return [string[:match.start()], string[match.start():]]


def parse_weights(nlp_client, row, field='product_name'):
    """Parse weights from an observation's name field."""
    try:
        doc = nlp_client(row[field])
        for entity in doc.ents:
            if entity.label_ == 'QUANTITY':
                parts = split_on_letter(entity.text.replace(' ', ''))
                weight = float(parts[0])
                units = parts[1]
                return (weight, units)
    except (AttributeError, ValueError):
        pass
    return (1, 'ea')


def augment_weights(nlp_client, df, field='product_name'):
    """Augment data with parsed weights from a name field."""
    df = df.assign(parsed_uom='ea', parsed_quantity=1)
    parsed_quantities = df.apply(lambda x: parse_weights(nlp_client, x, field), axis=1)
    df.loc[:, 'parsed_quantity'] = parsed_quantities.map(lambda x: x[0])
    df.loc[:, 'parsed_uom'] = parsed_quantities.map(lambda x: x[1])
    mgs = df.apply(calculate_milligrams, axis=1)
    df = df.assign(weight=mgs)
    thc_prices = df.apply(calculate_price_per_mg_thc, axis=1)
    df = df.assign(price_per_mg_thc=thc_prices)
    return df


#--------------------------------------------------------------------------
# Post-processing testing: Determine the weight in mg and the price per mg of THC
# for each observation.
#--------------------------------------------------------------------------

# Parse quantity and uom with the identified quantity.
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

# Look at the sample with weights.
sample_with_weights = sample.loc[
    (sample['parsed_uom'] != 'ea') &
    (sample['total_cannabinoid_percent'] > 0)
]

# Assign weight in mg.
milligrams = sample_with_weights.apply(calculate_milligrams, axis=1)
sample_with_weights = sample_with_weights.assign(weight=milligrams)

# Assign price per mg of THC.
price_per_mg_thc = sample_with_weights.apply(lambda x: calculate_price_per_mg_thc(x), axis=1)
sample_with_weights = sample_with_weights.assign(price_per_mg_thc=price_per_mg_thc)


#--------------------------------------------------------------------------
# Process the entire dataset.
#--------------------------------------------------------------------------

nlp = spacy.load('en_core_web_trf')
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

# Augment all of the data with weights from the product name.
print('Augmenting weights for all of the data...')
data = augment_weights(nlp, data, field='product_name')
data.to_csv(f'{DATA_DIR}/samples/random-sales-items-2022-03-09a.csv')
print('Augmented all data.')


#--------------------------------------------------------------------------
# Calculate (and visualize!) the average price per mg of THC ($/mg of THC)
# for different product types over time.
#--------------------------------------------------------------------------

# Specify the time period for analysis and look at only observations with
# identified weight and cannabinoid results.
sample = data.loc[
    (data['date'] >= pd.to_datetime('2020-02-01')) &
    (data['date'] <= pd.to_datetime('2021-10-31')) &
    (data['parsed_uom'] != 'ea') &
    (data['total_cannabinoid_percent'] > 0)
]
sample_types = [
    {'key': 'usable_marijuana', 'name': 'Flower'},
    {'key': 'concentrate_for_inhalation', 'name': 'Concentrate'},
]

# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rc('text', usetex=True)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times'], # Times, Palatino, New Century Schoolbook, Bookman
    'font.size': 24,
})

# Visualize the average price by month for each sample type.
fig, ax = plt.(figsize=(15, 7))
colors = sns.color_palette('Set2', n_colors=len(sample_types))
observations = []
for i, sample_type in enumerate(sample_types):

    # Restrict to a given sample type.
    sample_data = sample.loc[
        (sample['intermediate_type'] == sample_type['key']) &
        (sample['price_per_mg_thc'] > 0) &
        (sample['price_per_mg_thc'] < sample['price_per_mg_thc'].quantile(.95))
    ]
    observations.append(len(sample_data))

    # Estimate daily average price.
    daily_data = sample_data.groupby('day')
    avg_price = daily_data.mean()['price_per_mg_thc']
    avg_price.index = pd.to_datetime(avg_price.index)

    # Estimate monthly average price.
    monthly_avg_price = avg_price.groupby(pd.Grouper(freq='M')).mean()

    # Estimate the percent difference from the start to the end.
    percent_change = (monthly_avg_price[-1] - monthly_avg_price[0]) / monthly_avg_price[0] * 100
    direction = '+' if percent_change > 0 else ''

    # Plot monthly prices.
    price_per_gram = monthly_avg_price * 1000
    price_per_gram.plot(
        ax=ax,
        label=sample_type['name'] + ' (%s%.1f\%%)' % (direction, percent_change),
        color=colors[i],
    )
    plt.scatter(
        price_per_gram.index,
        price_per_gram,
        color=colors[i],
        s=100,
    )

plt.title('Average Retail Price per Gram of THC in Washington State', pad=20)
plt.legend(loc='upper right')
plt.xlabel('')
plt.ylabel('Price (\$)')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.figtext(
    x=0.05,
    y=-0.3,
    s="""Data: A random sample of {:,} flower and {:,} concentrate sale items.
Data Source: Washington State cannabis traceability data from {} through {}.
Notes: Average prices are calculated from retail price data before tax.
Quantities are identified from each product's name with natural language processing (NLP).
The top {} of observations by price were excluded as outliers.
THC is measured as the sum of {} and {} after its known
decarboxylated value conversion factor (0.877) has been applied.
""".format(
        observations[0],
        observations[1],
        'February 2020',
        'October 2021',
        '5\%',
        '\delta 9-THC',
        '\delta 9-THCA',
    ),
    fontsize=21,
    ha='left',
)
fig.savefig(
    f'{DATA_DIR}/figures/avg-retail-price-per-g-thc.png',
    format='png',
    dpi=96,
    bbox_inches='tight',
    pad_inches=0.75,
    transparent=False,
)
plt.show()
