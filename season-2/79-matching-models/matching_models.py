"""
Matching Models
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 8/16/2022
Updated: 8/17/2022
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    Recommend products similar to a given product out of a list of
    random products given their lab results by using matching models.

Data Source:

    - Aggregated Cannabis Laboratory Test Results
    URL: <https://huggingface.co/datasets/cannlytics/aggregated-cannabis-test-results>

Setup:

    1. Clone the `dev` branch of Cannlytics.

        ```
        git clone -b dev https://github.com/cannlytics/cannlytics.git
        ```

    2. Install the early release of Cannlytics.

        ```
        cd cannlytics
        pip install .
        ```

"""
# Standard imports.
from ast import literal_eval

# External imports.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Internal imports.
from cannlytics.data.coas import CoADoc
from cannlytics.utils.constants import (
    ANALYSES,
    ANALYTES,
    CODINGS,
    STANDARD_FIELDS,
    STANDARD_UNITS,
)
from cannlytics.utils.utils import (
    convert_to_numeric,
    snake_case,
)

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})


#-----------------------------------------------------------------------
# Get the data!
# URL: <https://huggingface.co/datasets/cannlytics/aggregated-cannabis-test-results>
#-----------------------------------------------------------------------

# Read in lab result data.
datafile = '../../.datasets/lab_results/aggregated-cannabis-test-results-2022-08-09.xlsx'
lab_results = pd.read_excel(datafile, sheet_name='sc_labs_raw_data')

# Identify flower samples from 2022.
flower_types = [
    'flower',
    'Flower',
    'Flower, Inhalable',
    'Flower, Product Inhalable',
    'Flower, Medical Inhalable',
    'Flower, Hemp Flower',
    'Flower, Hemp',
    'Indica, Flower, Inhalable',
    'Sativa, Flower, Inhalable',
]

# Remove timezones from `date_tested`.
# E.g. 2021-01-01 13:54:00-16:00
lab_results['date_tested'] = lab_results['date_tested'].apply(
    lambda x: pd.to_datetime(x).replace(tzinfo=None)
)

# Create a mock retailer that has 100 random products that were
# tested by SC Labs in 2022.
sample = lab_results.loc[
    (lab_results['product_type'].isin(flower_types)) &
    (lab_results['date_tested'] >= pd.to_datetime('2022-01-01'))
]
sample = sample.sample(100, random_state=420)


#-----------------------------------------------------------------------
# Clean the data.
#-----------------------------------------------------------------------

# Clean numeric columns.
sample['total_terpenes'] = pd.to_numeric(
    sample['total_terpenes'].apply(str).apply(convert_to_numeric, strip=True)
)
sample['total_cannabinoids'] = pd.to_numeric(
    sample['total_cannabinoids'].apply(str).apply(convert_to_numeric, strip=True)
)

# Look at the data!
sample['total_terpenes'].hist(bins=20)
sample['total_cannabinoids'].hist(bins=20)
plt.show()

# Clean the results.
for index, row in sample.iterrows():

    # Evaluate the JSON as a dictionary.
    results = literal_eval(row['results'])

    # Create a new clean result.
    clean_results = results.copy()

    # Separate `lod` and `loq`.
    lod_loq_values = [
        x['lodloq'].split(' / ') if x.get('lodloq') else None for x in results
    ]
    for i, values in enumerate(lod_loq_values):
        if values is not None:
            result = clean_results[i]
            result['lod'] = values[0]
            result['loq'] = values[1]
            del result['lodloq']
            clean_results[i] = result
    
    # Standardize fields.
    clean_results = [
        {STANDARD_FIELDS.get(k, k): v for k, v in x.items()}
        for x in clean_results
    ]

    # Add a standard key field.
    for i, values in enumerate(clean_results):
        clean_result = values.copy()
        analyte = snake_case(clean_result['name'])
        analyte = ANALYTES.get(analyte, analyte)
        clean_result['key'] = analyte

        # Apply codings and ensure that certain fields are numeric.
        numeric = ['value', 'limit', 'lod', 'loq', 'margin_of_error']
        for n in numeric:
            value = clean_result.get(n)
            value = CODINGS.get(value, value)
            try:
                value = convert_to_numeric(value, strip=True)
            except TypeError:
                continue
            clean_result[n] = value

        # Add standard units as units are missing.
        analysis = clean_result['analysis']
        analysis = ANALYSES.get(analysis, analysis)
        clean_result['units'] = STANDARD_UNITS.get(analysis)

        # Update the list of results.
        clean_results[i] = clean_result

    # Update the results.
    sample.at[index, 'results'] = clean_results

# Standardize the lab results with CoADoc!
outfile = '../../.datasets/lab_results/sample.xlsx'
parser = CoADoc()
parser.save(sample.to_dict('records'), outfile)
data = pd.read_excel(outfile, sheet_name='Values')

# Future work: Implement the standardization logic in CoADoc!
# data = parser.standardize(data)


#-----------------------------------------------------------------------
# Look at the data
#-----------------------------------------------------------------------

# Keep track of features.
features = pd.DataFrame()

# Set the index.
data.set_index('product_name', inplace=True)

# Look at known terpene ratio.
x, y = 'beta_caryophyllene', 'alpha_humulene'
ratio = data.eval(f'{y} / {x}').rename('ratio')
ratios = ratio.loc[ratio > 0].sort_values(ascending=False)
features['ratio_1'] = ratio
ax = sns.scatterplot(
    data=data,
    x=x,
    y=y,
    hue=ratio,
    s=400,
    palette='viridis_r',
)
top_ratios = ratios[:5]
for line in range(0, data.shape[0]):
    product_name = data.index[line]
    if product_name not in top_ratios:
        continue
    ax.text(
        data[x][line],
        data[y][line],
        data.index[line],
        horizontalalignment='center',
        size='small',
        color='black',
    )
plt.title('beta-Caryophyllene to alpha-Humulene Ratio')
plt.show()

# Look at known terpene ratio.
x, y = 'limonene', 'beta_pinene'
ratio = data.eval(f'{y} / {x}').rename('ratio')
features['ratio_2'] = ratio
ratios = ratio.loc[ratio > 0].sort_values(ascending=False)
ax = sns.scatterplot(
    data=data,
    x=x,
    y=y,
    hue=ratio,
    s=400,
    palette='viridis',
)
top_ratios = ratios[-5:] + ratios[:5]
for line in range(0, data.shape[0]):
    product_name = data.index[line]
    if product_name not in top_ratios:
        continue
    ax.text(
        data[x][line],
        data[y][line],
        data.index[line],
        horizontalalignment='center',
        size='small',
        color='black',
    )
plt.title('beta-Pinene to d-Limonene Ratio')
plt.show()

# Future work: Find the columns where most columns have a value.


#-----------------------------------------------------------------------
# Analyze the data: Given a product, find the most similar product.
#-----------------------------------------------------------------------

from sklearn.neighbors import NearestNeighbors

# Define the features to use.
X = features.dropna(how='all')

# Fit a nearest neighbors model.
model = NearestNeighbors(n_neighbors=4, algorithm='ball_tree')
model.fit(X)

# User input:
strain = 'Pipe Dream' # vs. Trill vs. Mandarin Skunk

# Predict the nearest neighbor of a given strain.
x_hat = pd.DataFrame([X.loc[strain]])
distance, prediction = model.kneighbors(x_hat)
nearest_strains = X.iloc[prediction[0]]
top_strains = list(nearest_strains.index)
print(top_strains)

# Look at known terpene ratio.
x, y = 'limonene', 'beta_pinene'
ratio = data.eval(f'{y} / {x}').rename('ratio')
features['ratio_2'] = ratio
ax = sns.scatterplot(
    data=data,
    x=x,
    y=y,
    hue=ratio,
    s=400,
    palette='viridis',
)
for line in range(0, data.shape[0]):
    product_name = data.index[line]
    if product_name not in top_strains:
        continue
    ax.text(
        data[x][line],
        data[y][line],
        data.index[line],
        horizontalalignment='center',
        size='small',
        color='black',
    )
plt.title('beta-Pinene to d-Limonene Ratio')
plt.show()

# Look at known terpene ratio.
x, y = 'beta_caryophyllene', 'alpha_humulene'
ratio = data.eval(f'{y} / {x}').rename('ratio')
features['ratio_2'] = ratio
ax = sns.scatterplot(
    data=data,
    x=x,
    y=y,
    hue=ratio,
    s=400,
    palette='viridis',
)
for line in range(0, data.shape[0]):
    product_name = data.index[line]
    if product_name not in top_strains:
        continue
    ax.text(
        data[x][line],
        data[y][line],
        data.index[line],
        horizontalalignment='center',
        size='small',
        color='black',
    )
plt.title('beta-Pinene to d-Limonene Ratio')
plt.show()


# Future work: Add more features to use in the nearest neighbor model.
# Look at less obvious factors:
# - moisture_content
# - total_thcv
# - total_cbdv
# - total_cbg
# - producer / distributor
# - date_tested?


# Future work: Estimate different types of matching models.
# mahalanobis(u, v, VI)


#------------------------------------------------------------------------------
# Future work:
#------------------------------------------------------------------------------

# Future work: Create name, value pairs for NLP.
