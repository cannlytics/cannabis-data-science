"""
Cultivar Prediction
Copyright (c) 2023 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 3/15/2023
Updated: 3/15/2023
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

References:

    - Cannabis Tests
    URL: <https://huggingface.co/datasets/cannlytics/cannabis_tests>

    - MCR Labs Results 2023-02-06
    URL: <https://cannlytics.page.link/mcr-lab-results-2023-02-06>

Data Sources:

    - MCR Labs Test Results
    URL: <https://reports.mcrlabs.com>

"""
# Internal imports:
import os
from typing import List

# External imports:
from cannlytics.data.coas import CoADoc
import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel
import seaborn as sns
import statsmodels.api as sm


#------------------------------------------------------------------------------
# Data: Identify common cultivars given MCR Labs test results.
#------------------------------------------------------------------------------

# Read in MCR Labs results.
data_dir = '.datasets/'
filename = 'mcr-lab-results-2023-02-06T14-35-24.xlsx'
ma_data = pd.read_excel(os.path.join(data_dir, filename))

# Standardize lab results.
parser = CoADoc()
standard_results = parser.standardize(ma_data, how='long')

varieties = [
    'og',
    'kush',
    'haze',
]
top_terpenes = [
    'beta_myrcene',
    'beta_ocimene',
    'alpha_pinene',
    'beta_caryophyllene',
    'beta_pinene',
    'alpha_humulene',
    'd_limonene',
    'eucalyptol',
    'linalool',
    'alpha_bisabolol',
    'guaiol',
    'caryophyllene_oxide',
    'camphene',
    'delta_3_carene',
    'alpha_ocimene',
    'p_cymene',
    'cis_nerolidol',
    'trans_nerolidol',
    'alpha_terpinene',
    'gamma_terpinene',
    'terpinolene',
    'isopulegol',
    'geraniol'
]

# Create a wide panel of strain results.
panel = pd.DataFrame()
for variety in varieties:

    # Identify all flower samples of a particular variety.
    strain = standard_results.loc[
        (standard_results['product_type'] == 'flower') &
        (standard_results['product_name'].str.contains(variety, case=False)) &
        (standard_results['analysis'] == 'terpenes')
    ]

    # Arrange the strain's data for the wide panel.
    data = strain.groupby(['product_name', 'date_tested'], as_index = False)
    data = data.first()
    data = data.drop(['key', 'lod', 'loq', 'name', 'value'], axis=1)
    data['cultivar'] = variety
    for index, values in data.iterrows():

        # Get the terpene lab test results.
        lab_test = strain.loc[
            (strain['product_name'] == values['product_name']) &
            (strain['date_tested'] == values['date_tested'])
        ]

        # Get the result for each terpene.
        for terpene in top_terpenes:
            try:
                result = lab_test.loc[lab_test['key'] == terpene].iloc[0]['value']
            except IndexError:
                result = 0
            data.loc[index, terpene] = result
    
    # Add the strain data to the pane.
    panel = pd.concat([panel, data], ignore_index=True, axis=0)


#------------------------------------------------------------------------------
# Model: Create a model for strain data.
#------------------------------------------------------------------------------

class StrainResult(BaseModel):
    """A data class representing a cannabis strain."""
    key: str
    avg: float
    min: float
    max: float
    median: float
    std: float


class Strain(BaseModel):
    """A data class representing a cannabis strain."""
    cultivar: str
    strain_name: str
    first_produced_at: str
    first_produced_by: str
    first_produced_state: str
    number_of_tests: int
    results: List[StrainResult]


#------------------------------------------------------------------------------
# Analysis: Calculate summary statistics for each cultivar.
#------------------------------------------------------------------------------

# Calculate number of tests by cultivar.
panel['cultivar'].value_counts()

# Find the average by cultivar.
cultivars = panel.groupby('cultivar')
for terpene in top_terpenes:
    print('\n', cultivars[terpene].mean())

# Find the min by cultivar.
cultivars = panel.groupby('cultivar')
for terpene in top_terpenes:
    print('\n', cultivars[terpene].min())

# Find the max by cultivar.
cultivars = panel.groupby('cultivar')
for terpene in top_terpenes:
    print('\n', cultivars[terpene].max())

# Find the median by cultivar.
cultivars = panel.groupby('cultivar')
for terpene in top_terpenes:
    print('\n', cultivars[terpene].median())

# Find the standard deviation by cultivar.
cultivars = panel.groupby('cultivar')
for terpene in top_terpenes:
    print('\n', cultivars[terpene].std())


#------------------------------------------------------------------------------
# Prediction: Predict product cultivar given terpene lab results.
#------------------------------------------------------------------------------

# Define dependent and independent variables.
y = panel['cultivar']
X = panel[top_terpenes]

# Fit the model.
model = sm.MNLogit(y, X)
results = model.fit()

# Print the results.
print(results.summary())

# Measure the accuracy of the model.
success = 0
for index, values in panel.iterrows():
    # x = values[top_terpenes]
    x = X.iloc[index]
    prediction = results.predict(x)
    outcome = prediction.idxmax(axis=1)
    predicted_cultivar = None
    for variety_id, variety in enumerate(varieties):
        if outcome.values[0] == variety_id:
            predicted_cultivar = variety
    actual = values['cultivar']
    print('\n', values['product_name'])
    print(
        'Actual:', actual,
        'Predicted:', predicted_cultivar,
    )

    if actual == predicted_cultivar:
        success += 1

# Calculate the accuracy of the model.
accuracy = success / len(panel)
print(f'Accuracy: {round(accuracy * 100, 2)}%')
