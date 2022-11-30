"""
Tobit Models to Predict Microbe Quantities
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 4/2/2022
Updated: 4/2/2022
License: MIT License <https://opensource.org/licenses/MIT>

Data sources:

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd

    - Random Sample of WA Sales Items from 2022-03-16
    https://cannlytics.page.link/cds57

    - WSLCB Limits
    https://app.leg.wa.gov/wac/default.aspx?cite=314-55-102&pdf=true

"""
# External imports.
import numpy as np
import pandas as pd
from tobit import TobitModel

#--------------------------------------------------------------------------
# Read the sample sales data.
# Random sample of sales: https://cannlytics.page.link/cds57
#--------------------------------------------------------------------------

# Specify the fields needed.
lab_result_fields = {
    'lab_result_id': 'string',
    'mme_id': 'string',
    'code': 'string',
    # 'global_for_inventory_id': 'string',
    'microbial_status': 'category',
    'microbial_bile_tolerant_cfu_g': 'float',
    'microbial_pathogenic_e_coli_cfu_g': 'float',
    'microbial_salmonella_cfu_g': 'float',
    'mycotoxin_aflatoxins_ppb': 'float',
    'mycotoxin_ochratoxin_ppb': 'float',
    'status': 'category',
    'intermediate_type' : 'category',
}
lab_result_date_fields = ['created_at']

limits = {
    'mycotoxin_aflatoxins_ppb': {
        'limit': 20,
        'units': 'ug/kg'
    },
    'mycotoxin_ochratoxin_ppb': {
        'limit': 20,
        'units': 'ug/kg'
    },
    'microbial_bile_tolerant_cfu_g': {
        'limit': 1_000,
        'units': 'cfu/g'
    },
    'microbial_pathogenic_e_coli_cfu_g': {
        'limit': 1,
        'units': 'ug/kg'
    },
    'microbial_salmonella_cfu_g': {
        'limit': 1,
        'units': 'ug/kg'
    }
}

# Read in the data from where your data lives.
DATA_DIR = '../.datasets'
DATA_FILE = f'{DATA_DIR}/augmented-washington-state-lab-results.csv'
lab_results = pd.read_csv(
    DATA_FILE,
    usecols=list(lab_result_fields.keys()) + lab_result_date_fields,
    dtype=lab_result_fields,
    parse_dates=lab_result_date_fields,
)


# FIXME: Restrict the time frame.
# start = pd.to_datetime('2021-01-01')
# lab_results['date'] = pd.to_datetime(lab_results['created_at'])
# lab_results = lab_results.loc[lab_results['date'] >= start]


#--------------------------------------------------------------------------
# Augment the lab results with explanatory variables.
#--------------------------------------------------------------------------

# Read licensees data for quick cross-referencing.
licensees = pd.read_csv(
    f'{DATA_DIR}/augmented-washington-state-licensees.csv',
    index_col=0,
    low_memory=False,
)
lab_results = pd.merge(
    left=lab_results,
    right=licensees,
    how='left',
    left_on='mme_id',
    right_on='global_id',
    validate='m:1',
        suffixes=(None, '_y'),
)

# Remove '_y' columns.
column_names = list(lab_results.columns)
drop_columns = ['global_id']
for name in column_names:
    if name.endswith('_y'):
        drop_columns.append(name)
try:
    lab_results.drop(drop_columns, axis=1, inplace=True, errors='ignore')
except TypeError:
    pass


#--------------------------------------------------------------------------
# Analyze lab results that failed for Enterobacteriaceae (entero.)
#--------------------------------------------------------------------------

# Identify all samples that failed for entero.
high_entero = lab_results.loc[
    (lab_results['microbial_bile_tolerant_cfu_g'] > 1_000)
]
print('High entero:', round(len(high_entero) / len(lab_results), 2))

# Identify all flower samples.
flower = lab_results.loc[
    (lab_results['intermediate_type'] == 'flower_lots') | 
    (lab_results['intermediate_type'] == 'flower')
]
flower_failures = flower.loc[
    (flower['microbial_bile_tolerant_cfu_g'] > 10_000)
]
flower_failure_rate = round(len(flower_failures) / len(flower), 2)
print('Percent of failing flower samples:', flower_failure_rate)

# Identify all failed concentrates.
concentrates = lab_results.loc[
    (lab_results['intermediate_type'] == 'non-solvent_based_concentrate')
]
concentrate_failures = concentrates.loc[
    (concentrates['microbial_bile_tolerant_cfu_g'] > 1_000)
]
concentrate_failure_rate = round(len(concentrate_failures) / len(concentrates), 2)
print('Percent of failing flower samples:', concentrate_failure_rate)


#--------------------------------------------------------------------------
# Estimate tobit models with various explanatory variables.
#--------------------------------------------------------------------------

# x = <your set of dependent variables>
# y = <your independent variable>
# cens = <the cens vector as defined in the library documentation>

# tobit  = TobitModel()
# rTobit = tobit.fit(x, y, cens, verbose=False)
# yHat   = rTobit.predict(x)


# yHat   = yHat.reshape(yHat.shape[0],1)
# yMean = np.full((yHat.shape[0],1), y.mean()[0])
# cDet  = np.dot(np.transpose(yHat-yMean), yHat-yMean) / np.dot(np.transpose(y-yMean), y-yMean)
