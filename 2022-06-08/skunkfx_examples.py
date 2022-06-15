"""
SkunkFx Examples | Cannabis Data Science
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 6/8/2022
Updated: 6/8/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description:

    Test SkunkFx, the official Cannlytics aroma and effect prediction model.

Data Sources:

    - Data from: Over eight hundred cannabis strains characterized
    by the relationship between their subjective effects, perceptual
    profiles, and chemical compositions
    URL: <https://data.mendeley.com/datasets/6zwcgrttkp/1>
    License: CC BY 4.0. <https://creativecommons.org/licenses/by/4.0/>

Resources:

    - Over eight hundred cannabis strains characterized by the
    relationship between their psychoactive effects, perceptual
    profiles, and chemical compositions
    URL: <https://www.biorxiv.org/content/10.1101/759696v1.abstract>

    - SkunkFx
    URL: <www.skunkfx.com>

"""
# Standard imports.
import json
import requests

# External imports.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})


#------------------------------------------------------------------------------

# Specify the API URL.
BASE = 'https://cannlytics.com/api'

# Get statistics for the `full` model.
url = f'{BASE}/stats/effects'
params = {'model': 'simple'}
response = requests.get(url, params=params)
model_stats = response.json()['data']
print(response.url)
print('Model Stats:\n', model_stats)

# Look at the highest true positive rates.
tpr = pd.Series(model_stats['true_positive_rate'])
tpr.sort_values(ascending=False, inplace=True)
tpr.head(10).plot(kind='bar')
plt.title('Top Effect or Aroma True Positive Rate')
plt.show()

# Look at the highest true positive rates for effects.
effects_tpr = tpr.loc[tpr.index.str.startswith('effect')]
effects_tpr.sort_values(ascending=False, inplace=True)
effects_tpr.head(10).plot(kind='bar')
plt.title('Top Effect True Positive Rate')
plt.show()

# Look at the lowest true positive rates.
effects_tpr.sort_values(ascending=True, inplace=True)
effects_tpr.loc[effects_tpr > 0].head(10).plot(kind='bar')
plt.title('Lowest Effect True Positive Rate (Sensitivity)')
plt.show()

# Look at the false positive rates.
fpr = pd.Series(model_stats['false_positive_rate'])
fpr.sort_values(ascending=True, inplace=True)
fpr.loc[fpr > 0].head(10).plot(kind='bar')
plt.title('Lowest Effect or Aroma False Positive Rate')
plt.show()

# Look at the lowest false positive rates.
fpr = pd.Series(model_stats['false_positive_rate'])
fpr.sort_values(ascending=True, inplace=True)
fpr.loc[
    (fpr > 0) &
    (fpr.index.str.startswith('effect'))
].head(10).plot(kind='bar')
plt.title('Lowest Effect False Positive Rate')
plt.show()

# Look at the highest false positive rates.
fpr = pd.Series(model_stats['false_positive_rate'])
fpr.sort_values(ascending=False, inplace=True)
fpr.loc[
    (fpr > 0) &
    (fpr.index.str.startswith('effect'))
].head(10).plot(kind='bar')
plt.title('Highst Effect False Positive Rate')
plt.show()

#------------------------------------------------------------------------------

# FIXME Get strains by aromas.
# url = f'{BASE}/data/strains'
# aromas = json.dumps(['rose'])
# params = {'limit': 5,'aromas': ['rose']}
# response = requests.get(url, params=params)
# print(response.url)
# assert response.status_code == 200
# data = pd.DataFrame(response.json()['data'])
# print('Found %i strains.' % len(data))
# print(data[['strain_name', 'predicted_aromas']])

# Lemon Alien Dawg, Sugar Black Rose


#------------------------------------------------------------------------------

# Post lab results to get potential effects and aromas.
data = {
    'model': 'full',
    'samples': [
        {
            'strain_name': 'Super Sport',
            'cbc': 0,
            'cbd': 0,
            'cbda': 0,
            'cbg': 0,
            'cbga': 1.58,
            'cbn': 0,
            'delta_8_thc': 0,
            'delta_9_thc': 0.65,
            'thca': 39.29,
            'thcv': 0.21,
            'alpha_bisabolol': 0,
            'alpha_pinene': 1.07,
            'alpha_terpinene': 0,
            'beta_caryophyllene': 0,
            'beta_myrcene': 0.63,
            'beta_pinene': 0.28,
            'camphene': 0,
            'carene': 0,
            'caryophyllene_oxide': 0,
            'd_limonene': 0.17,
            'eucalyptol': 0,
            'gamma_terpinene': 0,
            'geraniol': 0,
            'guaiol': 0,
            'humulene': 0,
            'isopulegol': 0,
            'linalool': 0,
            'nerolidol': 0,
            'ocimene': 0,
            'p_cymene': 0,
            'terpinene': 0,
            'terpinolene': 0,
        },
    ]
}
url = f'{BASE}/stats/effects'
response = requests.post(url, json=data)
data = response.json()['data']
model_stats = data['model_stats']
samples = pd.DataFrame(data['samples'])

# Collect outcomes.
outcomes = pd.DataFrame()
for index, row in samples.iterrows():
    for i, key in enumerate(row['predicted_effects']):
        tpr = round(model_stats['true_positive_rate'][key] * 100, 2)
        fpr = round(model_stats['false_positive_rate'][key] * 100, 2)
        title = key.replace('effect_', '').replace('_', ' ').title()
        outcomes = pd.concat([outcomes, pd.DataFrame([{
            'tpr': tpr,
            'fpr': fpr,
            'name': title,
            'strain_name': index,
        }])])
