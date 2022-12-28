"""
Analyze Cannabis Plant Patent Data
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 5/21/2022
Updated: 5/28/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description:

    Analyze plant patent data in relation to observed data
    to determine if plant patents are outliers and other
    potential outliers that would be good patent candidates.

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

    - Effects of cannabidiol in cannabis flower:
    Implications for harm reduction
    URL: <https://pubmed.ncbi.nlm.nih.gov/34467598/>

"""
# Internal imports.
import os

# External imports.
from cannlytics.utils.utils import snake_case # pip install cannlytics
import matplotlib.pyplot as plt
import pandas as pd
import pypatent # pip install pypatent
import seaborn as sns

# Decarboxylation rate. Source: <https://www.conflabs.com/why-0-877/>
DECARB = 0.877


# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})


#------------------------------------------------------------------------------
# Get cannabis plant patent data.
#------------------------------------------------------------------------------

# Read patent details.
datafile = '../.datasets/plant-patents/plant-patents.xlsx'
details = pd.read_excel(datafile, sheet_name='Patent Details')

# Read patent lab results.
datafile = '../.datasets/plant-patents/plant-patents.xlsx'
results = pd.read_excel(datafile, sheet_name='Patent Lab Results')

# Average results by cultivar.
avg_results = results.groupby('strain_name', as_index=False).mean()


#------------------------------------------------------------------------------

import numpy as np

# Look at terpene ratios.
x = 'd_limonene'
y = 'beta_pinene'
sample = avg_results.loc[(~avg_results[x].isna()) & (~avg_results[y].isna())]

fig = sns.scatterplot(x=x, y=y, data=sample, hue='strain_name')
for line in range(0, sample.shape[0]):
    x_value = sample[x].iloc[line]
    y_value = sample[y].iloc[line]
    # if x_value > samplkzzzze[x].quantile(.9) or y_value > sample[y].quantile(.9):
    plt.text(x_value, y_value, sample['strain_name'].iloc[line], horizontalalignment='left', size='small')

xpoints = (0, 1)
ypoints = (0, .25)
plt.plot(xpoints, ypoints)

plt.legend([], [], frameon=False)
plt.title('Beta-pinene to d-limonene ratio of cannabis plant patents')
plt.show()

# Identify the ratios with the highest correlation.
ratio = sample[y].div(sample[x])
sample = sample.assign(pinene_limonene_ratio=ratio)
sample.sort_values('pinene_limonene_ratio', ascending=False, inplace=True)
print(sample[['strain_name', 'pinene_limonene_ratio']])

# Look at the distribution of terpene ratios.
terpenes = list(avg_results.columns)
terpenes.remove('strain_name')
terpenes.remove('duplicate')
terpenes.remove('total_terpenes')
terpenes.remove('total_thc')
terpenes.remove('total_cbd')
terpenes.remove('total_cannabinoids')
for x in terpenes[11:]:
    for y in terpenes[11:]:
        if x == y:
            continue
        sample = avg_results.loc[(~avg_results[x].isna()) & (~avg_results[y].isna())]
        fig = sns.scatterplot(x=x, y=y, data=sample, hue='strain_name')
        for line in range(0, sample.shape[0]):
            x_value = sample[x].iloc[line]
            y_value = sample[y].iloc[line]
            #if x_value > sample[x].quantile(.9) or y_value > sample[y].quantile(.9):
            plt.text(x_value, y_value, sample['strain_name'].iloc[line], horizontalalignment='left', size='small')
        plt.legend([], [], frameon=False)
        plt.title(f'{x} to {y} ratio of cannabis plant patents')
        plt.show()


# Find samples in the top percentile for each ratio.


# Rank the strains by the percentile of their terpene.



#------------------------------------------------------------------------------

# TODO: Statistically identify outliers?
# These would be good strains for patents!
# URL: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
# z_score = 0.6745 * (x - x_bar) / mad
# outlier if z_score > 3.5

#------------------------------------------------------------------------------

# Future work: Get chemical data from ChemSpider, etc.
