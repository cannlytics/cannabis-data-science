"""
Strain Analysis
Copyright (c) 2022 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 1/4/2023
Updated: 1/4/2023
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Data Source:

    - Heritability in Cannabis by Chief Seven Turtles
    Sinsemilla Tips Domestic Marijuana Journal
    Compiled by Tom Alexander

"""
# Standard imports:
import os

# External imports:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 21,
})

# Read in the data.
data = pd.read_excel('heritability.xlsx')

# Separate the two methods.
seeds = data.loc[data['method'] == 'seed']
clones = data.loc[data['method'] == 'clone']

# Visualize seed plot differences.
sns.displot(
    data=seeds,
    x='thc',
    hue='plot',
    kind='kde',
    fill=True,
    palette=sns.color_palette('bright')[2:4],
    height=6,
    aspect=1.5
)
plt.title('Variance in THC in Cannabis Propagated by Seed (1988)')
plt.show()

# Visualize clone plot differences.
sns.displot(
    data=clones,
    x='thc',
    hue='plot',
    kind='kde',
    fill=True,
    palette=sns.color_palette('bright')[2:4],
    height=6,
    aspect=1.5
)
plt.title('Variance in THC in Cannabis Propagated by Clone (1988)')
plt.show()

# Visualize seed vs. clone variance.
sns.displot(
    data=data,
    x='thc',
    hue='method',
    kind='kde',
    fill=True,
    palette=sns.color_palette('bright')[:2],
    height=6,
    aspect=1.5
)
plt.title('Variance in THC in Cannabis by Propagation Method (1988)')
plt.show()

# Calculate heritability.
environmental_variation = clones.groupby('plot')['thc'].var().mean()
phenotypic_variation = seeds.groupby('plot')['thc'].var().mean()
genetic_variation = phenotypic_variation - environmental_variation
heritability = genetic_variation / phenotypic_variation
print('Heritability:', round(heritability, 2))
