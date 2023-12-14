"""
Abundance Analysis
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 10/31/2023
Updated: 11/1/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})


# === Read the data ===

# Read WA lab results.
wa_results = pd.read_csv('./data/wa-lab-results-latest.csv')
wa_results['date'] = pd.to_datetime(wa_results['date_tested'], format='mixed')
wa_results['month_year'] = wa_results['date'].dt.to_period('M')
wa_results['year'] = wa_results['date'].dt.strftime('%Y')


# === Clean the data ===

# Remove observations with missing strain name.
data = wa_results.copy()
exclude = ['No Strain', 'Mixed', '5', 'Unknown', 'Sativa', 'Indica',
           '1.12412E+16', 'MIXED', 'Mix']
data = data.loc[~data['strain_name'].isin(exclude)]

# Escape special characters in strain names
data['strain_name'] = data['strain_name'].apply(lambda s: s.translate(str.maketrans({"$":  r"\$"})))

# Restrict the timeframe.
data = data[
    (data['date'] >= datetime(2021, 1, 1)) &
    (data['date'] < datetime(2024, 1, 1))
]


# === Abundance ===

# Calculate abundance.
strains = data.groupby(['strain_name']).size().reset_index(name='abundance')
strains = strains.sort_values(by='abundance', ascending=False).reset_index(drop=True)

# Draw the abundance curve.
plt.bar(strains.index, strains['abundance'])
plt.yscale('log')
plt.title("Abundance of Strains in WA")
plt.xlabel('Abundance Rank')
plt.ylabel('Abundance (log)')
plt.xticks([])
plt.tight_layout()
plt.show()

# Visualize most abundant strains.
sample = strains.head(20)
plt.bar(sample.index, sample['abundance'])
plt.yscale('log')
plt.xticks(sample.index, sample['strain_name'], rotation=90, ha='right')
plt.title("Abundance of Strains in WA")
plt.xlabel('Abundance Rank')
plt.ylabel('Abundance (log)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# === Relative abundance ===

# Calculate relative abundance.
total_abundance = strains['abundance'].sum()
strains['relative_abundance'] = (strains['abundance'] / total_abundance) * 100
sample = strains.sort_values(by='relative_abundance', ascending=False).reset_index(drop=True)

# Draw the relative abundance curve.
plt.bar(strains.index, strains['relative_abundance'])
plt.yscale('log')
plt.title("Relative Abundance of Strains in WA")
plt.xlabel('Abundance Rank')
plt.ylabel('Relative Abundance (log)')
plt.xticks([])
plt.tight_layout()
plt.show()

# Draw the relative abundance curve of the top 20 strains.
sample = sample.head(20)
plt.bar(sample.index, sample['relative_abundance'])
plt.yscale('log')
plt.xticks(sample.index, sample['strain_name'], rotation=90, ha='right')
plt.title("Relative Abundance Curve of Strains")
plt.xlabel('Strain Rank')
plt.ylabel('Relative Abundance (log)')
plt.tight_layout()
plt.show()


# === Visualize abundance over time ===

# Calculate product abundance over time.
panel = data.groupby(['year', 'strain_name']).size().reset_index(name='abundance')

# Calculate relative abundance over time.
panel['relative_abundance'] = panel.groupby('year')['abundance'] \
                                .apply(lambda x: (x / x.sum()) * 100).reset_index(level=0, drop=True)

# Sorting the values based on year and relative_abundance.
panel = panel.sort_values(by=['year', 'relative_abundance'], ascending=[True, False]).reset_index(drop=True)

# Identify data by year.
top_n = 20
data_2021 = panel[panel['year'] == '2021'].sort_values(by='relative_abundance', ascending=False).head(top_n)
data_2022 = panel[panel['year'] == '2022'].sort_values(by='relative_abundance', ascending=False).head(top_n)
data_2023 = panel[panel['year'] == '2023'].sort_values(by='relative_abundance', ascending=False).head(top_n)

# Visualize abundance over time.
plt.bar(data_2023['strain_name'], data_2023['relative_abundance'], alpha=0.6, label='2023')
plt.bar(data_2022['strain_name'], data_2022['relative_abundance'], alpha=0.6, label='2022')
plt.bar(data_2021['strain_name'], data_2021['relative_abundance'], alpha=0.6, label='2021')
plt.ylabel('Relative Abundance (%)')
plt.xlabel('')
plt.title('Relative Abundance of Strains in WA Over Time')
plt.legend(title='Year')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
