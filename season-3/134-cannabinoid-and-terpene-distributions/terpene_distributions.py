"""
Terpene Distributions
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 11/7/2023
Updated: 11/9/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# Standard imports:
import json
import os

# External imports:
from cannlytics.utils import convert_to_numeric
from openai import OpenAI
from dotenv import dotenv_values
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress



# === Initialization ===

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})

# Initialize OpenAI
os.environ['OPENAI_API_KEY'] = dotenv_values('.env')['OPENAI_API_KEY']
openai_api_key = os.environ['OPENAI_API_KEY']
client = OpenAI()


# === Get the data ===

# Read TerpLife Labs COA data.
results = pd.read_excel('data/terplife-labs-coa-data-2023-11-15.xlsx')
print('Read {} rows of data.'.format(len(results)))

# Assign date columns.
results['date'] = pd.to_datetime(results['date_tested'])
results['month_year'] = results['date'].dt.to_period('M')
print('Starting date: {}'.format(results['date'].min()))
print('Ending date: {}'.format(results['date'].max()))


# === Look at the lab's statistics ===

# Plot monthly number of tests.
monthly_tests = results.groupby('month_year').size()
monthly_tests.plot()
plt.title('Monthly Number of Tests by TerpLife Labs in Florida')
plt.ylabel('Samples Tested')
plt.xlabel('')
plt.grid(True, which='both', linestyle='--', linewidth=2)
plt.tight_layout()
# plt.xticks(rotation=45)
plt.savefig(f'figures/terplife-monthly-tests.png', bbox_inches='tight', dpi=300)
plt.show()

# Plot types of products tested.
product_type_distribution = results['product_type'].value_counts()
plt.figure(figsize=(10, 6))
plt.pie(product_type_distribution, labels=product_type_distribution.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Product Types')
plt.axis('equal')
plt.show()


# === Look at the producer's statistics ===

# Visualize the number of tests by producer in a bar chart.
producer_test_counts = results['producer'].value_counts()
plt.figure(figsize=(10, 6))
producer_test_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Tests by Producer')
plt.xlabel('Producer')
plt.ylabel('Number of Tests')
plt.xticks(rotation=45)
plt.show()

# TODO: Look at time between date_produced and date_tested.
results['date_produced'] = pd.to_datetime(results['date_produced'])
time_diff = (results['date'] - results['date_produced']).dt.days


print(results['product_size'].unique())
print(results['batch_size'].unique())

# ['3.5g' '100g' '0.5g' '5g' '1g' '0.7g' '7g' '14g' '0.8g' nan '10g' '42g'
#  '25.8g' '25.2g' '28g' '15g' '30g' '95g' '105g' '2g' '26g' '25g' '3.59'
#  '0.3g' '3.58g' '20g' '12g' '24.6g' '24g']
# ['1095.5g' '25000g' '24600g' ... '6937g' '1595.3g' '4564.7g']


# === Future work: Look at geographic statistics ===




# === Look at chemical statistics ===

# Cannabinoids.
cannabinoids = {
    'cannabichromene_cbc': 'cbc',
    'cannabidiol_cbd': 'cbd',
    'cannabidiolic_acid_cbda': 'cbda',
    'cannabidivarin_cbdv': 'cbdv',
    'cannabigerol_cbg': 'cbg',
    'cannabigerolic_acid_cbga': 'cbga',
    'cannabinol_cbn': 'cbn',
    'd_8_tetrahydrocannabinoid_d_8_thc': 'delta_8_thc',
    'd_9_tetrahydrocannabinoid_d_9_thc': 'delta_9_thc',
    'd_9_tetrahydrocannabinolic_acid_thca': 'thca',
    'tetrahydrocannabivarin_thcv': 'thcv',
}
results.rename(columns=cannabinoids, inplace=True)

terpenes = [
    'trans_beta_farnesene',
    'e_caryophyllene',
    'alpha_humulene',
    'd_limonene',
    'alpha_fenchyl_alcohol',
    'alpha_terpineol',
    'linalool',
    'borneol',
    'e_nerolidol',
    'valencene',
    'alpha_bisabolol_l',
    'beta_ocimene',
    'beta_pinene',
    'caryophyllene_oxide',
    'alpha_pinene',
    'beta_myrcene',
    'guaiol',
    'terpinolene',
    'geranyl_acetate',
    'delta_3_carene',
    'alpha_cedrene',
    'alpha_phellandrene',
    'camphene',
    'fenchone',
    'alpha_terpinene',
    'camphor',
    'cedrol',
    'eucalyptol',
    'gamma_terpinene',
    'geraniol',
    'isoborneol',
    'isopulegol',
    'menthol',
    'nerol',
    'p_cymene',
    'pulegone',
    'sabinene',
    'sabinene_hydrate',
    'z_nerolidol',
]

# Count detections for each compound.
# analytes = list(cannabinoids.keys()) + terpenes
analytes = terpenes
results[analytes] = results[analytes].apply(pd.to_numeric, errors='coerce')
occurrences = (results[analytes] > 0).sum()
sorted_occurrences = occurrences.sort_values(ascending=False)
print(sorted_occurrences)

# Draw the chemical abundance curve.
plt.figure(figsize=(18, 8))
plt.barh(sorted_occurrences.index, sorted_occurrences.values)
plt.yscale('log')
plt.title("Abundance of Terpenes in TerpLife Labs Tests")
plt.xlabel('Abundance Rank')
plt.ylabel('Abundance (log)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Calculate beta-pinene and d-limonene ratio.
results['beta_pinene'] = results['beta_pinene'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
results['d_limonene'] = results['d_limonene'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
results['pinene_to_limonene'] = results['beta_pinene'] / results['d_limonene']
results['pinene_to_limonene'] = results['pinene_to_limonene'].replace([np.inf, -np.inf], np.nan)
sample = results.loc[results['pinene_to_limonene'].isna() == False]

# Visualize beta-pinene to d-limonene ratio.
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=sample.loc[(sample['beta_pinene'] < 2) & (sample['d_limonene'] < 2)],
    x='d_limonene',
    y='beta_pinene',
    hue='pinene_to_limonene',
    hue_norm=(0, 1),
    palette="viridis",
    legend=False,
)
plt.title('Beta-Pinene to D-Limonene in TerpLife Labs Tests')
plt.show()


# === Calculate Shannon Diversity Index ===

def calculate_shannon_diversity(df, compounds):
    """Calculate Shannon Diversity Index."""
    diversities = []
    for _, row in df.iterrows():
        # Convert the compound values to numeric and filter those greater than 0
        proportions = [pd.to_numeric(row[compound], errors='coerce') for compound in compounds if pd.to_numeric(row[compound], errors='coerce') > 0]
        proportions = np.array(proportions) / sum(proportions)
        shannon_index = -np.sum(proportions * np.log2(proportions))
        diversities.append(shannon_index)
    return diversities


# Applying the function to your DataFrame
compounds = terpenes
results['shannon_diversity'] = calculate_shannon_diversity(results, compounds)

# Visualize chemical diversity over time.
diversities = results.groupby('month_year')['shannon_diversity'].mean()
diversities.index = diversities.index.to_timestamp()
x_values = diversities.index
slope, intercept, r_value, p_value, std_err = linregress(np.arange(len(x_values)), diversities.values)
plt.figure(figsize=(12, 8))
plt.plot(x_values, diversities.values)
plt.plot(x_values, intercept + slope * np.arange(len(x_values)), color='red', linestyle="--", label="Trend")
plt.xlabel('')
plt.ylabel('Average Chemical Diversity')
plt.title('Diversity of Terpenes in TerpLife Labs Tests')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'figures/terplife-chemical-diversity.png', bbox_inches='tight', dpi=300)
plt.show()
print(f"Slope: {slope}, p-value: {p_value}")
