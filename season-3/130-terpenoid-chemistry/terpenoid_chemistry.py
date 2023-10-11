"""
Terpenoid Chemistry
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 10/11/2023
Updated: 10/11/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# Standard imports.
import ast
import json

# External imports:
from cannlytics.utils import convert_to_numeric
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})


# === Get the data ===

# Get all cannabis analytes.
dataset = load_dataset('cannlytics/cannabis_analytes', 'all')
analytes = dataset['data'].to_pandas()

# Read CT lab results.
ct_results = pd.read_excel('data/ct-lab-results-2023-10-05.xlsx')
ct_results['date'] = pd.to_datetime(ct_results['date_tested'])
ct_results['month_year'] = ct_results['date'].dt.to_period('M')

# Read MA lab results.
ma_results = pd.read_excel('data/ma-lab-results-2023-09-22.xlsx')
ma_results['date'] = pd.to_datetime(ma_results['date_tested'])
ma_results['month_year'] = ma_results['date'].dt.to_period('M')


# === Look at the data ===

# Set the time period.
period_start = pd.Period((pd.to_datetime('2023-01-01')).strftime('%Y-%m'))
period_end = pd.Period((pd.to_datetime('2023-08-31')).strftime('%Y-%m'))


# 1. Plot products by producer.

# Group by producer and month_year then count products
monthly_producer_counts = ct_results.groupby(['month_year', 'producer']).size().unstack().fillna(0)

# Plot.
plt.figure(figsize=(20, 10))
colors = sns.color_palette('tab10', n_colors=len(monthly_producer_counts.columns))
for idx, producer in enumerate(monthly_producer_counts.columns):
    monthly_producer_counts[producer].plot(label=producer, color=colors[idx], linewidth=2)
plt.title('Monthly Product Counts by Producer')
plt.ylabel('Number of Products')
plt.xlabel('Month-Year')
plt.legend(loc='upper left', ncol=1)
plt.grid(True, which='both', linestyle='--', linewidth=2)
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig(f'figures/ct-products-by-producer.png', bbox_inches='tight', dpi=300)
plt.show()
print(monthly_producer_counts.mean())

# Filter only the data from the last year.
filtered_monthly_producer_counts = monthly_producer_counts[
    (monthly_producer_counts.index > period_start) &
    (monthly_producer_counts.index <= period_end)
]
plt.figure(figsize=(20, 10))
colors = sns.color_palette('tab10', n_colors=len(filtered_monthly_producer_counts.columns))
for idx, producer in enumerate(filtered_monthly_producer_counts.columns):
    filtered_monthly_producer_counts[producer].plot(label=producer, color=colors[idx], linewidth=2)

plt.title('Monthly Product Counts by Producer (Last Year)')
plt.ylabel('Number of Products')
plt.xlabel('Month-Year')
plt.legend(loc='upper left', ncol=1)
plt.grid(True, which='both', linestyle='--', linewidth=2)
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig(f'figures/ct-products-by-producer-2023.png', bbox_inches='tight', dpi=300)
plt.show()
print(filtered_monthly_producer_counts.mean())


# 2. Plot products by product type.

# Create a mask for products of subtype 'flower' and 'preroll'.
flower_mask = ct_results['product_type'].str.contains('flower', case=False, na=False)
preroll_mask = ct_results['product_type'].str.contains('preroll|pre-roll|pre roll|cone|joint', case=False, na=False)

# Assign product subtype.
ct_results['product_subtype'] = None
ct_results.loc[flower_mask, 'product_subtype'] = 'flower'
ct_results.loc[preroll_mask, 'product_subtype'] = 'preroll'

# Plot products by product type over time.
monthly_subtype_counts = ct_results.groupby(['month_year', 'product_subtype']).size().unstack().fillna(0)
plt.figure(figsize=(20, 10))
colors_subtype = sns.color_palette('tab10', n_colors=len(monthly_subtype_counts.columns))
for idx, subtype in enumerate(monthly_subtype_counts.columns):
    monthly_subtype_counts[subtype][:-1].plot(
        label=subtype.title(),
        color=colors_subtype[idx],
        linewidth=2
    )
plt.title('Monthly Flower and Pre-roll Tests in CT')
plt.ylabel('Products Tested')
plt.xlabel('Month')
plt.legend(loc='upper left', ncol=1)
plt.grid(True, which='both', linestyle='--', linewidth=2)
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig(f'figures/ct-products-by-type.png', bbox_inches='tight', dpi=300)
plt.show()

# Filter the DataFrame to consider only the data from the last year
filtered_monthly_subtype_counts = monthly_subtype_counts[
    (monthly_subtype_counts.index > period_start) &
    (monthly_subtype_counts.index <= period_end)
]
plt.figure(figsize=(20, 10))
colors_subtype = sns.color_palette('tab10', n_colors=len(filtered_monthly_subtype_counts.columns))
for idx, subtype in enumerate(filtered_monthly_subtype_counts.columns):
    filtered_monthly_subtype_counts[subtype].plot(
        label=subtype.title(),
        color=colors_subtype[idx],
        linewidth=2,
    )
plt.title('Monthly Number of Flower and Pre-roll in CT in 2023')
plt.ylabel('Products Tested')
plt.xlabel('Month-Year')
plt.legend(loc='upper left', ncol=1)
plt.grid(True, which='both', linestyle='--', linewidth=2)
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig(f'figures/ct-products-by-type-2023.png', bbox_inches='tight', dpi=300)
plt.show()


# 3. Optional: Plot products by product type in MA.

# Identify MA pre-rolls.
flower_mask = ma_results['product_type'].str.contains('flower', case=False, na=False)
preroll_mask = ma_results['product_name'].str.contains('preroll|pre-roll|pre roll|cone|joint', case=False, na=False)
ma_results['product_subtype'] = None
ma_results.loc[flower_mask, 'product_subtype'] = 'flower'
ma_results.loc[preroll_mask, 'product_subtype'] = 'preroll'

# Plot products by product type over time.
monthly_subtype_counts = ma_results.groupby(['month_year', 'product_type']).size().unstack().fillna(0)
plt.figure(figsize=(20, 10))
colors_subtype = sns.color_palette('tab10', n_colors=len(monthly_subtype_counts.columns))
for idx, subtype in enumerate(monthly_subtype_counts.columns):
    monthly_subtype_counts[subtype].plot(
        label=subtype,
        color=colors_subtype[idx],
        linewidth=2,
    )
plt.title('Monthly Number of Products Tested by Type in MA')
plt.ylabel('Number of Products')
plt.xlabel('Month')
plt.legend(loc='upper left', ncol=1)
plt.grid(True, which='both', linestyle='--', linewidth=2)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()


# === Analyze the data ===


def get_result_value(
        results,
        analyte,
        key='key',
        value='value',
    ):
    """Get the value for an analyte from a list of standardized results."""
    # Ensure that the results are a list.
    try:
        result_list = json.loads(results)
    except:
        try:
            result_list = ast.literal_eval(results)
        except:
            result_list = []
    if not isinstance(result_list, list):
        return None

    # Get the value of interest from the list of results.
    result_data = pd.DataFrame(result_list)
    if result_data.empty:
        return None
    try:
        result = result_data.loc[result_data[key] == analyte, value].iloc[0]
    except:
        return 0
    try:
        return convert_to_numeric(result, strip=True)
    except:
        return result
    

# from cannlytics.data.coas import CoADoc

# parser = CoADoc()


# Identify all unique cannabinoids and terpenes.
cannabinoids = []
terpenes = []
for item in ct_results['results']:
    results = ast.literal_eval(item)
    for result in results:
        if result['analysis'] == 'cannabinoids':
            cannabinoids.append(result['name'])
        elif result['analysis'] == 'terpenes':
            terpenes.append(result['name'])
cannabinoids = list(set(cannabinoids))
terpenes = list(set(terpenes))
print('Cannabinoids:', cannabinoids)
print('Terpenes:', terpenes)

# Get the results for each cannabinoid and terpene.
for a in cannabinoids + terpenes:
    print('Augmenting:', a)
    ct_results[a] = ct_results['results'].apply(lambda x: get_result_value(x, a))

# Save the widened results to a file.
outfile = 'data/ct-lab-results-2023-10-05-widened.xlsx'
ct_results.to_excel(outfile, index=False)

# Read the widened results.
# ct_results = pd.read_excel(outfile)

# Future work: Analyze terpenes in the CT data.
