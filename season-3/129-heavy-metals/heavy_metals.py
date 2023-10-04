"""
Heavy Metal Analysis
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 10/3/2023
Updated: 10/4/2023
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    This script analyzes heavy metal data from Maryland and
    Washington for cannabis products tested for quality control.

References:

    - Maryland heavy metal limits: <https://mmcc.maryland.gov/Documents/2022_PDF_Files/Policy%20Committee%20Meeting%20Documents/MMCC%20TA-%20updatesrev%204%20draft-POLICY.pdf>
    - Washington heavy metal limits: <https://app.leg.wa.gov/wac/default.aspx?cite=246-70-050>

"""
# Standard imports.
import ast
import json

# External imports.
from cannlytics.utils import convert_to_numeric
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

# Read the lab results.
md_results = pd.read_csv('data/md-lab-results-2023-09-27.csv')

# Get the flower results.
print(md_results['product_subtype'].unique().tolist())
flower = md_results[md_results['product_subtype'] == 'Raw Plant Material']


# === Lab Analysis ===

# Value counts for each TestingFacilityId
counts = md_results['TestingFacilityId'].value_counts()
plt.figure(figsize=(10, 6))
counts.plot(kind='bar', color='skyblue')
plt.title('Sample Count by Testing Facility')
plt.xlabel('Testing Facility ID')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# === Cannabinoid Analysis ===

# Distribution Plots
cannabinoids = ['CBD', 'CBDA', 'CBG', 'CBN', 'THC', 'THCA']
for compound in cannabinoids:
    sample = flower[(flower[compound] > 0) & (flower[compound] < 100)]
    sns.histplot(sample[compound], bins=50, kde=True)
    plt.title(f'Distribution of {compound}')
    if compound != 'THCA':
        plt.xlim(0, 2)
    plt.show()

# Calculate total THC and CBD.
sample = flower[(flower[['CBD', 'CBDA', 'THC', 'THCA']] <= 100).all(axis=1)]
sample['Total THC'] = sample['THC'] + 0.877 * sample['THCA']
sample['Total CBD'] = sample['CBD'] + 0.877 * sample['CBDA']

# THC to CBD Ratio Scatter Plot
sns.scatterplot(x=sample['Total CBD'], y=sample['Total THC'])
plt.title('THC to CBD Ratio')
plt.xlabel('Total CBD (%)')
plt.ylabel('Total THC (%)')
plt.savefig(f'figures/thc_to_cbd_md.png', bbox_inches='tight', dpi=300)
plt.show()

# THC to CBG Ratio Scatter Plot
sns.scatterplot(x=sample['CBG'], y=sample['Total THC'])
plt.title('THC to CBG Ratio')
plt.xlabel('CBG (%)')
plt.ylabel('Total THC (%)')
plt.xlim(0, 5)
plt.savefig(f'figures/thc_to_cbg_md.png', bbox_inches='tight', dpi=300)
plt.show()

# Distribution of CBN
sns.scatterplot(x=sample['CBN'], y=sample['Total THC'])
plt.title('THC to CBN Ratio')
plt.xlabel('CBN (%)')
plt.ylabel('Total THC (%)')
plt.xlim(0, 5)
plt.savefig(f'figures/thc_to_cbn_md.png', bbox_inches='tight', dpi=300)
plt.show()


# === Terpene Analysis ===

# Identify outliers.
outliers = flower[flower['Terpenes'] > 6]
print(outliers)

# Filter outliers.
sample = flower[flower['Terpenes'] < 10]
print(sample['Terpenes'].describe())

# Plot the distribution of terpene content.
sns.histplot(sample['Terpenes'], bins=50, kde=True)
plt.xlabel('Terpene Content (%)')
plt.ylabel('Count')
plt.title('Distribution of Terpene Content in Raw Plant Material')
plt.savefig(f'figures/terpene_md_distribution.png', bbox_inches='tight', dpi=300)
plt.show()
print('Mean:', round(sample['Terpenes'].mean(), 2))


# === Heavy metal analysis ===

# Summarize silver data.
silver_data = md_results[md_results['Silver'] > 0]
silver_fail = silver_data[silver_data['TestPassed'] == 'Fail'].shape[0]
silver_pass = silver_data[silver_data['TestPassed'] == 'Pass'].shape[0]
print(f'Number of failed tests with silver: {silver_fail}')
print(f'Number of passed tests with silver: {silver_pass}')

# Distribution of Silver.
filtered_silver = silver_data[(silver_data['Silver'] >= 0) & (silver_data['Silver'] <= 1)]
sns.histplot(filtered_silver['Silver'], bins=100)
plt.xlabel('Silver (ppm)')
plt.ylabel('Count')
plt.title('Distribution of Silver in Maryland Flower (0 to 1 ppm)')
plt.show()

# Define heavy metals.
md_limits = {
    'Lead': {
        'Inhalation': 1.0,
        'Oral': 0.5
    },
    'Arsenic': {
        'Inhalation': 0.4,
        'Oral': 1.5
    },
    'Mercury': {
        'Inhalation': 0.2,
        'Oral': 3.0
    },
    'Cadmium': {
        'Inhalation': 0.4,
        'Oral': 0.5
    },
    'Chromium': {
        'Inhalation': 0.6,
        'Oral': 1070.0
    }
}
md_metals = [
    'Arsenic',
    'Barium',
    'Cadmium',
    'Chromium',
    'Lead',
    'Mercury',
    'Selenium',
    'Silver',
]

# Analyze all metals.
for metal in md_metals:
    print(f'{metal} Analysis')
    
    # Association with Testing Outcomes
    metal_data = flower[flower[metal] > 0]
    try:
        limit = md_limits[metal]['Inhalation']
    except:
        limit = None
    metal_fail = metal_data[metal_data[metal] >= limit].shape[0]
    metal_pass = metal_data[metal_data[metal] < limit].shape[0]

    # Calculate Failure Rate
    fail_rate = (metal_fail / len(flower)) * 100
    print(f"Number of Fail tests with {metal}: {metal_fail}")
    print(f"Number of Pass tests with {metal}: {metal_pass}")
    print(f"Failure rate for {metal}: {fail_rate:.2f}%")
    
    # Distribution of Metal (You can adjust the range if necessary)
    filtered_metal = metal_data[(metal_data[metal] >= 0) & (metal_data[metal] <= 2)]
    sns.histplot(filtered_metal[metal], bins=100)
    if limit:
        plt.axvline(limit, color='r', linestyle='--', label='Limit (Inhalation)')
    plt.xlabel(f'{metal} (ppm)')
    plt.ylabel('Count')
    plt.title(f'Distribution of {metal} in Maryland Flower (0 to 1 ppm)')
    plt.savefig(f'figures/{metal}_md_distribution.png', bbox_inches='tight', dpi=300)
    plt.show()

# Look at heavy metal detections by year.
for metal in md_metals:
    metal_data = flower[flower[metal] > 0]
    heavy_metal_counts_by_year = metal_data.groupby('TestYear')[metal].count()
    heavy_metal_counts_by_year.plot(kind='bar', color='skyblue')
    plt.title(f'Heavy Metal Detections of {metal} by Year in MD Cannabis Flower')
    plt.ylabel('Detections')
    plt.xlabel('Year')
    plt.savefig(f'figures/{metal}_md_detections_by_year.png', bbox_inches='tight', dpi=300)
    plt.show()


# === State-by-state comparison ===

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


# Compare with WA heavy metal data.
wa_results = pd.read_excel('data/wa-lab-results-2023-08-04.xlsx')
wa_results['heavy_metals'] = wa_results['heavy_metals'].apply(lambda x: np.nan if len(ast.literal_eval(x)) == 0 else x)
wa_heavy_metals = wa_results.loc[wa_results['heavy_metals'].isna() == False]

# Define WA heavy metals.
wa_limits = {
    'arsenic': 10.0,
    'cadmium': 4.1,
    'lead': 6.0,
    'mercury': 2.0
}
wa_metals = list(wa_limits.keys())

# Get WA heavy metals.
for metal in wa_metals:
    wa_heavy_metals[metal] = wa_heavy_metals['results'].apply(lambda x: get_result_value(x, metal))

# Plot distributions of WA heavy metals.
for metal in wa_metals:
    sns.histplot(wa_heavy_metals[metal], bins=25)
    plt.xlabel(f'{metal} (ppm)')
    plt.ylabel('Count')
    plt.title(f'Distribution of {metal} in WA Flower')
    plt.savefig(f'figures/{metal}_wa_distribution.png', bbox_inches='tight', dpi=300)
    plt.show()

# Plot distributions of WA and MD heavy metals.
for metal in wa_metals:
    
    # Filter out outliers and only select relevant data
    md_key = metal.title()
    md_filtered = md_results[(md_results[md_key] > 0) & (md_results[md_key] <= 1)]
    wa_filtered = wa_heavy_metals[(wa_heavy_metals[metal] > 0) & (wa_heavy_metals[metal] <= 1)]
    
    # Plot MD results
    sns.histplot(
        md_filtered[md_key],
        color='blue',
        bins=100,
        label='MD',
        alpha=0.5,
        stat='density',
    )
    
    # Plot WA results
    sns.histplot(
        wa_filtered[metal],
        color='red',
        bins=25,
        label='WA',
        alpha=0.5,
        stat='density',
    )

    # Add the Maryland limit.
    try:
        limit = md_limits[md_key]['Inhalation']
    except:
        limit = None
    if limit:
        plt.axvline(limit, color='r', linestyle='--', label='MD Limit (Inhalation)')

    # Label axes and title
    plt.xlabel(f'{md_key} (ppm)')
    plt.ylabel('Density (%)')
    plt.title(f'Distribution of {md_key} in MD and WA Flower when Detected')
    plt.xlim(0, 1)

    # Display legend
    plt.legend(loc='upper right')
    plt.savefig(f'figures/{metal}_wa_md_distribution.png', bbox_inches='tight', dpi=300)
    
    # Show the plot
    plt.show()
