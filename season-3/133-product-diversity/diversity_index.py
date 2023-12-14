"""
Diversity Index
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 10/31/2023
Updated: 10/31/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# External imports:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress


# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})

# === Get the data ===

# Read CT lab results.
ct_results = pd.read_excel('./data/ct-lab-results-2023-10-27-widened.xlsx')
ct_results['date'] = pd.to_datetime(ct_results['date_tested'])
ct_results['month_year'] = ct_results['date'].dt.to_period('M')


# === Chemical diversity ===

# Define compounds.
terpenes = ['a_farnesene', 'b_terpinene', 'a_bisabolol', 'b_myrcene', 'a_phellandrene', 'isoborneol', 'humulene_hum', 'trans_nerolidol', 'terpineol', 'b_caryophyllene', 'limonene', 'a_terpinene', 'isopulegol', 'borneol', 'ocimene', 'camphene', 'nerol', 'a_pinene', 'cedrol', 'eucalyptol', 'trans_b_farnesene', 'carene', 'b_farnesene', 'menthol', 'l_fenchone', 'geraniol', 'sabinene', 'b_pinene', 'geranyl_acetate', 'caryophyllene_oxide', 'valencene', 'pulegol', 'a_cedrene', 'cis_nerolidol', 'terpinolene', 'linalool_lin', 'guaiol', 'fenchol'] # 'camphor','b_eudesmol', 'fenchone',
cannabinoids = ['cbg', 'cannabidiol_acid_cbda', 'cbg_a', 'cannabavarin_cbdv', 'tetrahydrocannabivarin_thcv', 'cannbinol_cbn', 'cannabichromene_cbc', 'tetrahydrocannabinol_acid_thca', 'tetrahydrocannabinol_thc'] # 'cannabidiols_cbd', 

# Count detections for each compound.
ct_results[cannabinoids + terpenes] = ct_results[cannabinoids + terpenes].apply(pd.to_numeric, errors='coerce')
occurrences = (ct_results[cannabinoids + terpenes] > 0).sum()
sorted_occurrences = occurrences.sort_values(ascending=False)
print(sorted_occurrences)

# Draw the chemical abundance curve.
plt.figure(figsize=(12, 12))
plt.bar(sorted_occurrences.index, sorted_occurrences.values)
plt.yscale('log')
plt.title("Abundance of Chemicals in CT Cannabis Products")
plt.xlabel('Abundance Rank')
plt.ylabel('Abundance (log)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Analyze beta-pinene and d-limonene ratio over time.
ct_results['beta_pinene'] = ct_results['b_pinene'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
ct_results['d_limonene'] = ct_results['limonene'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
ct_results['pinene_to_limonene'] = ct_results['beta_pinene'] / ct_results['d_limonene']
ct_results['pinene_to_limonene'] = ct_results['pinene_to_limonene'].replace([np.inf, -np.inf], np.nan)
sample = ct_results.loc[ct_results['pinene_to_limonene'].isna() == False]

# Visualize beta-pinene to d-limonene ratio.
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=sample.loc[(sample['beta_pinene'] < 2) & (sample['d_limonene'] < 2)],
    # data=sample,
    x='d_limonene',
    y='beta_pinene',
    hue='pinene_to_limonene',
    hue_norm=(0, 2),
    palette="viridis",
    legend=False,
)
plt.title('Beta-Pinene to D-Limonene in CT Products')
plt.show()

# Regression on beta-pinene to d-limonene ratio over time.
monthly_data = ct_results.resample('M', on='date')['pinene_to_limonene'].mean()
x_values = np.arange(len(monthly_data))
x_values = np.arange(len(monthly_data))
slope, intercept, r_value, p_value, std_err = linregress(x_values, monthly_data.values)
plt.figure(figsize=(12, 8))
plt.plot(monthly_data.index, monthly_data, label='Average')
plt.plot(monthly_data.index, slope * x_values + intercept, label='Trend', linestyle='--')
plt.legend()
plt.ylim(0)
plt.title('Beta-Pinene to D-Limonene Ratio over Time in CT')
plt.show()

# Trend analysis.
if p_value < 0.05:
    significance = "Statistically significant"
else:
    significance = "Not statistically significant"
print(f"Trend: {round(slope * 100, 2)}% / mo.")
print(f"{significance} at the 0.05 level.")


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
compounds = cannabinoids + terpenes
ct_results['shannon_diversity'] = calculate_shannon_diversity(ct_results, compounds)

# Visualize chemical diversity over time.
diversities = ct_results.groupby('month_year')['shannon_diversity'].mean()
diversities.index = diversities.index.to_timestamp()
x_values = diversities.index
slope, intercept, r_value, p_value, std_err = linregress(np.arange(len(x_values)), diversities.values)
plt.figure(figsize=(12, 6))
plt.plot(x_values, diversities.values)
plt.plot(x_values, intercept + slope * np.arange(len(x_values)), color='red', linestyle="--", label="Trend")
plt.xlabel('')
plt.ylabel('Average Product Diversity')
plt.title('Diversity of Cannabinoids and Terpenes in CT Cannabis')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print(f"Slope: {slope}, p-value: {p_value}")

# Visualize chemical diversity by producer.
diversity_by_producer = ct_results.groupby(['month_year', 'producer'])['shannon_diversity'].mean().unstack()
diversity_by_producer.index = diversity_by_producer.index.to_timestamp()
plt.figure(figsize=(12, 8))
for producer in diversity_by_producer.columns:
    plt.plot(diversity_by_producer.index, diversity_by_producer[producer], label=producer)
plt.title('Cannabinoid and Terpene Diversity by CT Producer')
plt.xlabel('')
plt.ylabel('Average Product Diversity')
plt.legend(bbox_to_anchor=(0.75, -0.25))
plt.xticks(rotation=45)
plt.show()
