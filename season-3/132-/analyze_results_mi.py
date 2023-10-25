"""
Analyze Results from MI PRR
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 10/23/2023
Updated: 10/23/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

References:


"""
# External imports:
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib import cm
import numpy as np
import pandas as pd
import re
import seaborn as sns
from scipy import stats


# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})


def save_figure(filename, dpi=300, bbox_inches='tight'):
    """Save a figure to the figures directory."""
    plt.savefig(f'figures/{filename}', bbox_inches=bbox_inches, dpi=dpi)


# === Get the data ===

# Read MA lab results.
mi_results = pd.read_excel('./data/Metrc_Flower_Potency_Final_2.17.23.xlsx')


# === Clean the data ===

# Rename certain columns.
mi_results = mi_results.rename(columns={
    'ProductName': 'product_name',
    'ProductCategory': 'product_type',
    'TestType': 'test_type',
    'Quantity': 'total_thc',
    'Licensee': 'lab',
    'TestPerformedDate': 'date_tested',
    'Comment': 'notes',
    'Med AU': 'medical',
})

# Add a date column.
mi_results['date'] = pd.to_datetime(mi_results['date_tested'])
mi_results['month_year'] = mi_results['date'].dt.to_period('M')

# Exclude outliers.
sample = mi_results.loc[
    (mi_results['total_thc'] > 0) &
    (mi_results['total_thc'] < 100) &
    (mi_results['product_type'] == 'Flower')
]
print('Number of samples:', len(sample))


# === Analyze market share. ===

# Count the number of labs.
labs = sample['lab'].unique()
print('Number of labs:', len(labs))

# Visualize the number of tests by lab.
lab_results = sample.groupby('lab')
tests_by_lab = lab_results['total_thc'].count().sort_values(ascending=False)
tests_by_lab.plot(kind='bar')
save_figure('mi-tests-by-lab.png')
plt.show()

# Visualize market share by lab.
market_share = tests_by_lab.div(tests_by_lab.sum()).mul(100).round(2)
market_share.plot(kind='bar')
save_figure('mi-market-share-by-lab.png')
plt.show()


# === Analyze tests by month. ===

# Visualize the frequency of tests by month/year.
test_frequency = sample['month_year'].value_counts().sort_index()
subsample = test_frequency[2:-1]
subsample.index = subsample.index.to_timestamp()
plt.figure(figsize=(10, 6))
sns.lineplot(
    x=subsample.index,
    y=subsample.values,
    marker="o",
    color="mediumblue"
)
plt.title('Frequency of Tests by Month/Year', fontweight='bold')
plt.ylabel('Number of Tests')
plt.xlabel('Month/Year')
plt.xticks(rotation=45, ha='right')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
save_figure('mi-tests-by-month.png')
plt.show()


# === Analyze medical vs. adult-use testing. ===

# Visualize the frequency distribution for medical.
med_au_distribution = sample['medical'].value_counts()
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x=med_au_distribution.index, y=med_au_distribution.values, palette="muted")
plt.title('Frequency Distribution of Medical to Adult-Use', fontweight='bold')
plt.ylabel('Frequency')
plt.xlabel('Medical to Adult-Use')
for index, value in enumerate(med_au_distribution.values):
    bar_plot.text(index, value + 0.1, str(value), color='black', ha="center")
plt.tight_layout()
save_figure('mi-med-au-frequency.png')
plt.show()

# Visualize adult-use vs. medical tests over time.
grouped = sample.groupby(['month_year', 'medical']).size().reset_index(name='counts')
pivot_grouped = grouped.pivot(index='month_year', columns='medical', values='counts').fillna(0)
pivot_grouped = pivot_grouped.apply(pd.to_numeric, errors='coerce')
pivot_grouped.index = pivot_grouped.index.to_timestamp()
pivot_grouped = pivot_grouped[:-1]
plt.figure(figsize=(10, 6))
for column in pivot_grouped.columns:
    sns.lineplot(data=pivot_grouped, x=pivot_grouped.index, y=column, marker="o", label=column)
plt.title('Number of Adult Use vs Medical Tests by Month', fontweight='bold')
plt.ylabel('Number of Tests')
plt.xlabel('Month/Year')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Med/AU')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
save_figure('mi-med-au-tests-by-month.png')
plt.show()


# === Analyze total THC. ===

# Visualize the distribution of THC.
mode_value = sample['total_thc'].mode().iloc[0]
median_value = sample['total_thc'].median()
quantile_99 = sample['total_thc'].quantile(0.99)
plt.figure(figsize=(12, 7))
sns.histplot(sample['total_thc'], bins=100, color="lightblue", kde=True)
plt.axvline(mode_value, color='red', linestyle='dashed', linewidth=2, label=f"Mode: {mode_value:.2f}%")
plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f"Median: {median_value:.2f}%")
plt.axvline(quantile_99, color='blue', linestyle='dashed', linewidth=2, label=f"99th Quantile: {quantile_99:.2f}%")
plt.title('Distribution of Total THC in MI Cannabis Flower', fontweight='bold')
plt.xlabel('Total THC (%)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
save_figure('mi-total-thc-distribution.png')
plt.show()

# Visualize the difference between medical and adult-use THC.
plt.figure(figsize=(12, 7))
sns.histplot(
    data=sample,
    x='total_thc',
    hue='medical',
    bins=100,
    kde=True,
    palette={"Med": "blue", "AU": "green"},
    stat='density',
)
median_med = sample[sample['medical'] == 'Med']['total_thc'].median()
median_au = sample[sample['medical'] == 'AU']['total_thc'].median()
plt.axvline(median_med, color='blue', linestyle='--', linewidth=1.5, label=f"Medical Median: {median_med:.2f}%")
plt.axvline(median_au, color='green', linestyle='--', linewidth=1.5, label=f"Adult-Use Median: {median_au:.2f}%")
plt.title('Distribution of Total THC for Medical vs. Adult-Use', fontweight='bold')
plt.xlabel('Total THC (%)')
plt.ylabel('Frequency')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.tight_layout()
save_figure('mi-med-au-total-thc-distribution.png')
plt.show()

# Perform a t-test to determine if the difference between medical and adult-use THC is significant.
med_thc = sample[sample['medical'] == 'Med']['total_thc']
au_thc = sample[sample['medical'] == 'AU']['total_thc']
t_stat, p_val = stats.ttest_ind(med_thc, au_thc, equal_var=True) # You can set equal_var to False for Welch's t-test
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_val}")
alpha = 0.05
if p_val < alpha:
    print("The difference between medical and adult-use THC is statistically significant.")
else:
    print("The difference between medical and adult-use THC is not statistically significant.")


# === Analyze THC by lab. ===

# Visualize average THC percentage for each licensee.
average_thc_by_licensee = sample.groupby('lab')['total_thc'].mean()
plt.figure(figsize=(25, 6))
bar_plot = sns.barplot(x=average_thc_by_licensee.index, y=average_thc_by_licensee.values, palette="pastel")
plt.title('Average Total THC by Lab in MI', fontweight='bold')
plt.ylabel('Average Total THC (%)')
plt.xlabel('Lab')
plt.xticks(rotation=45, ha='right')
for index, value in enumerate(average_thc_by_licensee.values):
    bar_plot.text(index, value + 0.2, f"{value:.0f}%", color='black', ha="center")
plt.tight_layout()
save_figure('mi-total-thc-by-lab.png')
plt.show()


# === Augment strain data. ===

def extract_strain_name(product_name):
    """Extract the strain name from the product name."""
    name = str(product_name)
    strain_name = re.split(r' - | \| | _ | x | – | — |:|\(|\)|/', name)[0]
    strain_name = strain_name.split('Buds')[0].strip()
    strain_name = strain_name.split('Bulk')[0].strip()
    strain_name = strain_name.split('Flower')[0].strip()
    strain_name = strain_name.split('Pre-Roll')[0].strip()
    return strain_name


# Augment strain names.
sample['strain_name'] = sample['product_name'].apply(extract_strain_name)
print(sample.sample(10)['strain_name'])


# === Analyze strains. ===

# Exclude samples with strain_name set to None, '' or 'Unprocessed'
sample = sample[sample['strain_name'].notna()]
sample = sample[~sample['strain_name'].isin(['', 'Unprocessed'])]

# Visualize the frequency of each strain
strain_counts = sample['strain_name'].value_counts()
subsample = strain_counts.head(20)
plt.figure(figsize=(12, 10))
bar_plot = sns.barplot(
    y=subsample.index,
    x=subsample.values,
    palette="pastel",
)
plt.title('Frequency Distribution of Strains', fontweight='bold')
plt.xlabel('Number of Tests')
plt.ylabel('Strain Name')
for index, value in enumerate(subsample.values):
    bar_plot.text(value, index, str(value), color='black', ha="left", va="center")
plt.tight_layout()
save_figure('mi-top-strains.png')
plt.show()

# Visualize the average THC for the top strains.
avg_thc_per_strain = sample.groupby('strain_name')['total_thc'].mean().sort_values(ascending=False)
overall_avg_thc = sample['total_thc'].mean()
print('Overall average THC:', round(overall_avg_thc, 2))
print('99th percentile THC:', round(sample['total_thc'].quantile(0.99), 2))
top_20_strains = strain_counts.head(20).index
avg_thc_top_20_strains = avg_thc_per_strain[avg_thc_per_strain.index.isin(top_20_strains)]
avg_thc_top_20_strains = avg_thc_top_20_strains.loc[top_20_strains]
print('Average THC for top 20 strains:', round(avg_thc_top_20_strains.mean(), 2))
plt.figure(figsize=(15, 12))
bar_plot = sns.barplot(
    x=avg_thc_top_20_strains.index,
    y=avg_thc_top_20_strains.values,
    palette="pastel"
)
plt.axhline(y=overall_avg_thc, color='red', linestyle='--', label=f'Overall Avg: {overall_avg_thc:.2f}%')
plt.title('Average Total THC by Top 20 Strains', fontweight='bold')  # Modified the title
plt.ylabel('THC Percentage (%)')
plt.xlabel('Strain')
plt.xticks(rotation=45, ha='right')
plt.legend()
for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_height(), '.2f') + '%', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='center', 
                      xytext=(0, 9), 
                      textcoords='offset points')
plt.tight_layout()
save_figure('mi-avg-thc-by-top-20-strains.png')
plt.show()

# TODO: Look at top adult-use strains vs. top medical strains.
