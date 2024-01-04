"""
Connecticut Market Analysis
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 12/17/2023
Updated: 12/28/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# External imports:
from cannlytics.data import create_hash
from dotenv import dotenv_values
from fredapi import Fred
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


# === Setup ===

plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
})

# Define colors.
COLORS = [
    "#228B22", "#8B4513", "#708090", "#808000", "#87CEEB", "#D2691E", 
    "#BDB76B", "#483D8B", "#DAA520", "#191970", "#FF4500", "#2E8B57", 
    "#4682B4", "#D2B48C", "#FF6347", "#40E0D0", "#EE82EE", "#F5DEB3", 
    "#FFD700", "#ADFF2F", "#F08080", "#20B2AA", "#778899", "#6A5ACD"
]


def anonymize(
        df: pd.DataFrame,
        columns = None,
        suffix: str = '_by',
        hash_function=create_hash,
    ) -> pd.DataFrame:
    """Anonymize a dataset by creating a hash for fields that end in "_by" or "_By."""
    if columns is None:
        suffix_pattern = f'.*{suffix}$'
        columns = df.filter(regex=suffix_pattern, axis=1, case=False).columns
    df.loc[:, columns] = df.loc[:, columns].astype(str).applymap(hash_function)
    return df


# === Connecticut Analysis ===

# Read Connecticut lab results.
datafile = './data/lab-results-ct-2023-12-28.xlsx'
ct_results = pd.read_excel(datafile)

# Add time variables.
ct_results['date'] = pd.to_datetime(ct_results['date_tested'], format='mixed', errors='coerce')
ct_results['month_year'] = ct_results['date'].dt.to_period('M')
ct_results['year'] = ct_results['date'].dt.to_period('Y')

# Tests per lab by month figure.
tests_per_lab_month = ct_results.groupby(['lab', 'month_year']).size().reset_index(name='total_tests')
pivot_data = tests_per_lab_month.pivot(index='month_year', columns='lab', values='total_tests')
pivot_data.fillna(0, inplace=True)
pivot_data = pivot_data.apply(pd.to_numeric, errors='coerce')
# TODO: Don't plot values of 0.
# TODO: Plot a red 'x' when AltaSci Labs exits.
fig, ax = plt.subplots(figsize=(14, 8))
for i, column in enumerate(pivot_data.columns):
    pivot_data[column][:-1].plot(linewidth=3.3, color=COLORS[i])
ax.set_title('Tests per Lab by Month in Connecticut', fontsize=32, weight='bold', pad=20)
ax.set_xlabel('', fontsize=28, labelpad=15)
ax.set_ylabel('Number of Tests', fontsize=28, labelpad=15)
ax.tick_params(axis='x', rotation=0)
ax.tick_params(axis='both', labelsize=24)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.legend(title='Lab', loc='best', fontsize=28, title_fontsize=32)
ax.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.7)
sns.despine(trim=True)
# Label news stories.
plt.vlines(
    pd.to_datetime('2021-12-06'),
    0,
    525,
    lw=2,
    linestyles='--',
    colors=['black'],
)
plt.text(
    pd.to_datetime('2021-12-06'),
    525,
    'News story about AltaSci Labs',
    verticalalignment='center',
    color='black'
)
plt.savefig('./figures/ct-tests-per-lab-per-month.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Market share by month figure.
market_share_by_month = ct_results.groupby(['month_year', 'lab']).size().unstack(fill_value=0)
market_share_percentage = market_share_by_month.divide(market_share_by_month.sum(axis=1), axis=0)
plt.figure(figsize=(14, 8))
market_share_percentage = market_share_percentage.iloc[:, ::-1]
market_share_percentage.plot(
    kind='area',
    stacked=True,
    ax=plt.gca(),
    alpha=0.8,
    color=[COLORS[1], COLORS[0]],
)
plt.hlines(
    y=0.5,
    xmin=market_share_percentage.index[0],
    xmax=market_share_percentage.index[-1],
    linestyles='dashed',
    linewidth=3.3,
    color='black',
    alpha=0.5,
)
plt.title('Market Share by Lab by Month in Connecticut', fontsize=20, weight='bold', pad=20)
plt.xlabel('', fontsize=16, labelpad=15)
plt.ylabel('Market Share (%)', fontsize=16, labelpad=15)
plt.yticks(fontsize=12)
plt.legend(title='Lab', loc='best', fontsize=12, title_fontsize=14)
plt.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.7)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig('./figures/ct-market-share-per-lab-per-month.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Product types by lab by month.
product_category_mapping = {
    'Cigarettes (pre-rolled cones)': 'Flower',
    'Raw material (flower)': 'Flower',
    'Usable Marijuana': 'Flower',
    'Liquid Marijuana RSO': 'Concentrate',
    'Marijuana Extract for Inhalation': 'Concentrate',
    'Vape Cartridge': 'Concentrate',
    'Baked goods': 'Other',
    'Capsule': 'Other',
    'Capsules or pills': 'Other',
    'Extracts, sprays, tinctures or oils': 'Concentrate',
    'Liquid Marijuana Infused Edible': 'Other',
    'Marijuana Infused Topicals': 'Other',
    'Non Smokable Infused Extract': 'Other',
    'Pill': 'Other',
    'Solid Marijuana Infused Edible': 'Other',
    'Tincture': 'Other',
    'Topical applications, oils or lotions': 'Other',
    'Transdermal Patch': 'Other',
    'Transdermal patches': 'Other'
}
ct_results['product_category'] = ct_results['product_type'].map(product_category_mapping)
labs = ['Northeast Laboratories', 'AltaSci Laboratories']
time_series_data = ct_results.groupby(['month_year', 'lab', 'product_category']).size().unstack(fill_value=0)
time_series_data = time_series_data.div(time_series_data.sum(axis=1), axis=0)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for i, lab in enumerate(labs):
    lab_data = time_series_data.xs(lab, level='lab')
    lab_data.plot(kind='area', stacked=True, ax=axes[i], alpha=0.8)
    axes[i].set_title(lab)
    axes[i].set_ylabel('Proportion')
    axes[i].set_xlabel('')
plt.tight_layout()
plt.savefig('./figures/ct-product-categories-time-series.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
