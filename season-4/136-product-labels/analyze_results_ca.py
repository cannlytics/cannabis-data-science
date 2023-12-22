"""
Analyze California Cannabis Lab Results
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 12/10/2023
Updated: 12/21/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# Standard imports:
import re
import matplotlib.pyplot as plt
import seaborn as sns

# External imports:
import pandas as pd
import statsmodels.api as sm


# === Setup ===

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})


# === Read the data ===

# Read the California lab results.
datafile ='./data/aggregated-ca-results-2023-12-21.xlsx'
results = pd.read_excel(datafile)


# === Pre-roll analysis ===

# Look at the pre-roll data.
product_categories = {
    'pre-roll': [
        'Pre-roll',
        'Plant (Preroll)',
        'Infused Pre-roll',
        'Plant (Enhanced/Infused Preroll)',
        'Infused Flower/Pre-Roll, Product Inhalable',
    ],
}
pre_roll_data = results.loc[results['product_type'].isin(product_categories['pre-roll'])]

# Look at types of pre-rolls.
plt.figure(figsize=(14, 8))
pre_roll_data['product_subtype'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=140,
    textprops={'fontsize': 18},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
)
plt.title('Types of pre-rolls observed in California in December 2023', fontsize=20, fontweight='bold', pad=20)
plt.ylabel('')
plt.tight_layout()
plt.savefig('./figures/pre-roll-subtypes-types.png', dpi=300, bbox_inches='tight')
plt.show()


# === Batch size analysis ===


def extract_pack_size(product_name):
    """Search for a pattern where a number is followed by 'pk' (e.g., '5pk')."""
    match = re.search(r'(\d+)pk', product_name)
    if match:
        return int(match.group(1))
    elif 'pk' in product_name:
        return 1
    else:
        return 1
    
def extract_weight(product_name):
    """Extract the weight of a product from the product name."""
    matches = re.findall(r'(\d+\.?\d*) *[gG]', product_name)
    if len(matches) == 0:
        return '?'
    return [float(match) for match in matches][0]


# Look batch sizes of pre-rolls.
pre_roll_data['batch_size'] = pd.to_numeric(pre_roll_data.batch_size.astype(str).replace('nan', '0').str.replace(' units', '', case=False))
prerolls = pre_roll_data.loc[pre_roll_data['batch_size'] > 0]
prerolls['batch_size'].hist(bins=25)
plt.title('Pre-roll batch sizes observed in California in December 2023', fontsize=32, fontweight='bold', pad=20)
plt.ylabel('Count')
plt.xlabel('Batch Size')
plt.tight_layout()
plt.savefig('./figures/pre-roll-batch-sizes.png', dpi=300, bbox_inches='tight')
plt.show()

# Identify prerolls sold in packs.
prerolls['pack'] = prerolls['product_name'].apply(extract_pack_size)
prerolls['pack'].value_counts().plot(kind='bar')
plt.xticks(rotation=0)
plt.title('Pre-roll pack sizes observed in California in December 2023', fontsize=32, fontweight='bold', pad=20)
plt.ylabel('Count')
plt.xlabel('Prerolls per Pack')
plt.tight_layout()
plt.savefig('./figures/pre-roll-pack-sizes.png', dpi=300, bbox_inches='tight')
plt.show()

# Identify prerolls with weights
prerolls['weights'] = prerolls['product_name'].apply(extract_weight)
plt.figure(figsize=(14, 8))
plt.title('Pre-roll weights observed in California in December 2023', fontsize=32, fontweight='bold', pad=20)
prerolls['weights'].value_counts().plot(kind='bar')
plt.ylabel('Count')
plt.xlabel('Pre-roll weight')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('./figures/pre-roll-weights.png', dpi=300, bbox_inches='tight')
plt.show()

# Look at the relationship between weight and d-limonene.
plt.scatter(
    prerolls['weights'].replace('?', 0),
    prerolls['d_limonene'],
    s=400,
)
plt.xlabel('Weight (g)')
plt.ylabel('d-Limonene Concentration (%)')
plt.show()


# === Cost of lab testing ===

# Convert prices to numeric.
prerolls['discount_price'] = pd.to_numeric(prerolls['discount_price'].str.replace('$', ''), errors='coerce')

# Get an estimate for lab prices.
# Data source: <https://caligreenlaboratory.com/wp-content/uploads/2022/04/CaliGreen_Price-Sheet_NonEditable_2022.pdf>
lab_test_price = 1_000
prerolls['lab_cost_per_unit'] = lab_test_price / prerolls['batch_size']
prerolls['lab_cost_percent'] = prerolls['lab_cost_per_unit'] / prerolls['discount_price'] * 100

# Calculate lab cost per preroll.
prerolls['lab_cost_per_preroll'] = lab_test_price / prerolls['batch_size'] / prerolls['pack']
plt.figure(figsize=(14, 8))
plt.title('Estimated Lab Testing Cost per Pre-roll in California in December 2023', fontsize=32, fontweight='bold', pad=20)
plt.scatter(
    prerolls['discount_price'],
    prerolls['lab_cost_per_preroll'],
    alpha=0.5,
    s=200,
)
plt.ylabel('Estimated Cost ($)')
plt.xlabel('Retail Price ($)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('./figures/pre-roll-weights.png', dpi=300, bbox_inches='tight')
plt.show()


# === Environment analysis ===

# Compare indoor vs. outdoor
indoor = results[results['product_subtype'] == 'Indoor']
outdoor = results[results['product_subtype'] == 'Full Sun']
indoor_outdoor_comparison = indoor.describe().join(outdoor.describe(), lsuffix='_indoor', rsuffix='_outdoor')

# Visualize indoor vs. outdoor cannabis.
plt.ylabel('Count')
plt.xlabel('Percent')
plt.title('Terpene Concentrations in Indoor vs. Full Sun Cannabis in CA')
indoor['total_terpenes'].hist(bins=40)
outdoor['total_terpenes'].hist(bins=40)
plt.legend(['Indoor', 'Outdoor'])
plt.xlim(0)
plt.tight_layout()
plt.savefig(f'./figures/indoor-outdoor-terpenes.png', bbox_inches='tight', dpi=300)
plt.show()


# === Chemical Analysis ===

# Look at total cannabinoids.
key = 'total_cannabinoids'
results[key] = pd.to_numeric(results[key], errors='ignore')
sample = results.dropna(subset=[key])
sample[key].hist(bins=1000)
plt.xlim(0, 100)
plt.show()

# Look at total terpenes.
key = 'total_terpenes'
results[key] = pd.to_numeric(results[key], errors='coerce')
sample = results.dropna(subset=[key])
sample[key].hist(bins=40)
plt.show()

# Look at moisture content.
key = 'moisture_content'
results[key] = pd.to_numeric(results[key], errors='coerce')
sample = results.dropna(subset=[key])
sample[key].hist(bins=40)
plt.show()

# Look at water activity.
key = 'water_activity'
results[key] = pd.to_numeric(results[key], errors='coerce')
sample = results.dropna(subset=[key])
sample[key].hist(bins=40)
plt.show()

# Look at moisture-adjusted total cannabinoids in flower.
types = ['Flower', 'Flower, Inhalable']
sample = results.loc[results['product_type'].isin(types)]
sample = sample.loc[~sample['total_cannabinoids'].isna()]
sample = sample.loc[~sample['moisture_content'].isna()]
sample['wet_total_cannabinoids'] = sample['total_cannabinoids'] / (1 + sample['moisture_content'] * 0.01)
sample['wet_total_cannabinoids'].hist(bins=1000)
plt.xlim(0, 100)
plt.show()

# Calculate the mean wet total cannabinoids in flower in CA.
valid = sample['wet_total_cannabinoids'].loc[sample['wet_total_cannabinoids'] < 100]
valid.mean()


# === Product subtype analysis ===

# Look at terpene concentrations in concentrate products:
concentrate_types = [
    'Badder',
    'Diamond',
    'Diamond Infused',
    'Crushed Diamond',
    'Liquid Diamonds',
    'Distillate',
    'Resin',
    'Live Resin',
    'Live Resin Infused',
    'Live Resin Sauce',
    'Sauce',
    'Live Rosin',
    'Unpressed Hash Green',
    # 'Fresh Press',
    # 'Hash Infused',
    # 'Rosin Infused',
]

# Creating a box plot of total terpenes in concentrates.
concentrate_data = results.loc[results['product_subtype'].isin(concentrate_types)]
filtered_data = concentrate_data.loc[~concentrate_data['total_terpenes'].isna()]
grouped_data = filtered_data.groupby('product_subtype')['total_terpenes'].apply(list)
mean_terpenes = {subtype: sum(values) / len(values) for subtype, values in grouped_data.items()}
sorted_subtypes = sorted(mean_terpenes, key=mean_terpenes.get)
data = [grouped_data[subtype] for subtype in sorted_subtypes]
labels = sorted_subtypes
plt.figure(figsize=(15, 11))
plt.boxplot(data, vert=False, labels=labels)
plt.xlabel('Total Terpenes (%)', labelpad=20)
plt.title('Terpene Concentrations in Concentrates in Observed in California in December 2023', pad=20)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('./figures/terpene-concentrations-ca-2023-12.png', dpi=300, bbox_inches='tight')
plt.show()


# === Lineage analysis ===

# Look at the most common parents.
lineage_data = results['lineage'].dropna()
unique_parents = pd.Series([parent for lineage in lineage_data for parent in lineage.split(' x ')]).value_counts()
filtered_strains = unique_parents[unique_parents >= 2]
plt.figure(figsize=(10, 8))
filtered_strains.sort_values()[-20:].plot(kind='barh')
plt.title('Lineages observed in California in December 2023', fontsize=32, fontweight='bold', pad=20)
plt.xlabel('Number of Descendants')
plt.ylabel('')
plt.title('Number of Descendants by Strain')
plt.tight_layout()
plt.savefig('./figures/lineages.png', dpi=300, bbox_inches='tight')
plt.show()


# === Timeseries analysis ===

# Format the date.
results['date'] = pd.to_datetime(results['date_tested'], format='mixed')

# TODO: Look at total THC levels over time.


# TODO: Look at total terpene levels over time.


# === Economic analysis ===


# TODO: Build a inverse-demand curve for milligrams of THC.



# === Price analysis ===

# Clean the price data.
results['discount_price'] = results['discount_price'].str.replace('$', '').astype(float)
price_data = results.loc[results['discount_price'] > 0]
price_data = price_data.loc[~price_data['amount'].isna()]
price_data['price_per_gram'] = price_data['discount_price'] / price_data['amount']

# See if THC, terpenes, etc. are correlated with price.
types = ['Flower', 'Flower, Inhalable']
# types = [
#     # 'Infused Flower/Pre-Roll, Product Inhalable',
#     'Pre-roll',
#     'Plant (Preroll)',
#     # 'Infused Pre-roll',
# ]
# types = [
#     'Concentrates & Extracts (Other)',
#     'Concentrates & Extracts (Distillate)',
#     'Extract',
#     'Concentrates & Extracts (Diamonds)',
#     'Concentrates & Extracts (Live Resin)',
#     'Concentrates & Extracts (Live Rosin)',
#     'Concentrates & Extracts (Vape)',
#     'Concentrate, Product Inhalable',
#     'Distillate',
# ]
type_price_data = price_data.loc[price_data['product_type'].isin(types)]
type_price_data = type_price_data.loc[type_price_data['total_cannabinoids'] < 100]

# Visualize the relationship between cannabinoids and price.
plt.figure(figsize=(10, 6))
sns.regplot(
    data=type_price_data,
    x='total_cannabinoids',
    y='price_per_gram',
)
plt.xlabel('Total Cannabinoids')
plt.ylabel('Price per gram ($)')
plt.title('Price per gram of flower to total cannabinoids in CA')
plt.grid(True)
plt.show()

# Visualize the relationship between terpenes and price.
plt.figure(figsize=(10, 6))
sns.regplot(
    data=type_price_data,
    x='total_terpenes',
    y='price_per_gram',
)
plt.xlabel('Total Terpenes')
plt.ylabel('Price per gram ($)')
plt.title('Price per gram of flower to total terpenes in CA')
plt.grid(True)
plt.show()

# Price vs. Chemical Properties Regression
X = type_price_data[['total_cannabinoids', 'total_terpenes']]
y = type_price_data['price_per_gram']
X_clean = X.dropna()
y_clean = y.reindex(X_clean.index)
X_clean = sm.add_constant(X_clean)
model = sm.OLS(y_clean, X_clean)
regression = model.fit()
print(regression.summary())

# Look at the average discount.
results['discount'].hist(bins=40)
plt.vlines(results['discount'].mean(), 0, 60, color='darkorange')
plt.show()

# TODO: Look at the average price per product type.


# === Visualize the data. ===

# # Look at sizes by product type.
# for product_type in results['product_type'].unique():
#     product_results = results.loc[results['product_type'] == product_type]
#     if len(product_results.amount.value_counts()) < 2:
#         continue
#     plt.figure(figsize=(8, 8))
#     product_results.amount.value_counts().plot(kind='pie')
#     plt.title(f'Amounts of {product_type} listed by The Flower Company in CA in 2023 Q4')
#     plt.gcf().set_facecolor('white')
#     plt.show()


# === Look at prices by product type and amount ===
    
# Look at prices by product type and amount.
# for product_type in results['product_type'].unique():
#     product_results = results.loc[results['product_type'] == product_type]
#     amounts_numeric = pd.to_numeric(product_results['amount'], errors='coerce')
#     prices_numeric = pd.to_numeric(product_results['price'], errors='coerce')
#     plt.figure(figsize=(10, 6))
#     plt.scatter(amounts_numeric, prices_numeric, alpha=0.5)
#     plt.title(f'Price vs Amount for {product_type}')
#     plt.xlabel('Amount')
#     plt.ylabel('Price')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.show()
