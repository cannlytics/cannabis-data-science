"""
Curate CCRS Strain Data
Copyright (c) 2023 Cannabis Data

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
    Candace O'Sullivan-Sutherland <https://github.com/candy-o>
Created: 1/25/2023
Updated: 1/25/2023
License: <https://github.com/cannabisdata/cannabisdata/blob/main/LICENSE>

Data Sources:

    - Washington State Liquor and Cannabis Board (WSLCB)
    URL: <https://lcb.box.com/s/xseghpsq2t4i1musxj6mgd7b8rhxe7bm>

    - Curated CCRS Strain Statistics
    URL: <https://cannlytics.page.link/ccrs-strain-stats-2022-12-07>

"""
# Standard imports:
import os

# External imports:
from cannlytics.utils import sorted_nicely
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

# Specify where your data lives.
base = 'D:\\data\\washington\\'
DATA_DIR = f'{base}\\CCRS PRR (12-7-22)\\CCRS PRR (12-7-22)\\'
STATS_DIR = f'{base}\\ccrs-stats\\'
FIGURES_DIR = './.figures'

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 21,
})


#------------------------------------------------------------------------------
# Data curation
#------------------------------------------------------------------------------

# Define the fields that will be used.
FIELDS = {
    'StrainType': 'string',
    'InventoryType': 'string',
    'UnitWeightGrams': 'string',
    'InitialQuantity': 'string',
    'QuantityOnHand': 'string',
    'strain_name': 'string',
}
numeric_columns = ['UnitWeightGrams', 'InitialQuantity', 'QuantityOnHand']

# Create stats directory if it doesn't already exist.
inventory_dir = os.path.join(STATS_DIR, 'inventory')
inventory_files = sorted_nicely(os.listdir(inventory_dir))

# Calculate strain statistics using curated inventory items.
strain_stats = pd.DataFrame({})
for i, datafile in enumerate(inventory_files):
    print('Augmenting:', datafile, i + 1, '/', len(inventory_files))

    # DEV:
    if i == 0:
        continue

    # Read the inventory items.
    data = pd.read_excel(
        os.path.join(inventory_dir, datafile),
        usecols=list(FIELDS.keys()),
        dtype=FIELDS,
    )

    # Get all inventory types of `InventoryType == 'Usable Marijuana'`
    flower = data.loc[data['InventoryType'] == 'Usable Marijuana']

    # Convert columns to numeric.
    for col in numeric_columns:
        flower[col] = flower[col].apply(lambda x: pd.to_numeric(x, errors='coerce'))

    # Sum `UnitWeightGrams` x `InitialQuantity` to get `total_weight`.
    total_weight = flower['UnitWeightGrams'].mul(flower['InitialQuantity'])

    # Sum `UnitWeightGrams` x (`InitialQuantity` - `QuantityOnHand`)
    # to get `total_sold`
    quantity_sold = flower['InitialQuantity'] - flower['QuantityOnHand']
    total_sold = flower['UnitWeightGrams'].mul(quantity_sold)

    # Convert weight to pounds.
    flower = flower.assign(
        total_weight = total_weight * 0.00220462,
        total_sold = total_sold * 0.00220462,
    )

    # Aggregate weights by `strain_name`.
    flower_stats = flower.groupby('strain_name').agg({
        'total_weight': 'sum',
        'total_sold': 'sum',
    })

    # Simply copy statistics on the first iteration.
    if i == 0:
        strain_stats = strain_stats.add(flower_stats, fill_value=0)

    # Otherwise, aggregate statistics.
    else:
        strain_weight = pd.concat(
            [strain_stats['total_weight'], flower_stats['total_weight']],
            axis=1
        ).sum(axis=1)
        strain_sold = pd.concat(
            [strain_stats['total_sold'], flower_stats['total_sold']],
            axis=1
        ).sum(axis=1)

        # Increment strain weight statistics.
        strain_stats = strain_stats.reindex(strain_weight.index)
        strain_stats.loc[strain_weight.index, 'total_weight'] = strain_weight

        # Increment strain sold statistics.
        strain_stats = strain_stats.reindex(strain_sold.index)
        strain_stats.loc[strain_sold.index, 'total_sold'] = strain_sold

    # Add strain type.
    flower_types = flower.groupby('strain_name')['StrainType'].first()
    strain_stats.loc[flower_types.index, 'strain_type'] = flower_types

# Save the strain statistics.
strains_dir = os.path.join(STATS_DIR, 'strains')
if not os.path.exists(strains_dir): os.makedirs(strains_dir)
outfile = os.path.join(strains_dir, 'strain-statistics.xlsx')
strain_stats.to_excel(outfile)


#------------------------------------------------------------------------------
# Data visualization
#------------------------------------------------------------------------------

# Create a `.figures` folder.
if not os.path.exists(FIGURES_DIR): os.makedirs(FIGURES_DIR)

# Visualize the total sold by `Hybrid`, `Indica`, and `Sativa`.
type_stats = strain_stats.groupby('strain_type').agg({
    'total_weight': 'sum',
    'total_sold': 'sum',
})
type_stats.div(1000).plot.bar(figsize=(8, 8))
plt.ylabel('Thousands of Pounds')
plt.title('Cannabis Pounds Sold in WA in 2022 by Strain Type')
plt.savefig(f'{FIGURES_DIR}/pounds-by-strain-type.png', bbox_inches='tight', dpi=300)
plt.show()

# Visualize yield by strain of the top 20 strains.
top_yielders = strain_stats['total_weight'].sort_values(ascending=False)[:20]
top_yielders.div(1000).sort_values(ascending=True)[:-2].plot.barh(figsize=(8, 12))
plt.ylabel('')
plt.xlabel('Thousands of Pounds')
plt.title('Cannabis Pounds Produced in WA in 2022 by Strain')
plt.savefig(f'{FIGURES_DIR}/pounds-produced-by-strain.png', bbox_inches='tight', dpi=300)
plt.show()

# Visualize the top selling 20 strains.
top_sellers = strain_stats['total_sold'].sort_values(ascending=False)[:20]
top_sellers.div(1000).sort_values(ascending=True).plot.barh(figsize=(8, 12))
plt.ylabel('')
plt.xlabel('Thousands of Pounds')
plt.title('Cannabis Pounds Sold in WA in 2022 by Strain')
plt.savefig(f'{FIGURES_DIR}/pounds-sold-by-strain.png', bbox_inches='tight', dpi=300)
plt.show()


#------------------------------------------------------------------------------
# Data analysis
#------------------------------------------------------------------------------

# Does `StrainType` correlate with higher or lower `total_weight`?
Y = strain_stats['total_sold']
X = pd.get_dummies(strain_stats['strain_type'])
del X['Hybrid']
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
print(model.summary())

# Calculate expected quantity sold by strain type.
const = model.params['const']
hybrid_expectations = const
indica_expectations = const + model.params['Indica']
sativa_expectations = const + model.params['Sativa']
print('Expected pounds sold for a Hybrid:', round(hybrid_expectations, 2), 'lbs.')
print('Expected pounds sold for a Indica:', round(indica_expectations, 2), 'lbs.')
print('Expected pounds sold for a Sativa:', round(sativa_expectations, 2), 'lbs.')
