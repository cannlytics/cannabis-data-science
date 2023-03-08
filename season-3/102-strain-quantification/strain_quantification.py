"""
Strain Quantification | What makes a cheese a cheese?
Copyright (c) 2023 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 3/1/2023
Updated: 3/1/2023
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

References:

    - Cannabis Tests
    URL: <https://huggingface.co/datasets/cannlytics/cannabis_tests>

    - MCR Labs Results 2023-02-06
    URL: <https://cannlytics.page.link/mcr-lab-results-2023-02-06>

    - Curated CCRS Inventory Lab Results
    URL: <https://cannlytics.page.link/ccrs-inventory-lab-results-2022-12-07>


Data Sources:

    - MCR Labs Test Results
    URL: <https://reports.mcrlabs.com>

    - Washington State Lab Test Results
    URL: <https://lcb.box.com/s/xseghpsq2t4i1musxj6mgd7b8rhxe7bm>

"""
# Internal imports:
import os

# External imports:
from cannlytics.data.coas import CoADoc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


#------------------------------------------------------------------------------
# Data: Quantify strains given MCR Labs test results.
#------------------------------------------------------------------------------

# Read in MCR Labs results.
data_dir = '.datasets/'
filename = 'mcr-lab-results-2023-02-06T14-35-24.xlsx'
ma_data = pd.read_excel(os.path.join(data_dir, filename))

# Standardize lab results.
parser = CoADoc()
standard_results = parser.standardize(ma_data, how='long')

# Find all unique analyses.
list(standard_results['analysis'].unique())

# Find all unique product types.
list(standard_results['product_type'].unique())

# Identify all flower samples of a particular variety, "cheese".
strain = standard_results.loc[
    (standard_results['product_type'] == 'flower') &
    (standard_results['product_name'].str.contains('cheese', case=False)) &
    (standard_results['analysis'] == 'terpenes')
]
len(strain)

# Find all "cheese" varieties.
list(strain['product_name'].unique())

# Find all terpenes.
terpenes = list(strain['key'].unique())
print(terpenes)


#------------------------------------------------------------------------------
# Visualization:
#------------------------------------------------------------------------------

# Plot the distribution of each terpene (separate and together on 1 plot).
terpene_data = standard_results.loc[
    (standard_results['product_type'] == 'flower') &
    (standard_results['analysis'] == 'terpenes')
]
terpene_data['value'] = pd.to_numeric(terpene_data['value'])
sns.displot(
    data=terpene_data,
    x='value',
    hue='key',
    kind='kde',
    fill=False,
    palette=sns.color_palette('bright', n_colors=len(terpene_data['key'].unique())),
    height=7,
    aspect=1.5,
)
plt.xlim(0, 0.075)


#------------------------------------------------------------------------------
# Analysis: What makes a cheese a cheese?
#------------------------------------------------------------------------------

# Disable warning.
pd.options.mode.chained_assignment = None  # default='warn'

# Run a regression with a dummy variable for "cheese" on each terpene.
averages = {}
for terpene in terpenes:

    # Isolate terpene results.
    # terpene = 'terpinolene'
    sample = standard_results.loc[
        (standard_results['product_type'] == 'flower') &
        (standard_results['analysis'] == 'terpenes') &
        (standard_results['key'] == terpene)
    ]

    # Isolate cheese results.
    criterion = sample['product_name'].str.contains('cheese', case=False)
    sample['cheese'] = 0 # FIXME: Raises pandas warning.
    sample.loc[sample.loc[criterion].index.values, 'cheese'] = 1

    # Restrict to numeric.
    sample = sample.loc[
        (sample['value'] > 0) &
        (sample['value'] < 100)
    ]

    # Run a regression of value on "cheese" dummy.
    Y = pd.to_numeric(sample['value'])
    X = pd.get_dummies(sample['cheese'])
    del X[0]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    # print(model.summary())

    # Predict the difference of terpene concentrations.
    const = model.params['const']
    y_0_hat = const
    y_1_hat = const + model.params[1]
    averages[terpene] = {
        'average': round(y_0_hat, 2),
        'variety_average': round(y_1_hat, 2),
        'percent_difference': round((y_1_hat - y_0_hat) / y_0_hat * 100, 2)
    }

# Print out averages.
print('Terpene | MA Average | MA Cheese Average | Percent Difference')
print('--------|---------|----------------|------------------|')
for terpene, values in averages.items():
    print(
        terpene, 
        f'  |    {values["average"]}%  |  ',
        f'{values["variety_average"]}%  |',
        f'{values["percent_difference"]}%  |'
    )


# TODO: Run a regression on ratios:
# - beta-Pinene to D-limonene
# - beta-Caryophyllene to alpha-Humulene
# - THC to CBD


#------------------------------------------------------------------------------
# Data: Compare strains observed in Washington to those in Massachusetts.
#------------------------------------------------------------------------------

# Read in WA lab results.
data_dir = '.datasets/'
filename = 'ccrs-inventory-lab-results-2022-12-07.xlsx'
wa_data = pd.read_excel(os.path.join(data_dir, filename))

# Find WA product types.
list(wa_data.inventory_type.unique())

# Identify all "cheese" flower tests.
wa_cheese = wa_data.loc[
    (wa_data['inventory_type'] == 'Flower Lot') &
    (wa_data['strain_name'].str.contains('cheese', case=False))
]
print(len(wa_cheese))

# Find all "cheese" varieties in WA.
list(wa_cheese['strain_name'].unique())


#------------------------------------------------------------------------------
# Analysis: Compare the distribution of THC and CBD in WA to MA.
#------------------------------------------------------------------------------

# Get WA cheese results.
cannabinoid = 'thca'
wa_cheese = wa_cheese.loc[
    (wa_cheese[cannabinoid] > 0) &
    (wa_cheese[cannabinoid] < 100)
]
wa_cheese[cannabinoid].hist(bins=50)
wa_average = round(wa_cheese[cannabinoid].mean(), 2)
print('Average THCA in WA cheese:', f'{wa_average}%')

# Get MA cheese results.
ma_cheese = standard_results.loc[
    (standard_results['product_type'] == 'flower') &
    (standard_results['product_name'].str.contains('cheese', case=False)) &
    (standard_results['key'] == cannabinoid)
]
ma_cheese['value'].hist(bins=50)
ma_average = round(ma_cheese['value'].mean(), 2)
print('Average THCA in MA cheese:', f'{ma_average}%')


#------------------------------------------------------------------------------
# Insight: Who has the cheesier cheese?
#------------------------------------------------------------------------------

# Compile all MA and WA data into a panel dataset.
ma_sample = pd.DataFrame({
    'product_name': ma_cheese['product_name'],
    'value': ma_cheese['value'],
    'state': 'MA',
})
wa_sample = pd.DataFrame({
    'product_name': wa_cheese['strain_name'],
    'value': wa_cheese['thca'],
    'state': 'WA',
})
sample = pd.concat([ma_sample, wa_sample])

# Run a regression of THCA conditional on state.
Y = pd.to_numeric(sample['value'])
X = pd.get_dummies(sample['state'])
del X['MA']
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
print(model.summary())

# TODO: Compare THC to CBD ratio.

