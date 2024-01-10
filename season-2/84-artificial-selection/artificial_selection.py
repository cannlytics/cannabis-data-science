"""
Artificial Selection
Cannabis Cultivation, Chemical Extraction, and Data Science
Copyright (c) 2022-2024 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
    Candace O'Sullivan-Sutherland <https://github.com/candy-o>
Created: 9/21/2022
Updated: 1/9/2024
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description: 

    Today we gather data on the emergence of cannabis varieties, or chemoforms,
    some of which are extracted into concentrated products.

Data Source:

    - Aggregated Cannabis Lab Test Results, Curated by Cannlytics
    URL: <https://cannlytics.page.link/tests>

"""
# External imports.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})

CA_FLOWER = [
    'Flower',
    'Flower, Inhalable',
    'Flower, Product Inhalable',
    'Flower, Medical Inhalable',
    'Indica, Flower, Inhalable',
    'Sativa, Flower, Inhalable',
]


#------------------------------------------------------------------------------
# Get the lab result data.
#------------------------------------------------------------------------------

# Download the data.
data_url = 'https://cannlytics.page.link/tests'

# Read California lab results (SC Labs).
ca = pd.read_excel(data_url, sheet_name='sc_labs_raw_data')

# Read Massachusetts lab results (MCR Labs).
ma = pd.read_excel(data_url, sheet_name='mcr_labs_raw_data')

# Read Michigan lab results (PSI Labs).
mi = pd.read_excel(data_url, sheet_name='psi_labs_raw_data')


#------------------------------------------------------------------------------
# Clean the lab result data!
#------------------------------------------------------------------------------

# pip install cannlytics
from cannlytics.data.coas import CoADoc

# Initialize a COA parser.
parser = CoADoc()

# Clean the lab result data.
ca_data = parser.standardize(ca)
ma_data = parser.standardize(ma)
mi_data = parser.standardize(mi)

# Get the lab result values for each subset.
data_dir = '../../.datasets/lab_results'

# Get MA lab result values.
# parser.save(ma, outfile=f'{data_dir}/ma.xlsx')
ma_values = pd.read_excel(f'{data_dir}/ma.xlsx', sheet_name='Values')
# ma_values = parser.standardize(ma, how='wide')

# FIXME: Get CA lab result values.
# parser.save(ca, outfile=f'{data_dir}/ca.xlsx')
# ca_values = pd.read_excel(f'{data_dir}/ca.xlsx', sheet_name='Values')

# FIXME: Get MI lab result values.
# parser.save(mi, outfile=f'{data_dir}/mi.xlsx')
# mi_values = pd.read_excel(f'{data_dir}/mi.xlsx', sheet_name='Values')


#------------------------------------------------------------------------------
# Look at the data!
#------------------------------------------------------------------------------

# Look at the distribution of THC in the various states.
ca_sample = ca_data.loc[ca_data['total_cannabinoids'] < 100]
ma_sample = ma_data.loc[ma_data['total_cannabinoids'] < 100]

# Plot the distribution of THC in the various states.
plt.figure(figsize=(15, 8))
ca_sample.total_cannabinoids.hist(bins=100, alpha=0.45, label='CA', color='red')
ma_sample.total_cannabinoids.hist(bins=100, alpha=0.45, label='MA', color='blue')
plt.title('Total Cannabinoids Observed in Cannabis Products in CA and MA in 2022')
plt.xlabel('Total Cannabinoids (%)')
plt.ylabel('Observations')
plt.legend()
plt.savefig('figures/total-cannabinoids-ca-ma-2022', dpi=300, bbox_inches='tight', transparent=False)
plt.show()


#------------------------------------------------------------------------------
# Look at the flower data.
#------------------------------------------------------------------------------

# Get the color palette.
colors = sns.color_palette()

# Look at the distribution in flower.
ca_flower = ca_sample.loc[ca_sample['product_type'].isin(CA_FLOWER)]
ma_flower = ma_sample.loc[ma_sample['product_type'].isin(['flower'])]

# Plot the distribution of THC in the various states.
ca_color = 'red'
ma_color = 'blue'
plt.figure(figsize=(15, 8))
ca_flower.total_cannabinoids.hist(bins=100, alpha=0.45, label='CA', color=ca_color)
ma_flower.total_cannabinoids.hist(bins=100, alpha=0.45, label='MA', color=ma_color)
plt.axvline(
    ca_flower.total_cannabinoids.mean(),
    ymin=0,
    ymax=1,
    linestyle='dotted',
    color=ca_color,
)
plt.text(
    ca_flower.total_cannabinoids.mean(),
    211,
    f"{round(ca_flower.total_cannabinoids.mean(), 1)}%",
    ha='center',
    color=ca_color,
    weight='bold',
)
plt.axvline(
    ma_flower.total_cannabinoids.mean(),
    ymin=0,
    ymax=0.95,
    linestyle='dotted',
    color=ma_color,
)
plt.text(
    ma_flower.total_cannabinoids.mean(),
    200,
    f"{round(ma_flower.total_cannabinoids.mean(), 1)}%",
    ha='center',
    color=ma_color,
    weight='bold'
)
difference = ca_flower.total_cannabinoids.mean() - ma_flower.total_cannabinoids.mean()
arrow_y = 0.5 * (plt.ylim()[1] + plt.ylim()[0])
midpoint_x = (ca_flower.total_cannabinoids.mean() + ma_flower.total_cannabinoids.mean()) / 2
plt.annotate(
    '',
    xy=(ca_flower.total_cannabinoids.mean(), arrow_y),
    xytext=(ma_flower.total_cannabinoids.mean(), arrow_y),
    arrowprops=dict(arrowstyle='<|-|>', lw=2, color='black')
)
plt.text(
    midpoint_x, 
    arrow_y + 5, 
    f"{round(abs(difference), 1)}%", 
    ha='center', 
    weight='bold',
)
plt.title('Total Cannabinoids Observed in Cannabis Flower in CA and MA in 2022\n',
          pad=20)
plt.xlabel('Total Cannabinoids (%)')
plt.ylabel('Observations')
plt.xlim(right=45)
plt.legend()
plt.savefig('figures/total-cannabinoids-flower-ca-ma-2022', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

print('CA average:', round(ca_flower.total_cannabinoids.mean(), 2))
print('MA average:', round(ma_flower.total_cannabinoids.mean(), 2))


#------------------------------------------------------------------------------
# Look at the analyte data.
#------------------------------------------------------------------------------

# Isolate a sample.
ma_values_sample = ma_values.loc[
    (ma_values['thc-a'] > 0.1) &
    (ma_values['thc-a'] < 100) &
    (ma_values['cbd-a'] > 0.1) &
    (ma_values['cbd-a'] < 100)
]

# Plot a histogram of the compounds.
ma_values_sample['cbda_to_thca_ratio'] = ma_values_sample['cbd-a'] / ma_values_sample['thc-a']
cmap = sns.color_palette("coolwarm_r", as_cmap=True)
plt.figure(figsize=(15, 8))
scatterplot = sns.scatterplot(
    data=ma_values_sample,
    x='thc-a',
    y='cbd-a',
    hue='cbda_to_thca_ratio',
    palette=cmap,
    s=400,
    alpha=0.7,
    legend=None,
)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.title('CBDA to THCA Observed in Cannabis Products in MA in 2022', pad=20)
plt.xlabel('THCA (%)')
plt.ylabel('CBDA (%)')
plt.grid(True)
plt.savefig('figures/cbda-to-thca-ma-2022', dpi=300, bbox_inches='tight', transparent=False)
plt.show()


#------------------------------------------------------------------------------
# Begin researching hypotheses.
#------------------------------------------------------------------------------

# Can we find the data / producer / state of the 1st occurrence of a
# particular strain? E.g. When did "Runtz" emerge?


# Can we find the date a particular strain spread to another state?



# Do chemical profiles vary by region?



# Does CA cannabis show more chemical variation than in MI or MA?



# Do new strains arise more often in CA than in MI or MA because a larger
# proportion of plants are grown from seed versus clone?


# Are concentrates more variable in CA than in MI or MA because of source
# flower?
