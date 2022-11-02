"""
Dosage Analysis
Copyright (c) 2022 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 10/10/2022
Updated: 10/12/2022
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# Standard imports.
from datetime import datetime
import os
import random

# External imports.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Internal imports.
from cannlytics.data.coas import CoADoc


#------------------------------------------------------------------------------
# Parse COA PDFs with CoADoc!
#------------------------------------------------------------------------------

# Specify where your Cannlytics repo lives.
COA_DIR = '../../assets/coas/ma'
DATA_DIR = '../../.datasets/coas'

# Initialize CoADoc.
parser = CoADoc()

# Read COA data from all specified directories.
coa_data = []
filenames = os.listdir(COA_DIR)
for filename in filenames:
    filename = filename.replace('.PDF', '.pdf')
    file_path = f'{DATA_DIR}/{filename}'
    data = parser.parse(file_path)
    coa_data.append(data[0])
    print('Parsed:', filename)

# Save the data.
timestamp = datetime.now().isoformat()[:19].replace(':', '-')
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
outfile = f'{DATA_DIR}/ma-coas-{timestamp}.xlsx'
parser.save(coa_data, outfile)
print('Successfully parsed %i COAs!' % len(coa_data))


#------------------------------------------------------------------------------
# Visualize the COA data.
#------------------------------------------------------------------------------

def scatter_text(ax, x, y, text_column, data, color='#004b09'):
    """Add text besides each point of a scatterplot.
       Based on this answer: https://stackoverflow.com/a/54789170/2641825"""
    for line in range(0, data.shape[0]):
         ax.text(
            data[x].iloc[line] + 0.01,
            data[y].iloc[line], 
            data[text_column].iloc[line],
            horizontalalignment='left', 
            color=color,
        )


# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 21,
})

# Read in the COA data.
datafile = f'{DATA_DIR}/ma-coas-2022-10-10T15-56-06.xlsx'
data = pd.read_excel(datafile)

# List the product types.
product_types = list(data.product_type.unique())
print('Product types:', product_types)

# Clean product names.
data['strain_name'] = data['product_name'].apply(
    lambda x: x.split(')')[-1].split(',,')[0].split('Bulk')[0]
)

# Calculate total THC and total CBD.
from cannlytics.utils.constants import DECARB
data['total_thc'] = data['delta_9_thc'] + data['thca'] * DECARB
data['total_cbd'] = data['cbda'] + data['cbda'] * DECARB
data['total_cbg'] = data['cbg'] + data['cbga'] * DECARB

# Calculate log THC to CBG ratio.
data['thc_to_cbg'] = data['total_thc'] / data['total_cbg']

# Calculate beta-pinene to d-limonene ratio.
data['beta_pinene_to_d_limonene'] = data['beta_pinene'] / data['d_limonene']

# Calculate
data['alpha_humulene_to_beta_caryophyllene'] = data['alpha_humulene'] / data['beta_caryophyllene']

# Separate product types.
joints = data.loc[data['product_type'] == 'Flower']

# Plot CBD to THC scatterplot with annotations.
ax = sns.scatterplot(
    data=joints,
    y='total_cbg',
    x='total_thc',
    hue='thc_to_cbg',
    palette='viridis_r',
    s=400,
)
scatter_text(
    ax,
    y='total_cbg',
    x='total_thc',
    text_column='strain_name',
    data = joints,
)
plt.legend(loc='lower right')
plt.title('CBG to THC ratio of a sample of MA joints')
plt.show()

# Restrict to those with terpenes.
terpene_joints = joints.loc[
    ~joints['beta_pinene_to_d_limonene'].isna()
]

# Plot beta-pinene to d-limonene scatterplot with annotations.
ax = sns.scatterplot(
    data=terpene_joints,
    x='d_limonene',
    y='beta_pinene',
    hue='beta_pinene_to_d_limonene',
    palette='viridis',
    s=400,
)
scatter_text(
    ax,
    x='d_limonene',
    y='beta_pinene',
    text_column='strain_name',
    data = terpene_joints,
)
plt.title('beta-Pinene to D-limonene ratio of a sample of MA flower')
plt.show()

# Plot alpha-humulene to beta-caryophyllene scatterplot with annotations.
ax = sns.scatterplot(
    data=terpene_joints,
    x='beta_caryophyllene',
    y='alpha_humulene',
    hue='alpha_humulene_to_beta_caryophyllene',
    palette='viridis_r',
    s=400,
)
scatter_text(
    ax,
    x='beta_caryophyllene',
    y='alpha_humulene',
    text_column='strain_name',
    data = terpene_joints,
)
plt.title('alpha-Humulene to beta-caryophyllene of a sample of MA flower')
plt.show()


#------------------------------------------------------------------------------
# Example analysis of the COA data.: Smoke 2 Joints
#------------------------------------------------------------------------------

# Read the data.
random.choice()

# Draw 6 joint, 365 times.
# ASSUMPTION: Each joint 1g.
consumption = []
days = range(0, 365)
for day in days:
    daily_consumption = terpene_joints.sample(
        1,
        replace=True,
    )
    consumption.append(daily_consumption)

# Plot daily, weekly, monthly consumption.
daily = pd.DataFrame()
analytes = ['total_thc', 'total_cbg', 'beta_pinene', 'd_limonene',
    'beta_caryophyllene', 'alpha_humulene']
colors = sns.color_palette('tab10', n_colors=len(analytes))
for analyte in analytes:
    daily[analyte] = pd.Series(
        [x[analyte].sum() * 10 for x in consumption],
        name=analyte,
    )

# Visualize daily consumption.
for i, analyte in enumerate(analytes):
    daily[analyte].plot(color=colors[i])
    name = analyte.replace('_', ' ').title()
    plt.title(f'Daily {name} consumption (mg)')
    plt.show()

# Visualize weekly consumption.
for i, analyte in enumerate(analytes):
    daily[analyte].rolling(7).sum().plot(color=colors[i])
    name = analyte.replace('_', ' ').title()
    plt.title(f'Weekly {name} consumption (mg)')
    plt.show()

# Visualize monthly consumption.
for i, analyte in enumerate(analytes):
    daily[analyte].rolling(30).sum().plot(color=colors[i])
    name = analyte.replace('_', ' ').title()
    plt.title(f'Monthly {name} consumption (mg)')
    plt.show()

# Calculate annual consumption.
for analyte in analytes:
    average_daily = daily[analyte].mean()
    average_weekly = daily[analyte].rolling(7).sum().mean()
    average_monthly = daily[analyte].rolling(30).sum().mean()
    annual = daily[analyte].sum()
    name = analyte.replace('_', ' ').title()
    print(f'\nConsumption of {name}')
    print('----------------------')
    print('Avg. Daily (mg):', round(average_daily, 2))
    print('Avg. Weekly (mg):', round(average_weekly, 2))
    print('Avg. Monthly (mg):', round(average_monthly, 0))
    print('Annual (g):', round(annual / 1000, 0))
