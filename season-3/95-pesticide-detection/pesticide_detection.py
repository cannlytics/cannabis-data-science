"""
Pesticide Detection
Copyright (c) 2023 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 1/18/2023
Updated: 1/18/2023
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Data Sources:

    - Washington State Liquor and Cannabis Board (WSLCB)
    URL: <https://lcb.box.com/s/xseghpsq2t4i1musxj6mgd7b8rhxe7bm>

    - Curated CCRS Inventory Lab Results
    URL: <https://cannlytics.page.link/ccrs-inventory-lab-results-2022-12-07>

"""
# Standard imports:
import ast
import gc
import os

# External imports:
from cannlytics.utils import camel_to_snake
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

# Specify where your data lives.
DATA_DIR = '../../.datasets/washington'

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 21,
})


#------------------------------------------------------------------------------
# Get the data.
#------------------------------------------------------------------------------

# Read in the data.
lab_results = pd.read_excel(os.path.join(DATA_DIR, 'ccrs-lab-results-2022.xlsx'))
inventory = pd.read_excel(os.path.join(DATA_DIR, 'ccrs-inventory-lab-results-2022.xlsx'))
inventory.rename(columns=lambda x: camel_to_snake(x), inplace=True)

# Merge lab results and inventory.
data = pd.merge(
    inventory,
    lab_results,
    on='inventory_id',
    how='left',
    validate='m:1',
)

# Clean columns.
drop = []
for column in list(data.columns):
    if column.endswith('_y'):
        key = column.replace('_y', '')
        old_key = column.replace('_y', '_x')
        try:
            data[key] = data[column].fillna(data[old_key])
        except:
            pass
        drop.extend([column, old_key])
data.drop(drop, axis=1, inplace=True)

# Perform garbage cleaning.
del inventory
del lab_results
gc.collect()


#------------------------------------------------------------------------------
# Look at the data.
#------------------------------------------------------------------------------

# Find out all the unique product types.
product_types = list(data['inventory_type'].unique())
counts = data.groupby('inventory_type')['inventory_type'].count()
counts.sort_values(inplace=True, ascending=True)
counts.loc[counts > 1000].plot.barh(
    figsize=(8, 8),
)
plt.ylabel('Count')
plt.title('Cannabis product types tested in WA in 2022')
plt.show()


#------------------------------------------------------------------------------
# Analyze the data.
#------------------------------------------------------------------------------

# Count the pesticides detected.
pesticides = {}
for _, values in data.iterrows():
    detected = ast.literal_eval(values['pesticides'])
    if detected:
        for analyte in detected:
            count = pesticides.get(analyte, 0) + 1
            pesticides[analyte] = count

# Visualize the top pesticides detected.
pesticides_detected = pd.Series(pesticides)
pesticides_detected.sort_values(inplace=True, ascending=False)
pesticides_detected[:20].plot.barh(
    figsize=(8, 8),
)
plt.ylabel('Count')
plt.title('Pesticides detected in cannabis in WA in 2022')
plt.show()


#------------------------------------------------------------------------------
# Model the data.
#------------------------------------------------------------------------------

def identify_detection(value, analyte):
    """Determine if an analyte is detected."""
    detected = ast.literal_eval(value)
    if analyte in detected:
        return 1
    else:
        return 0


# Predict the probability of detecting piperonyl_butoxide.
data['piperonyl_butoxide'] = data['pesticides'].apply(
    lambda x: identify_detection(x, 'piperonyl_butoxide')
)
print(data['piperonyl_butoxide'].value_counts())

# Isolate a sub-sample.
sample = data.loc[
    (data['inventory_type'] == 'Flower Lot') |
    (data['inventory_type'] == 'Hydrocarbon Concentrate')
]

# Estimate the model.
Y = sample['piperonyl_butoxide']
X = pd.get_dummies(sample['inventory_type'])['Hydrocarbon Concentrate']
X = sm.add_constant(X)
model = sm.Probit(Y, X).fit(disp=0)
print(model.summary())

# Calculate marginal effect of detection in hydrocarbon concentrates.
marginal_prob_flower = model.predict(pd.Series(
    {'const': 1, 'Hydrocarbon Concentrate': 0},
))
marginal_prob_concentrate = model.predict(pd.Series(
    {'const': 1, 'Hydrocarbon Concentrate': 1},
))
plot_data = pd.DataFrame({
    'Hydrocarbon Concentrates': marginal_prob_concentrate  * 100,
    'Flower Lots': marginal_prob_flower * 100,
})
plt.figure(figsize=(8, 6))
sns.barplot(
    data=plot_data,
    palette=sns.color_palette('husl', n_colors=2),
)
plt.ylabel('Percent (%)')
plt.title('Probability of piperonyl butoxide detection\nin cannabis in WA in 2022')
plt.show()

# Future work: Predict the number of pesticides to be detected.
