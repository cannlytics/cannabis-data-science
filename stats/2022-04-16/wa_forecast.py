"""
WA Forecast
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 4/10/2022
Updated: 4/15/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script iterates on Washington State forecasts.

Data sources:

    - CCRS PRR All Data Up To 3-12-2022
    https://lcb.app.box.com/s/7pi3wqrmkuo3bh5186s5pqa6o5fv8gbs

Setup:

    1. pip install cannlytics

"""
from cannlytics.utils.utils import snake_case
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm

# Use the new CCRS interface.
from ccrs import CCRS


# Create a place for your data to live.
DATA_DIR = 'D:\\data\\washington'
FOLDER = 'CCRS PRR All Data Up To 3-12-2022'

#------------------------------------------------------------------------------
# Read the data.
#------------------------------------------------------------------------------

# Initialize a CCRS client.
ccrs = CCRS()

# Read lab results.
lab_results = ccrs.read_lab_results(DATA_DIR, FOLDER)

# Read licensee data.
licensees = ccrs.read_licensees(DATA_DIR, FOLDER)

# Read areas data.
dataset = 'Areas_0'
areas = pd.read_csv(
    f'{DATA_DIR}/{FOLDER}/{dataset}/{dataset}.csv',
    # usecols=model_names['lab_results']['fields'],
    # parse_dates=model_names['lab_results']['date_fields'],
    low_memory=False,
    nrows=1000,
)

# Read contacts data.
dataset = 'Contacts_0'
contacts = pd.read_csv(
    f'{DATA_DIR}/{FOLDER}/{dataset}/{dataset}.csv',
    # usecols=model_names['lab_results']['fields'],
    # parse_dates=model_names['lab_results']['date_fields'],
    low_memory=False,
    nrows=1000,
)

# Read integrator data.
dataset = 'Integrator_0'
integrators = pd.read_csv(
    f'{DATA_DIR}/{FOLDER}/{dataset}/{dataset}.csv',
    # usecols=model_names['lab_results']['fields'],
    # parse_dates=model_names['lab_results']['date_fields'],
    low_memory=False,
    nrows=1000,
)

# Read inventory data.
dataset = 'Inventory_0'
inventory = pd.read_csv(
    f'{DATA_DIR}/{FOLDER}/{dataset}/{dataset}.csv',
    # usecols=model_names['lab_results']['fields'],
    # parse_dates=model_names['lab_results']['date_fields'],
    low_memory=False,
    nrows=1000,
)


# Read inventory adjustment data.
dataset = 'InventoryAdjustment_0'
inventory_adjustments = pd.read_csv(
    f'{DATA_DIR}/{FOLDER}/{dataset}/{dataset}.csv',
    # usecols=model_names['lab_results']['fields'],
    # parse_dates=model_names['lab_results']['date_fields'],
    low_memory=False,
    nrows=1000,
)


# Read inventory plant transfer data.


# Read plant data.
dataset = 'Plant_0'
plants = pd.read_csv(
    f'{DATA_DIR}/{FOLDER}/{dataset}/{dataset}.csv',
    # usecols=model_names['lab_results']['fields'],
    # parse_dates=model_names['lab_results']['date_fields'],
    low_memory=False,
    nrows=1000,
)

# Read plant destruction data.
dataset = 'PlantDestructions_0'
plant_destructions = pd.read_csv(
    f'{DATA_DIR}/{FOLDER}/{dataset}/{dataset}.csv',
    # usecols=model_names['lab_results']['fields'],
    # parse_dates=model_names['lab_results']['date_fields'],
    low_memory=False,
    nrows=1000,
)


# Read product data.
dataset = 'Product_0'
products = pd.read_csv(
    f'{DATA_DIR}/{FOLDER}/{dataset}/{dataset}.csv',
    # usecols=model_names['lab_results']['fields'],
    # parse_dates=model_names['lab_results']['date_fields'],
    low_memory=False,
    nrows=1000,
)


# Read sale header data.
dataset = 'SaleHeader_0'
sale_headers = pd.read_csv(
    f'{DATA_DIR}/{FOLDER}/{dataset}/{dataset}.csv',
    # usecols=model_names['lab_results']['fields'],
    # parse_dates=model_names['lab_results']['date_fields'],
    low_memory=False,
    nrows=1000,
)

# Read sale detail data.
dataset = 'SalesDetail_0'
sale_details = pd.read_csv(
    f'{DATA_DIR}/{FOLDER}/{dataset}/{dataset}.csv',
    # usecols=model_names['lab_results']['fields'],
    # parse_dates=model_names['lab_results']['date_fields'],
    low_memory=False,
    nrows=1000,
)


# Read strain data.
dataset = 'Strains_0'
strains = pd.read_csv(
    f'{DATA_DIR}/{FOLDER}/{dataset}/{dataset}.csv',
    # usecols=model_names['lab_results']['fields'],
    # parse_dates=model_names['lab_results']['date_fields'],
    low_memory=False,
)


# Read transfer data.
dataset = 'Transfers_0'
transfers = pd.read_excel(
    f'{DATA_DIR}/{FOLDER}/{dataset}/{dataset}.xlsx',
    # usecols=model_names['lab_results']['fields'],
    # parse_dates=model_names['lab_results']['date_fields'],
    nrows=1000,
    skiprows=2,
)


def parse_transfer_items(row):
    """Parse transfer items from an observation of items shipped."""
    transfer_items = row.split('\n\n')
    items = transfer_items[-1].split('\n')
    items = [x.replace('\t', '/').split('/') for x in items]
    items = [[z.strip() for z in x] for x in items]
    item_data = pd.DataFrame(items)
    # FIXME: Assign columns
    # item_headers = transfer_items[0].split(' / ')
    # item_data.columns = [
    #     'category',
    #     'uid',
    #     'quantity',
    #     'medical',
    #     'uom',
    #     'inventory_name',
    #     'strain_name',
    #     'weight',
    #     'description',
    # ]
    return item_data

# Parse all inventory items.
columns = [
    'category',
    'uid',
    'quantity',
    'medical',
    'uom',
    'inventory_name',
    'strain_name',
    'weight',
    'description',
]
k = len(columns)
transfer_items = pd.DataFrame()
for index, transfer in transfers.iterrows():
    items = parse_transfer_items(transfer['Items Shipped'])
    items = items.iloc[: , :k]
    transfer_items = pd.concat([transfer_items, items], ignore_index=True)
transfer_items = transfer_items.loc[transfer_items[0] != 'Sub-category']
transfer_items.columns = columns


#------------------------------------------------------------------------------
# Future work: Augment the data with inventory data.
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# Clean the data.
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Explore the data.
#------------------------------------------------------------------------------

# Find the unique labs.
labs = list(lab_results['lab_licensee_id'].unique())

# Find the unique producers and processors.

# Find the samples that failed QC testing.
fail = lab_results.loc[lab_results['lab_test_status'] == 'Fail']

# TODO: Aggregate analyses by external ID.

# Count the number of samples by lab.
lab_tests = lab_results.groupby('lab_licensee_id')['external_identifier'].nunique()

# Look at the failure rate by lab.
group = ['lab_licensee_id', 'lab_test_status']
lab_fails = lab_results.groupby(group)['external_identifier'].nunique()

# # Estimate a logistic regression to see if any lab is more or less likely
# # to fail a sample.
# y = []
# x = []
# model = LogisticRegression(solver='liblinear', random_state=0)
# regression = model.fit(x, y)

# # Model evaluation.
# p_pred = model.predict_proba(x)
# y_pred = model.predict(x)
# score_ = model.score(x, y) # Model accuracy.
# conf_m = confusion_matrix(y, y_pred)
# report = classification_report(y, y_pred)

# model.score(x, y)

# # Plot the Confusion matrix.
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.imshow(conf_m)
# ax.grid(False)
# ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
# ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
# ax.set_ylim(1.5, -0.5)
# for i in range(2):
#     for j in range(2):
#         ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
# plt.show()


# Estimate a Logit with Statsmodels.
# x = np.arange(10).reshape(-1, 1)
# y = np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1])
# x = sm.add_constant(x)
# model = sm.Logit(y, x)
# result = model.fit(method='newton')
# (result.predict(x) >= 0.5).astype(int)
# cm = result.pred_table()


# TODO: Estimate a probit model to compare to the logit.
# Y = pd.get_dummies(subsample['strain'])['Indica']
# X = subsample[explanatory_variables]
# model = sm.Probit(Y, X).fit()
# print(model.summary())


#------------------------------------------------------------------------------
# Augment the data.
#------------------------------------------------------------------------------

# Get lab prices.

# Estimate laboratory revenue.

# Estimate laboratory market share.


#------------------------------------------------------------------------------
# Summarize the data.
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Analyze the data.
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Estimate ARIMAX for every variable.
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Save the date and forecasts.
#------------------------------------------------------------------------------


# TODO: Upload the data and make it available
# through the Cannlytics API.
