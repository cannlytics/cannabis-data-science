"""
Estimation with Bayesian Methods
Cannabis Data Science Meetup Group | Saturday Morning Statistics #12
Copyright (c) 2021 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 2/10/2022
Updated: 2/12/2022
License: MIT License <https://opensource.org/licenses/MIT>

Data sources:

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd?page=1
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd?page=2

Data Guide:

    - Washington State Leaf Data Systems Guide
    https://lcb.wa.gov/sites/default/files/publications/Marijuana/traceability/WALeafDataSystems_UserManual_v1.37.5_AddendumC_LicenseeUser.pdf

"""
# Standard imports.
from datetime import datetime
import gc
import json
import random
from typing import Any, Optional

# External imports.
import pandas as pd

# Internal imports.
from augmentation import augment_dataset
from utils import format_millions, get_number_of_lines


# Specify where the data lives.
DATA_DIR = 'D:\\leaf-data'
DATA_FILE = f'{DATA_DIR}/samples/random-sales-items-2022-02-11.csv'


#--------------------------------------------------------------------------
# Sample the data.
#--------------------------------------------------------------------------

# Optional: Determine the optimal sample size.
sale_items_datasets = [
    {'file_name': 'D:/leaf-data/SaleItems_0.csv', 'rows': 90_000_001},
    {'file_name': 'D:/leaf-data/SaleItems_1.csv', 'rows': 90_000_001},
    {'file_name': 'D:/leaf-data/SaleItems_2.csv', 'rows': 90_000_001},
    {'file_name': 'D:/leaf-data/SaleItems_3.csv', 'rows': 76_844_111},
]
total_sale_items = 0
for row in sale_items_datasets:
    total_sale_items += row['rows']

# Specify the sales items needed.
sales_items_fields = {
    'price_total': 'float',
    'sale_id': 'string',
    'mme_id': 'string',
    'inventory_id': 'string',
    'qty': 'float',
    'uom': 'string',
    'name': 'string',
}
sales_items_date_fields = [
    'created_at',
]

# Specify the time range to calculate statistics.
daily_total_sales = {}
time_range = pd.date_range(start='2021-02-01', end='2021-11-30')

# # Read a sample of each chunk of each sales items datafile.
# sample_size = 9_000
# percent = 0.0001
# random.seed(420)
# samples = []
# for dataset in sale_items_datasets:

#     # Read a random portion of each chunk in the dataset.
#     number_of_rows = dataset['rows']
#     file_name = dataset['file_name']
#     sample = pd.read_csv(
#         file_name,
#         sep='\t',
#         encoding='utf-16',
#         usecols=list(sales_items_fields.keys()) + sales_items_date_fields,
#         dtype=sales_items_fields,
#         parse_dates=sales_items_date_fields,
#         skiprows=lambda i: i > 0 and random.random() > percent
#     )
#     sample.rename(
#         columns={'name': 'product_name'},
#         inplace=True,
#     )
#     samples.append(sample)
#     print('Sampled:', len(sample))

#     # TODO: Augment with lab results, inventory data, etc.

#     # # Iterate over the time range.
#     # for date in time_range:

#     #     day = date.date()

#     #     # Get the day's sales.
#     #     day_sales = sale_items.loc[sale_items['created_at'] == date]

#     #     #  Add price_total to the daily total sales.
#     #     existing_stats = daily_total_sales.get(day, {'total': 0})
#     #     existing_sales = existing_stats['total']
#     #     total_sales = day_sales['price_total'].sum()
#     #     daily_total_sales[day] = {'total': existing_sales + total_sales}

#         # rows_read += chunk_size
#         # percent_read = round(rows_read / number_of_rows * 100)
#         # print('Augmented %s / %s (%i%%) observations from %s' %
#         #       (format_millions(rows_read), format_millions(number_of_rows),
#         #        percent_read, file_name))

# # Combine all samples.
# data = pd.concat(samples)

# # Save the random sample for future use.
# data.to_csv(DATA_FILE, index=False)


#--------------------------------------------------------------------------
# Augment the data with sales data.
#--------------------------------------------------------------------------

# # Read in the random sample.
# data = pd.read_csv(DATA_FILE)

# # Add sales type from sales data.
# chunk_size = 10_000_001
# sales_datasets = [
#     {'file_name': 'D:/leaf-data/Sales_0.csv', 'rows': 100_000_001},
#     {'file_name': 'D:/leaf-data/Sales_1.csv', 'rows': 100_000_001},
#     {'file_name': 'D:/leaf-data/Sales_2.csv', 'rows': 28_675_356},
# ]
# sales_fields = {
#     'global_id': 'string',
#     'type': 'string', # wholesale or retail_recreational
# }
# sales_columns = list(sales_fields.keys())

# for dataset in sales_datasets:

#     # Read in sales chunk by chunk and merge sales type with the data.
#     skip_rows = None
#     rows_read = 0
#     number_of_rows = dataset['rows']
#     file_name = dataset['file_name']
#     while rows_read < number_of_rows:

#         # Define the chunk size.
#         if rows_read > 0:
#             skip_rows = [i for i in range(1, rows_read)]

#         # Read and merge the chunk of sales.
#         samples = pd.read_csv(
#             file_name,
#             sep='\t',
#             encoding='utf-16',
#             usecols=sales_columns,
#             dtype=sales_fields,
#             nrows=chunk_size,
#             skiprows=skip_rows,
#         )
#         data = pd.merge(
#             left=data,
#             right=samples,
#             how='left',
#             left_on='sale_id',
#             right_on='global_id',
#         )
#         data.rename(columns={'type_x': 'type'}, inplace=True)
#         data.drop(['global_id', 'type_y'], axis=1, inplace=True, errors='ignore')

#         # Iterate.
#         rows_read += chunk_size
#         percent_read = round(rows_read / number_of_rows * 100)
#         print('Augmented %s / %s (%i%%) observations from %s' %
#               (format_millions(rows_read), format_millions(number_of_rows),
#                percent_read, file_name))

#         # Save the data.
#         data.to_csv(DATA_FILE, index=False)

# # Finish cleaning the sales data.
# data.rename(columns={
#     'name': 'product_name',
#     'type': 'sale_type',
# }, inplace=True,)

# # Save the data.
# data.to_csv(DATA_FILE, index=False)


#--------------------------------------------------------------------------
# Augment the data with inventory and inventory type data.
#--------------------------------------------------------------------------

# Read in the random sample.
data = pd.read_csv(DATA_FILE)

# Merge sale_items inventory_id to inventories inventory_id.
# FIXME:
# data = augment_dataset(
#     data,
#     data_file='D:/leaf-data/Inventories_0.csv',
#     fields={
#         'global_id': 'string',
#         'strain_id': 'string',
#         'inventory_type_id': 'string',
#         'lab_result_id': 'string',
#     },
#     merge_key='inventory_id',
#     chunk_size=13_000_000,
#     row_count=129_920_072,
# )
read_rows = 0
skiprows = None
columns = list(fields.keys())
if date_columns:
    columns += date_columns
if row_count is None:
    row_count = get_number_of_lines(data_file)
while read_rows < row_count:
    if read_rows:
        skiprows = [i for i in range(1, read_rows)]
    shard = pd.read_csv(
        data_file,
        sep=sep,
        encoding=encoding,
        usecols=columns,
        dtype=fields,
        skiprows=skiprows,
        nrows=chunk_size,
        parse_dates=date_columns,
    )
    match_columns = {}
    match_columns[match_key] = merge_key
    shard.rename(
        columns=match_columns,
        inplace=True,
    )
    data = data.merge(
        shard,
        on=merge_key,
        how='left'
    )
    column_names = list(data.columns)
    drop_columns = []
    rename_columns = {}
    for name in column_names:
        if name.endswith('_y'):
            drop_columns.append(name)
        if name.endswith('_x'):
            rename_columns[name] = name.replace('_x', '')
    try:
        data.drop(drop_columns, axis=1, inplace=True, errors='ignore')
    except TypeError:
        pass
    try:
        data.rename(columns=rename_columns, inplace=True)
    except TypeError:
        pass
    read_rows += chunk_size
    percent_read = round(read_rows / row_count * 100)
    print('Augmented %s / %s (%i%%) observations from %s' %
            (format_millions(read_rows), format_millions(row_count),
            percent_read, data_file))
del shard
gc.collect()

# Get inventory type (global_id) with inventory_type_id to get
# name and intermediate_type.
# data = augment_dataset(
#     data,
#     data_file='D:/leaf-data/InventoryTypes_0.csv',
#     fields={
#         'global_id': 'string',
#         'name': 'string',
#         'intermediate_type': 'string',
#     },
#     merge_key='inventory_type_id',
#     chunk_size=28_510_000,
#     row_count=57_016_229,
# )
# data.rename(
#     columns={'name': 'inventory_type_name'},
#     inplace=True,
# )

# Save the data.
data.to_csv(DATA_FILE, index=False)


#--------------------------------------------------------------------------
# Augment the data with lab results data.
#--------------------------------------------------------------------------

# Add lab result data.
# lab_result_fields = {
#     'global_id': 'string',
#     # 'global_for_inventory_id': 'string',
#     'cannabinoid_status': 'category',
#     'cannabinoid_cbc_percent': 'float',
#     'cannabinoid_cbc_mg_g': 'float',
#     'cannabinoid_cbd_percent': 'float',
#     'cannabinoid_cbd_mg_g': 'float',
#     'cannabinoid_cbda_percent': 'float',
#     'cannabinoid_cbda_mg_g': 'float',
#     'cannabinoid_cbdv_percent': 'float',
#     'cannabinoid_cbg_percent': 'float',
#     'cannabinoid_cbg_mg_g': 'float',
#     'cannabinoid_cbga_percent': 'float',
#     'cannabinoid_cbga_mg_g': 'float',
#     'cannabinoid_cbn_percent': 'float',
#     'cannabinoid_cbn_mg_g': 'float',
#     'cannabinoid_d8_thc_percent': 'float',
#     'cannabinoid_d8_thc_mg_g': 'float',
#     'cannabinoid_d9_thca_percent': 'float',
#     'cannabinoid_d9_thca_mg_g': 'float',
#     'cannabinoid_d9_thc_percent': 'float',
#     'cannabinoid_d9_thc_mg_g': 'float',
#     'cannabinoid_thcv_percent': 'float',
#     'cannabinoid_thcv_mg_g': 'float',
# }
# lab_results = pd.read_csv(
#     'D:/leaf-data/augmented/augmented-washington-state-lab-results.csv',
#     usecols=list(lab_result_fields.keys()),
#     dtype=lab_result_fields,
#     # nrows=1000,
# )
# data = pd.merge(
#     left=data,
#     right=lab_result_fields,
#     how='left',
#     left_on='lab_result_id',
#     right_on='global_id',
# )

# TODO: Calculate total cannabinoids.
# Use percent and mg_g if percent is missing.

# Save the data.
# data.to_csv(DATA_FILE, index=False)


#--------------------------------------------------------------------------
# Optional: Augment the data with retailer licensee data.
# Fields: latitude, longitude, name, postal_code
#--------------------------------------------------------------------------


#--------------------------------------------------------------------------
# Analyze the data.
#--------------------------------------------------------------------------

# Read in the random sample.
# data = pd.read_csv(DATA_FILE)

# TODO: Determine wholesale vs retail transactions.
# Drop observations with negative prices or prices > $1000?
# data = data.loc[data['sale_type'] != 'wholesale']
# data = data.loc[data.price_total > 0]

# Estimate the average_price (by sample type?)?


#--------------------------------------------------------------------------
# Visualize the data.
#--------------------------------------------------------------------------

# Optional: Chloropleth of average price by zip code.


# Look at the probability density functions (.pdfs) of the data.

#--------------------------------------------------------------------------
# Estimate statistics with Bayesian methods.
# P(Θ|X) = P(X|Θ)*P(Θ) / P(X)
# where:
# p(X|Θ) = p(X|μ) ~ N(μ, σ²)
#--------------------------------------------------------------------------

# TODO: Sample a bigger and bigger sample and see if estimates become more confident.


# Estimate prices.



#--------------------------------------------------------------------------
# Naive Bayes Algorithm.
#--------------------------------------------------------------------------

# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# # Training the Naive Bayes model on the Training set
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred = classifier.predict(X_test)

# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix, accuracy_score
# ac = accuracy_score(y_test,y_pred)
# cm = confusion_matrix(y_test, y_pred)
