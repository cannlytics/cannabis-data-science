"""
Sample Statistics
Cannabis Data Science Meetup Group
Copyright (c) 2021 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 2/5/2021
Updated: 2/5/2022
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
import random

# External imports.
import pandas as pd

# Internal imports.
from utils import format_millions, get_number_of_lines


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
    # 'mme_id': 'string',
    # 'inventory_id': 'string',
    # 'qty': 'float',
    # 'uom': 'string',
    # 'name': 'string',
}
sales_items_date_fields = [
    'created_at',
]

# Specify the time range to calculate statistics.
daily_total_sales = {}
time_range = pd.date_range(start='2021-02-01', end='2021-11-30')

# Read a sample of each chunk of each sales items datafile.
chunk_size = 10_000_000
percent = 0.001
for dataset in sale_items_datasets:

    skip_rows = None
    rows_read = 0
    number_of_rows = dataset['rows']
    file_name = dataset['file_name']
    while rows_read < number_of_rows:
        skip_rows = [i for i in range(1, rows_read + chunk_size) if random.random() > percent]

    # FIXME: Read a random portion of each chunk in the dataset.

        # Read in the chunk of sales.
        sale_items = pd.read_csv(
            file_name,
            sep='\t',
            encoding='utf-16',
            usecols=list(sales_items_fields.keys()) + sales_items_date_fields,
            dtype=sales_items_fields,
            parse_dates=sales_items_date_fields,
            nrows=chunk_size,
            skiprows=skip_rows,
        )
        sale_items.rename(
            columns={'name': 'product_name'},
            inplace=True,
        )

        # TODO: Augment with lab results, inventory data, etc.

        # Iterate over the time range.
        for date in time_range:

            day = date.date()

            # Get the day's sales.
            day_sales = sale_items.loc[
                sale_items['created_at'] == date
            ]

            #  Add price_total to the daily total sales.
            existing_stats = daily_total_sales.get(day, {'total': 0})
            existing_sales = existing_stats['total']
            total_sales = day_sales['price_total'].sum()
            daily_total_sales[day] = {'total': existing_sales + total_sales}

        rows_read += chunk_size
        percent_read = round(rows_read / number_of_rows * 100)
        print('Augmented %s / %s (%i%%) observations from %s' %
              (format_millions(rows_read), format_millions(number_of_rows),
               percent_read, file_name))


#--------------------------------------------------------------------------
# Analyze the data.
#--------------------------------------------------------------------------

# Estimate various statistics.
daily_sales_data = pd.DataFrame.from_dict(
    daily_total_sales,
    orient='index'
)

# Estimate total revenue by multiplying by 1000?


# Estimate the average_price (by sample type?)?

