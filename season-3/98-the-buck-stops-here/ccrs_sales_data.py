"""
Curate CCRS Sales
Copyright (c) 2022-2023 Cannabis Data

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 1/1/2023
Updated: 2/1/2023
License: <https://github.com/cannabisdata/cannabisdata/blob/main/LICENSE>

Data Source:

    - Washington State Liquor and Cannabis Board (WSLCB)
    URL: <https://lcb.box.com/s/xseghpsq2t4i1musxj6mgd7b8rhxe7bm>

"""
# Standard imports:
from datetime import datetime
import os
from typing import Optional

# External imports:
from cannlytics.data.ccrs import (
    get_datafiles,
    merge_datasets,
    unzip_datafiles,
)
from cannlytics.data.ccrs.constants import CCRS_DATASETS
from cannlytics.utils import rmerge, sorted_nicely
import pandas as pd


def calc_daily_sales(
        df: pd.DataFrame,
        stats: dict,
    ) -> dict:
    """Calculate sales by licensee by day.
    Note: The absolute value of the `Discount` is used.
    """
    group = ['LicenseeId', 'SaleDate']
    daily = df.groupby(group, as_index=False).sum()
    for _, row in daily.iterrows():
        licensee_id = row['LicenseeId']
        date = row['SaleDate'].isoformat()[:10]
        licensee_data = stats.get(licensee_id, {})
        date_data = licensee_data.get(date, {})
        licensee_data[date] = {
            'total_price': date_data.get('total_price', 0) + row['UnitPrice'],
            'total_discount': date_data.get('total_discount', 0) + abs(row['Discount']),
            'total_sales_tax': date_data.get('total_sales_tax', 0) + row['SalesTax'],
            'total_other_tax': date_data.get('total_other_tax', 0) + row['OtherTax'],
        }
        stats[licensee_id] = licensee_data
    return stats


def save_licensee_items_by_month(
        df: pd.DataFrame,
        data_dir: str,
        item_type: Optional[str] = 'sales',
        subset: Optional[str] = '',
        parse_dates: Optional[list] =None,
        dtype: Optional[dict] = None,
        verbose: Optional[bool] = True,
    ) -> None:
    """Save items by licensee by month to licensee-specific directories.
    Note: Datafiles must be under 1 million items.
    """
    licensees = list(df['LicenseeId'].unique())
    for licensee_id in licensees:
        licensee_dir = os.path.join(data_dir, licensee_id)
        if not os.path.exists(licensee_dir): os.makedirs(licensee_dir)
        licensee_items = df.loc[df['LicenseeId'] == licensee_id]
        months = list(licensee_items['month'].unique())
        for month in months:
            outfile = f'{licensee_dir}/{item_type}-{licensee_id}-{month}.xlsx'
            month_items = licensee_items.loc[licensee_items['month'] == month]
            try:
                existing_items = pd.read_excel(
                    outfile,
                    parse_dates=parse_dates,
                    dtype=dtype,
                )
                month_items = pd.concat([existing_items, month_items])
                month_items[subset] = month_items[subset].astype(str)
                month_items.drop_duplicates(subset=subset, keep='last', inplace=True)
            except FileNotFoundError:
                pass
            month_items.to_excel(outfile, index=False)
            if verbose:
                print('Saved', licensee_id, month, 'items:', len(month_items))


def save_stats_by_month(
        df: pd.DataFrame,
        data_dir: str,
        series: str,
    ) -> None:
    """Save given series statistics by month to given data directory."""
    df['month'] = df['date'].apply(lambda x: x[:7])
    months = list(df['month'].unique())
    for month in months:
        outfile = f'{data_dir}/{series}-{month}.xlsx'
        month_stats = df.loc[df['month'] == month]
        month_stats.to_excel(outfile, index=False)


def stats_to_df(stats: dict[dict]) -> pd.DataFrame:
    """Compile statistics from a dictionary of dictionaries into a DataFrame."""
    data = []
    for licensee_id, dates in stats.items():
        for date, values in dates.items():
            data.append({
                'licensee_id': licensee_id,
                'date': date,
                **values,
            })
    return pd.DataFrame(data)


#------------------------------------------------------------------------------
# DEV: Once stabilized, write the code below into function.
#------------------------------------------------------------------------------

# def curate_ccrs_sales(data_dir, stats_dir):
#     """Curate CCRS sales by merging additional datasets."""

# Specify where your data lives.
base = 'D:\\data\\washington\\'
data_dir = f'{base}\\CCRS PRR (12-7-22)\\CCRS PRR (12-7-22)\\'
stats_dir = f'{base}\\ccrs-stats\\'

print('Curating sales...')
start = datetime.now()

# Unzip all CCRS datafiles.
unzip_datafiles(data_dir)

# Create stats directory if it doesn't already exist.
licensees_dir = os.path.join(stats_dir, 'licensee_stats')
sales_dir = os.path.join(stats_dir, 'sales')
if not os.path.exists(licensees_dir): os.makedirs(licensees_dir)
if not os.path.exists(stats_dir): os.makedirs(sales_dir)

# Define all sales fields.
# Note: `IsDeleted` throws a ValueError if it's a bool.
fields = CCRS_DATASETS['sale_details']['fields']
date_fields = CCRS_DATASETS['sale_details']['date_fields']
item_cols = list(fields.keys()) + date_fields
item_types = {k: fields[k] for k in fields if k not in date_fields}
item_types['IsDeleted'] = 'string'

# Iterate over all sales items files to calculate stats.
daily_licensee_sales = {}
inventory_dir = os.path.join(stats_dir, 'inventory')
inventory_files = sorted_nicely(os.listdir(inventory_dir))
lab_results_dir = os.path.join(stats_dir, 'lab_results')
results_file = os.path.join(lab_results_dir, 'lab_results_0.xlsx')
sales_items_files = get_datafiles(data_dir, 'SalesDetail_')
sales_items_files.reverse()
for i, datafile in enumerate(sales_items_files[6:]):
    print('Augmenting:', datafile)
    midpoint_start = datetime.now()

    # Read in the sales items.
    items = pd.read_csv(
        datafile,
        sep='\t',
        encoding='utf-16',
        parse_dates=date_fields,
        usecols=item_cols,
        dtype=item_types,
    )

    # Remove any sales items that were deleted.
    items = items.loc[
        (items['IsDeleted'] != 'True') &
        (items['IsDeleted'] != True)
    ]

    # Iterate over the sales headers until all items have been augmented.
    # Note: There is probably a clever way to reduce the number of times
    # that the headers are read. Currently reads all sale headers from
    # current to earliest then reads earliest to current for the
    # 2nd half to try to reduce unnecessary reads.
    if i < len(sales_items_files) / 2:
        sale_headers_files = get_datafiles(data_dir, 'SaleHeader_')
    else:
        sale_headers_files = get_datafiles(data_dir, 'SaleHeader_', desc=False)
    print('Merging sale header data...')
    items = merge_datasets(
        items,
        sale_headers_files,
        dataset='sale_headers',
        on='SaleHeaderId',
        target='LicenseeId',
        how='left',
        validate='m:1',
    )

    # Augment with curated inventory.
    print('Merging inventory data...')
    for datafile in inventory_files:
        try:
            data = pd.read_excel(os.path.join(inventory_dir, datafile))
        except:
            continue
        data['InventoryId'] = data['InventoryId'].astype(str)
        # FIXME: Why are there duplicates?
        data.drop_duplicates(subset='InventoryId', keep='first', inplace=True)
        data.rename(columns={
            'inventory_id': 'InventoryId',
            'CreatedBy': 'inventory_created_by',
            'CreatedDate': 'inventory_created_date',
            'UpdatedBy': 'inventory_updated_by',
            'UpdatedDate': 'product_updated_date',
            'updatedDate': 'inventory_updated_date',
            'LicenseeId': 'inventory_licensee_id',
        }, inplace=True)
        items = rmerge(
            items,
            data,
            on='InventoryId',
            how='left',
            validate='m:1',
        )

    # Augment with curated lab results.
    print('Merging lab result data...')
    data = pd.read_excel(results_file)
    data.rename(columns={
        'inventory_id': 'InventoryId',
        'created_by': 'results_created_by',
        'created_date': 'results_created_date',
        'updated_by': 'results_updated_by',
        'updated_date': 'results_updated_date',
    }, inplace=True)
    data['InventoryId'] = data['InventoryId'].astype(str)
    items = rmerge(
        items,
        data,
        on='InventoryId',
        how='left',
        validate='m:1',
    )

    # At this stage, sales by licensee by day can be incremented.
    # print('Updating sales statistics...')
    # daily_licensee_sales = calc_daily_sales(items, daily_licensee_sales)

    # Save augmented sales to licensee-specific files by month.
    print('Saving augmented sales...')
    items['month'] = items['SaleDate'].apply(lambda x: x.isoformat()[:7])
    save_licensee_items_by_month(
        items,
        licensees_dir,
        subset='SaleDetailId',
        verbose=False,
        # FIXME: Pass item date columns and types.
        # parse_dates=list(set(date_fields + supp_date_fields)),
        # dtype={**supp_types, **item_types},
    )
    midpoint_end = datetime.now()
    print('Curated sales file in:', midpoint_end - midpoint_start)

end = datetime.now()
print('✓ Finished curating sales in', end - start)


# Planned usage:
# === Test ===
if __name__ == '__main__' and False:

    # Specify where your data lives.
    base = 'D:\\data\\washington\\'
    DATA_DIR = f'{base}\\CCRS PRR (12-7-22)\\CCRS PRR (12-7-22)\\'
    STATS_DIR = f'{base}\\ccrs-stats\\'
    curate_ccrs_sales(DATA_DIR, STATS_DIR)


#------------------------------------------------------------------------------
# DEV: Should statistics be moved to `ccrs_sales_stats`?
#------------------------------------------------------------------------------

# # Compile the statistics.
# print('Compiling licensee sales statistics...')
# stats = stats_to_df(daily_licensee_sales)

# # Save the compiled statistics.
# stats.to_excel(f'{sales_dir}/sales-by-licensee.xlsx', index=False)

# # Save the statistics by month.
# save_stats_by_month(stats, sales_dir, 'sales-by-licensee')

# Future work: Calculate and save aggregate statistics.
