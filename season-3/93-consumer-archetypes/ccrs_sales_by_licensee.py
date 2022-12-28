"""
CCRS Sales by Licensee
Copyright (c) 2022 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 12/23/2022
Updated: 12/28/2022
License: <https://github.com/cannabisdata/cannabisdata/blob/main/LICENSE>

Data Source:

    - Washington State Liquor and Cannabis Board (WSLCB)
    URL: <https://lcb.box.com/s/xseghpsq2t4i1musxj6mgd7b8rhxe7bm>

"""
# Standard imports:
import gc
import os
from zipfile import ZipFile

# External imports:
from cannlytics.data.ccrs.constants import CCRS_DATASETS
from cannlytics.utils import rmerge, sorted_nicely
import pandas as pd


def get_ccrs_datafiles(data_dir, dataset='', desc=True):
    """Get all CCRS datafiles of a given type in a directory."""
    files = os.listdir(data_dir)
    datafiles = [f'{data_dir}/{f}/{f}/{f}.csv' for f in files if f.startswith(dataset)]
    datafiles = sorted_nicely(datafiles)
    if desc:
        datafiles.reverse()
    return datafiles


def unzip_ccrs(data_dir, verbose=True):
    """Unzip all CCRS datafiles."""
    zip_files = [f for f in os.listdir(data_dir) if f.endswith('.zip')]
    for zip_file in zip_files:
        filename = os.path.join(data_dir, zip_file)
        zip_dest = filename.rstrip('.zip')
        if not os.path.exists(zip_dest):
            os.makedirs(zip_dest)
        zip_ref = ZipFile(filename)
        zip_ref.extractall(zip_dest)
        zip_ref.close()
        os.remove(filename)
        if verbose:
            print('Unzipped:', zip_file)


def calc_daily_sales(df, stats):
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
        df,
        data_dir,
        item_type='sales',
        subset='',
        parse_dates=None,
        dtype=None,
    ):
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
                month_items.drop_duplicates(subset=subset, keep='last', inplace=True)
            except FileNotFoundError:
                pass
            month_items.to_excel(outfile, index=False)
            print('Saved', len(month_items), 'items for license', licensee_id)


def save_stats_by_month(data, data_dir, series):
    """Save given series statistics by month to given data directory."""
    data['month'] = data['date'].apply(lambda x: x[:7])
    months = list(data['month'].unique())
    for month in months:
        month_stats = data.loc[data['month'] == month]
        month_stats.to_excel(f'{data_dir}/{series}-{month}.xlsx', index=False)


def stats_to_df(stats: dict[dict]) -> pd.DataFrame:
    """Compile the statistics from a dictionary of dictionaries into a DataFrame."""
    data = []
    for licensee_id, dates in stats.items():
        for date, values in dates.items():
            data.append({
                'licensee_id': licensee_id,
                'date': date,
                **values,
            })
    return pd.DataFrame(data)


# === Test ===
if __name__ == '__main__':

    # Specify where your data lives.
    DATA_DIR = 'D:\\data\\washington\\ccrs-2022-11-22\\ccrs-2022-11-22\\'
    STATS_DIR = 'D:\\data\\washington\\ccrs-stats\\'

    # Create stats directory if it doesn't already exist.
    licensees_dir = os.path.join(STATS_DIR, 'licensee_stats')
    sales_dir = os.path.join(STATS_DIR, 'sales')
    if not os.path.exists(STATS_DIR): os.makedirs(STATS_DIR)
    if not os.path.exists(licensees_dir): os.makedirs(licensees_dir)
    if not os.path.exists(sales_dir): os.makedirs(sales_dir)

    # Unzip all CCRS datafiles.
    unzip_ccrs(DATA_DIR)

    # Define all sales fields.
    # Hot-fix for `IsDeleted` b/c it throws a ValueError if it's a bool.
    fields = CCRS_DATASETS['sale_details']['fields']
    date_fields = CCRS_DATASETS['sale_details']['date_fields']
    item_cols = list(fields.keys()) + date_fields
    item_types = {k: fields[k] for k in fields if k not in date_fields}
    item_types['IsDeleted'] = 'string'

    # Define all sales headers fields.
    supp_fields = CCRS_DATASETS['sale_headers']['fields']
    supp_date_fields = CCRS_DATASETS['sale_headers']['date_fields']
    supp_cols = list(supp_fields.keys()) + supp_date_fields
    supp_types = {k: supp_fields[k] for k in supp_fields if k not in supp_date_fields}

    # Read licensees data.
    licensees = pd.read_csv(
        f'{DATA_DIR}/Licensee_0/Licensee_0/Licensee_0.csv',
        sep='\t',
        encoding='utf-16',
        usecols=['LicenseeId', 'Name', 'DBA'],
        dtype={
            'LicenseeId': 'string',
            'Name': 'string',
            'DBA': 'string',
        },
    )
    licensee_columns = {'Name': 'retailer', 'DBA': 'retailer_dba'}

    # Iterate over all sales items files to calculate stats.
    daily_licensee_sales = {}
    sales_items_files = get_ccrs_datafiles(DATA_DIR, 'SalesDetail')
    inventory_files = get_ccrs_datafiles(DATA_DIR, 'Inventory')
    for i, datafile in enumerate(sales_items_files):

        # DEV: Stop iterating early.
        # if i != 0:
        #     continue

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
        items = items.loc[(items['IsDeleted'] != 'True') & (items['IsDeleted'] != True)]
        item_count = len(items)

        # Iterate over the sales headers until all items have been augmented.
        # Note: There is probably a clever way to reduce the number of times
        # that the headers are read. Currently reads all current to earliest then
        # reads earliest to current for the 2nd half to reduce unnecessary reads.
        augmented = pd.DataFrame()
        if i < len(sales_items_files) / 2:
            sale_headers_files = get_ccrs_datafiles(DATA_DIR, 'SaleHeader')
        else:
            sale_headers_files = get_ccrs_datafiles(DATA_DIR, 'SaleHeader', desc=False)
        for filename in sale_headers_files:

            # Read in the sale headers data to supplement the sales data.
            supplement = pd.read_csv(
                filename,
                sep='\t',
                encoding='utf-16',
                parse_dates=supp_date_fields,
                usecols=supp_cols,
                dtype=supp_types,
            )
            data = rmerge(items, supplement, on='SaleHeaderId', how='left')
            augmented = pd.concat([augmented, data.loc[~data['LicenseeId'].isna()]])

            # Merge with licensee data.
            licensees.rename(columns=licensee_columns, inplace=True)
            augmented = rmerge(
                augmented,
                licensees,
                on='LicenseeId',
                how='left',
                validate='m:1',
            )

            # TODO: Merge with inventory data to get `ProductId`.


            # TODO: Merge with product data to get `InventoryType`, `Name`,
            # `Description`, `UnitWeightGrams`.


            # TODO: Get all lab results with `InventoryId`.
        

            # Stop iterating once all items have been matched.
            print(
                'Matched %.2f%%' % (len(augmented) / item_count * 100),
                f'of {datafile.split("/")[-1]}'
            )
            if len(augmented) == item_count:
                break

        # Perform garbage cleaning.
        del items
        gc.collect()

        # At this stage, sales by licensee by day can be incremented.
        # daily_licensee_sales = calc_daily_sales(augmented, daily_licensee_sales)

        # Save augmented sales to licensee-specific files by month.
        augmented['month'] = augmented['SaleDate'].apply(lambda x: x.isoformat()[:7])
        save_licensee_items_by_month(
            augmented,
            licensees_dir,
            subset='SaleDetailId',
            parse_dates=list(set(date_fields + supp_date_fields)),
            dtype={**supp_types, **item_types},
        )

    # Compile the statistics.
    # stats = stats_to_df(daily_licensee_sales)
    
    # Optional: Add licensee data.

    # Save the statistics by month.
    # save_stats_by_month(stats, sales_dir, 'sales-by-licensee')

    # TODO: Calculate and save aggregate statistics.

    # Save the master file.
    # stats.to_excel(f'{sales_dir}/sales-by-licensee.xlsx', index=False)
