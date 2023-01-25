"""
Curate CCRS Inventory
Copyright (c) 2022-2023 Cannabis Data

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
    Candace O'Sullivan-Sutherland <https://github.com/candy-o>
Created: 1/1/2023
Updated: 1/12/2023
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
    CCRS_DATASETS,
    anonymize,
    get_datafiles,
    merge_datasets,
    save_dataset,
    unzip_datafiles,
)
from cannlytics.utils import camel_to_snake, rmerge, sorted_nicely
import pandas as pd


def read_licensees():
    """Read CCRS licensees data."""
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
    columns = {'Name': 'retailer', 'DBA': 'retailer_dba'}
    return licensees.rename(columns, axis=1)


def merge_lab_results(
        results_file: str,
        directory: str,
        on: Optional[str] = 'InventoryId',
        target: Optional[str] = 'LabResultId',
        verbose: Optional[bool] = True,
    ) -> pd.DataFrame:
    """Merge lab results with items in a given directory."""
    matched = pd.DataFrame()
    lab_results = pd.read_excel(results_file)
    lab_results.rename(columns={
        'inventory_id': 'InventoryId',
        'lab_result_id': target,
    }, inplace=True)
    lab_results[on] = lab_results[on].astype(str)
    datafiles = sorted_nicely(os.listdir(directory))
    for datafile in datafiles:
        data = pd.read_excel(os.path.join(directory, datafile))
        data[on] = data[on].astype(str)
        match = rmerge(
            data,
            lab_results,
            on=on,
            how='left',
            validate='m:1',
        )
        match = match.loc[~match[target].isna()]
        matched = pd.concat([matched, match], ignore_index=True)
        if verbose:
            print('Matched', len(matched), 'lab results...')
    return matched


def curate_ccrs_inventory(data_dir, stats_dir):
    """Curate CCRS inventory by merging additional datasets."""
    print('Curating inventory...')
    start = datetime.now()

    # Unzip all CCRS datafiles.
    unzip_datafiles(data_dir)

    # Read licensees data.
    licensees = read_licensees()

    # Define all fields.
    # Note: `IsDeleted` throws a ValueError if it's a bool.
    fields = CCRS_DATASETS['inventory']['fields']
    date_fields = CCRS_DATASETS['inventory']['date_fields']
    item_cols = list(fields.keys()) + date_fields
    item_types = {k: fields[k] for k in fields if k not in date_fields}
    item_types['IsDeleted'] = 'string'

    # Create stats directory if it doesn't already exist.
    inventory_dir = os.path.join(stats_dir, 'inventory')
    if not os.path.exists(inventory_dir): os.makedirs(inventory_dir)

    # Iterate over all inventory datafiles to curate.
    inventory_files = get_datafiles(data_dir, 'Inventory_')
    product_files = get_datafiles(data_dir, 'Product_')
    strain_files = get_datafiles(data_dir, 'Strains_')
    area_files = get_datafiles(data_dir, 'Areas_')
    print(len(inventory_files), 'datafiles to curate.')
    print('Estimated runtime:', len(inventory_files) * 0.25 + 1.5, 'hours')
    for i, datafile in enumerate(inventory_files):
        print('Augmenting:', datafile)

        # Read in the items.
        items = pd.read_csv(
            datafile,
            sep='\t',
            encoding='utf-16',
            parse_dates=date_fields,
            usecols=item_cols,
            dtype=item_types,
        )

        # Merge licensee data using `LicenseeId`.
        print('Merging licensee data...')
        items = rmerge(
            items,
            licensees,
            on='LicenseeId',
            how='left',
            validate='m:1',
        )

        # Merge product data using `ProductId`.
        print('Merging product data...')
        items = merge_datasets(
            items,
            product_files,
            dataset='products',
            on='ProductId',
            target='InventoryType',
            how='left',
            validate='m:1',
            rename={
               'CreatedDate': 'product_created_at',
               'updatedDate': 'product_updated_at',
               'ExternalIdentifier': 'product_external_id',
               'LicenseeId': 'producer_licensee_id',
            },
        )

        # Merge strain `Name` using `StrainId`.
        print('Merging strain data...')
        items = merge_datasets(
            items,
            strain_files,
            dataset='strains',
            on='StrainId',
            target='StrainId',
            how='left',
            validate='m:1',
            rename={
               'Name': 'strain_name',
               'CreatedDate': 'strain_created_date',
            },
            drop=['CreatedBy', 'UpdatedBy', 'UpdatedDate'],
        )

        # Merge area `Name` using `AreaId`.
        print('Merging area data...')
        items = merge_datasets(
            items,
            area_files,
            dataset='areas',
            on='AreaId',
            target='AreaId',
            how='left',
            validate='m:1',
            rename={
               'Name': 'area_name',
            },
            drop=['LicenseeId', 'IsQuarantine', 'ExternalIdentifier',
            'IsDeleted', 'CreatedBy', 'CreatedDate', 'UpdatedBy', 'UpdatedDate']
        )

        # Save the curated inventory data.
        # FIXME: This takes a long time.
        print('Saving the curated inventory data...')
        outfile = os.path.join(inventory_dir, f'inventory_{i}.xlsx')
        items = anonymize(items)
        items.to_excel(outfile, index=False)
        print('Curated inventory datafile:', i + 1, '/', len(inventory_files))

    # Merge and save inventory data with curated lab result data.
    try:
        print('Merging lab results...')
        inventory_dir = os.path.join(stats_dir, 'inventory')
        inventory_files = sorted_nicely(os.listdir(inventory_dir))
        lab_results_dir = os.path.join(stats_dir, 'lab_results')
        results_file = os.path.join(lab_results_dir, 'lab_results_0.xlsx')
        matched = merge_lab_results(results_file, inventory_dir)
        matched.rename(columns=lambda x: camel_to_snake(x), inplace=True)
        save_dataset(matched, lab_results_dir, 'inventory_lab_results')
        print('Merged inventory items with curated lab results.')
    except:
        print('Failed to merge lab results. Curate lab results first.')

    end = datetime.now()
    print('✓ Finished curating inventory in', end - start)


# === Test ===
if __name__ == '__main__':

    # Specify where your data lives.
    base = 'D:\\data\\washington\\'
    DATA_DIR = f'{base}\\CCRS PRR (12-7-22)\\CCRS PRR (12-7-22)\\'
    STATS_DIR = f'{base}\\ccrs-stats\\'
    curate_ccrs_inventory(DATA_DIR, STATS_DIR)
