"""
Curate CCRS Lab Results
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
    CCRS_ANALYTES,
    CCRS_ANALYSES,
    CCRS_DATASETS,
    anonymize,
    get_datafiles,
    find_detections,
    format_test_value,
    save_dataset,
    unzip_datafiles,
)
from cannlytics.utils import convert_to_numeric, camel_to_snake
import pandas as pd


def read_lab_results(
        data_dir: str,
        value_key: Optional[str] = 'TestValue',
    ) -> pd.DataFrame:
    """Read CCRS lab results."""
    lab_results = pd.DataFrame()
    lab_result_files = get_datafiles(data_dir, 'LabResult_')
    fields = CCRS_DATASETS['lab_results']['fields']
    parse_dates = CCRS_DATASETS['lab_results']['date_fields']
    usecols = list(fields.keys()) + parse_dates
    dtype = {k: v for k, v in fields.items() if v != 'datetime64'}
    dtype[value_key] = 'string' # Hot-fix for `ValueError`.
    for datafile in lab_result_files:
        data = pd.read_csv(
            datafile,
            sep='\t',
            encoding='utf-16',
            engine='python',
            parse_dates=parse_dates,
            dtype=dtype,
            usecols=usecols,
            on_bad_lines='skip',
            # DEV: Uncomment to make development quicker.
            # nrows=1000,
        )
        lab_results = pd.concat([lab_results, data])
    values = lab_results[value_key].apply(convert_to_numeric)
    lab_results = lab_results.assign(TestValue=values)
    return lab_results


def augment_lab_results(
        results: pd.DataFrame,
        item_key: Optional[str] = 'InventoryId',
        analysis_name: Optional[str] = 'TestName',
        analysis_key: Optional[str] = 'TestValue',
        verbose: Optional[str] = True,
    ) -> pd.DataFrame:
    """Format CCRS lab results to merge into another dataset."""

    # Augment lab results with standard analyses and analyte keys.
    analyte_data = results[analysis_name].map(CCRS_ANALYTES).values.tolist()
    results = results.join(pd.DataFrame(analyte_data))
    results['type'] = results['type'].map(CCRS_ANALYSES)

    # Find lab results for each item.
    # FIXME: This is ridiculously slow. Is there any way to optimize?
    curated_results = []
    item_ids = list(results[item_key].unique())
    drop = [
        analysis_name,
        analysis_key,
        'LabTestStatus',
        'key',
        'type',
        'units',
    ]
    if verbose:
        print('Curating', len(item_ids), 'items...')
        print('Estimated runtime:', len(item_ids) * 0.00016, 'hours')
    for n, item_id in enumerate(item_ids):
        item_results = results.loc[results[item_key].astype(str) == item_id]
        if item_results.empty:
            continue

        # Map certain test values.
        item = item_results.iloc[0].to_dict()
        [item.pop(key) for key in drop]
        entry = {
            **item,
            'delta_9_thc': format_test_value(item_results, 'delta_9_thc'),
            'thca': format_test_value(item_results, 'thca'),
            'total_thc': format_test_value(item_results, 'total_thc'),
            'cbd': format_test_value(item_results, 'cbd'),
            'cbda': format_test_value(item_results, 'cbda'),
            'total_cbd': format_test_value(item_results, 'total_cbd'),
            'moisture_content': format_test_value(item_results, 'moisture_content'),
            'water_activity': format_test_value(item_results, 'water_activity'),
        }

        # Determine `status`.
        statuses = list(item_results['LabTestStatus'].unique())
        if 'Fail' in statuses:
            entry['status'] = 'Fail'
        else:
            entry['status'] = 'Pass'

        # Augment the `results`.
        entry_results = []
        for _, item_result in item_results.iterrows():
            test_name = item_result['TestName']
            analyte = CCRS_ANALYTES[test_name]
            try:
                analysis = CCRS_ANALYSES[analyte['type']]
            except KeyError:
                print('Unidentified analysis:', analyte['type'])
                analysis = analyte['type']
            entry_results.append({
                'analysis': analysis,
                'key': analyte['key'],
                'name': item_result['TestName'],
                'units': analyte['units'],
                'value': item_result['TestValue'],
            })
        entry['results'] = entry_results

        # Determine detected contaminants.
        entry['pesticides'] = find_detections(entry_results, 'pesticides')
        entry['residual_solvents'] = find_detections(entry_results, 'residual_solvents')
        entry['heavy_metals'] = find_detections(entry_results, 'heavy_metals')

        # Record the lab results for the item.
        curated_results.append(entry)
        if verbose and (n + 1) % 1_000 == 0:
            percent = round((n + 1) / len(item_ids) * 100, 2)
            print(f'Curated: {n + 1} ({percent}%)')

    # Return the curated lab results.
    return pd.DataFrame(curated_results)


def curate_ccrs_lab_results(data_dir, stats_dir):
    """Curate CCRS lab results."""

    # Start curating lab results.
    print('Curating lab results...')
    start = datetime.now()

    # Unzip all CCRS datafiles.
    unzip_datafiles(data_dir)

    # Read all lab results.
    lab_results = read_lab_results(data_dir)

    # Curate all lab results.
    lab_results = augment_lab_results(lab_results)

    # Save the curated lab results.
    lab_results_dir = os.path.join(stats_dir, 'lab_results')
    lab_results = anonymize(lab_results)
    lab_results.rename(columns={'ExternalIdentifier': 'lab_id'}, inplace=True)
    lab_results.rename(columns=lambda x: camel_to_snake(x), inplace=True)
    save_dataset(lab_results, lab_results_dir, 'lab_results')

    # Finish curating lab results.
    end = datetime.now()
    print('✓ Finished curating lab results in', end - start)


# === Test ===
if __name__ == '__main__':

    # Specify where your data lives.
    base = 'D:\\data\\washington\\'
    DATA_DIR = f'{base}\\CCRS PRR (12-7-22)\\CCRS PRR (12-7-22)\\'
    STATS_DIR = f'{base}\\ccrs-stats\\'
    curate_ccrs_lab_results(DATA_DIR, STATS_DIR)
