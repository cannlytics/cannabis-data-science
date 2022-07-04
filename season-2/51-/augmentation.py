"""
Data Augmentation Tools
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 2/7/2022
Updated: 2/7/2022
License: MIT License <https://opensource.org/licenses/MIT>
"""
# Standard imports.
import gc
from typing import Any, Optional

# External imports.
import pandas as pd

# Internal imports
from utils import format_millions, get_number_of_lines


def augment_dataset(
        data: Any,
        data_file: str,
        fields: dict,
        merge_key: str,
        match_key: Optional[str] = 'global_id',
        chunk_size: Optional[int] = 1_000_000,
        row_count: Optional[int] = None,
        sep: Optional[str] = '\t',
        encoding: Optional[str] = 'utf-16',
        date_columns: Optional[list] = None,
) -> Any:
    """Augment a given dataset with another dataset from its datafile, by
    follwing these steps:
        1. Read in a chunk of the augmenting dataset and iterate until all of
            its rows are read;
        2. Merge the chunk of augmenting data with the data to be augmented;
        3. Keep the augmented data.
    Args:
        data (DataFrame): The data to be augmented.
        data_file (str): The file name of the dataset used for augmenting.
        fields (dict): A dictionary of fields to merge from the augmenting dataset.
        merge_key (str): The field in the data being augmented to merge on.
        match_key (str): The field in the augmenting data to merge on,
            `global_id` by default (optional).
        chunk_size (int): The number of rows to read in the augmenting dataset
            at 1 time (optional).
        row_count (int): The number of rows in the augmenting datafile (optional).
        sep (str): The type of separation in the augmenting datafile (optional).
        encoding (str): The type of encoding of the augmenting datafile (optional).
        date_columns (list): A list of date columns in the augmenting datafile (optional).
    Returns:
    """
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
        data = pd.merge(
            left=data,
            right=shard,
            how='left',
            left_on=merge_key,
            right_on=merge_key,
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
        print('Augmented %i / %i (%i%%) observations from %s' %
              (format_millions(read_rows), format_millions(row_count),
               percent_read, data_file))
    del shard
    gc.collect()
    return data
