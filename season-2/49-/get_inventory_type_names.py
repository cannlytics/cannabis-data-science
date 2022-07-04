"""
Get inventory type names from Washington Leaf traceability data.
Cannabis Data Science Meetup Group
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 1/18/2022
Updated: 1/19/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script extracts the names from the inventory types data and
saves them in a smaller file than the original.

Data Sources:

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd?page=1

    - Leaf Data Systems Data Guide:
    https://lcb.wa.gov/sites/default/files/publications/Marijuana/traceability/WALeafDataSystems_UserManual_v1.37.5_AddendumC_LicenseeUser.pdf

"""
# Standard imports.
import gc

# External imports.
import pandas as pd

# Internal imports.
from utils import get_number_of_lines

# Define the inventory types file.
DATA_FILE = '../.datasets/InventoryTypes_0.csv'
SAVE_FILE = '../.datasets/inventory_type_names.csv'


def get_inventory_type_names():
    """Creates a smaller inventory types file with only the inventory type name."""

    # Define the number of rows.
    inventory_type_rows = get_number_of_lines(DATA_FILE) # 57_016_229

    # Define inventory type fields.
    inventory_type_fields = {
        'global_id': 'string',
        'name': 'string',
    }
    inventory_type_columns = list(inventory_type_fields.keys())

    # Read inventory type names in chunks.
    chunk_size = 1_000_000
    read_rows = 0
    skiprows = None
    inventory_names = []
    while read_rows < inventory_type_rows:

        # Specify the chunk.
        if read_rows:
            skiprows = [i for i in range(1, read_rows)]

        # Read in inventory types chunk.
        inventory_types = pd.read_csv(
            DATA_FILE,
            sep='\t',
            encoding='utf-16',
            usecols=inventory_type_columns,
            dtype=inventory_type_fields,
            index_col='global_id',
            skiprows=skiprows,
            nrows=chunk_size,
        )
        inventory_types.rename(columns={'name': 'inventory_name'}, inplace=True)

        # Keep the necessary fields.
        inventory_names.append(inventory_types)
        read_rows += chunk_size
        print('Read:', read_rows)

    # Create a small file of inventory type names.
    inventory_type_names = pd.concat(inventory_names)
    inventory_type_names.to_csv(SAVE_FILE)
    del inventory_names
    del inventory_types
    gc.collect()


if __name__ == '__main__':

    get_inventory_type_names()
