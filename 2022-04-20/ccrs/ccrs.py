"""
CCRS Client | Cannlytics
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 4/10/2022
Updated: 4/20/2022
License: <https://github.com/cannlytics/cannlytics/blob/main/LICENSE>
"""
# Standard imports.
import csv
import os

# External imports.
import pandas as pd
import requests

# Internal imports.
from cannlytics.firebase import initialize_firebase, update_documents
from cannlytics.utils.utils import snake_case
# Planned release: cannlytics==0.0.420
# from cannlytics.ccrs.constants import analytes, analyses, datasets
from constants import analytes, analyses, datasets


class CCRS(object):
    """An instance of this class handles CCRS data."""


    def __init__(self, data_dir='C:\\data', test=True):
        """Initialize a CCRS client."""
        self.data_dir = data_dir
        self.state = 'wa'
        self.test = test
        try:
            self.db = initialize_firebase()
        except ValueError:
            self.db = None
    

    def initialize_firebase(self):
        """Initialize a Firebase client."""
        self.db = initialize_firebase()


    def read_areas(self, data_dir=None, limit=None):
        """Read areas into a well-formatted DataFrame.
        Future work: Allow users to specify which fields to read.
        """
        return self.read('Areas_0', 'areas', data_dir, limit)


    def read_contacts(self, data_dir=None, limit=None):
        """Read contacts into a well-formatted DataFrame.
        Future work: Allow users to specify which fields to read.
        """
        return self.read('Contacts_0', 'contacts', data_dir, limit)


    def read_integrators(self, data_dir=None, limit=None):
        """Read integrators into a well-formatted DataFrame.
        Future work: Allow users to specify which fields to read.
        """
        return self.read('Integrator_0', 'integrators', data_dir, limit)
    

    def read_inventory(self, data_dir=None, limit=None):
        """Read inventory into a well-formatted DataFrame.
        Future work: Allow users to specify which fields to read.
        """
        return self.read('Inventory_0', 'inventory', data_dir, limit)
    

    def read_inventory_adjustments(self, data_dir=None, limit=None):
        """Read inventory adjustments into a well-formatted DataFrame.
        Future work: Allow users to specify which fields to read.
        """
        return self.read('InventoryAdjustment_0', 'inventory_adjustments', data_dir, limit)


    def read_lab_results(self, data_dir=None, limit=None):
        """Read lab results into a well-formatted DataFrame,
        mapping analyses from `test_name` into `key`, `type`, `units`.
        Future work: Allow users to specify which fields to read.
        """
        data = self.read('LabResult_0', 'lab_results', data_dir, limit)
        parsed_analyses = data['test_name'].map(analytes).values.tolist()
        data = data.join(pd.DataFrame(parsed_analyses))
        data['type'] = data['type'].map(analyses)
        # TODO: Exclude any test lab results.
        return data


    def read_licensees(self, data_dir=None):
        """Read licensee data into a well-formatted DataFrame.
            1. If a row has a value in cell 22, shift 2 to the left,
            condensing column 4 to 3 and column 6 to 5.
            2. If a row has a value in cell 21, shift 1 to the left,
            condensing column 4 to 3.

        Future work: Allow users to specify which fields to read.
        """
        dataset = 'Licensee_0'
        if data_dir is None:
            data_dir = self.data_dir
        datafile = f'{data_dir}/{dataset}/{dataset}.csv'
        csv_list = []
        with open(datafile, 'r', encoding='latin1') as f:
            for line in csv.reader(f):
                csv_list.append(line)
        headers = csv_list[:1][0]
        raw_data = pd.DataFrame(csv_list[1:])
        csv_list = []
        # FIXME: Some rows are even longer due to addresses.
        for _, values in raw_data.iterrows():
            if values[22]:
                values[5] = values[5] + values[6]
                values[3] = values[3] + values[4]
                values.pop(6)
                values.pop(4)
            elif values[21]:
                values[3] = values[3] + values[4]
                values.pop(4)
            csv_list.append(values)
        data = pd.DataFrame(csv_list)
        data.columns = headers + [''] * (len(data.columns) - len(headers))
        data.drop('', axis=1, inplace=True)
        for key in datasets['licensees']['date_fields']:
            data[key] = pd.to_datetime(data[key], errors='coerce')
        data.columns = [snake_case(x) for x in data.columns]
        # TODO: Clean names more elegantly?
        data['name'] = data['name'].str.title()
        data['dba'] = data['dba'].str.title()
        data['city'] = data['city'].str.title()
        data['county'] = data['county'].str.title()
        data['license_number'] = data['license_number'].str.strip()
        return data


    def read_plants(self, data_dir=None, limit=None):
        """Read plants into a well-formatted DataFrame.
        Future work: Allow users to specify which fields to read.
        """
        return self.read('Plant_0', 'plants', data_dir, limit)
    

    def read_plant_destructions(self, data_dir=None, limit=None):
        """Read plant destructions into a well-formatted DataFrame.
        Future work: Allow users to specify which fields to read.
        """
        return self.read('PlantDestructions_0', 'plant_destructions', data_dir, limit)


    def read_products(self, data_dir=None, limit=None):
        """Read products into a well-formatted DataFrame.
        Future work: Allow users to specify which fields to read.
        """
        return self.read('Product_0', 'products', data_dir, limit)


    def read_sale_headers(self, data_dir=None, limit=None):
        """Read sale headers into a well-formatted DataFrame.
        Future work: Allow users to specify which fields to read.
        """
        return self.read('SaleHeader_0', 'sale_headers', data_dir, limit)


    def read_sale_details(self, data_dir=None, limit=None):
        """Read sale headers into a well-formatted DataFrame.
        Future work: Allow users to specify which fields to read.
        """
        return self.read('SalesDetail_0', 'sale_details', data_dir, limit)


    def read_strains(self, data_dir=None, limit=None):
        """Read strains into a well-formatted DataFrame.
        Future work: Allow users to specify which fields to read.
        """
        return self.read('Strains_0', 'strains', data_dir, limit)
    

    def read_transfers(self, data_dir=None, limit=None):
        """Read transfers into a well-formatted DataFrame.
        Future work: Allow users to specify which fields to read.
        """
        dataset = 'Transfers_0'
        if data_dir is None:
            data_dir = self.data_dir
        datafile = f'{data_dir}/{dataset}/{dataset}.xlsx'
        data = pd.read_excel(
            datafile,
            usecols=datasets['transfers']['fields'],
            parse_dates=datasets['transfers']['date_fields'],
            nrows=limit,
            skiprows=2,
        )
        data.columns = [snake_case(x) for x in data.columns]
        return data


    def read(self, dataset, dataset_name,  data_dir=None, limit=None):
        """Read a dataset from local storage."""
        if data_dir is None:
            data_dir = self.data_dir
        datafile = f'{data_dir}/{dataset}/{dataset}.csv'
        data = pd.read_csv(
            datafile,
            low_memory=False,
            nrows=limit,
            parse_dates=datasets[dataset_name]['date_fields'],
            usecols=datasets[dataset_name]['fields'],
        )
        data.columns = [snake_case(x) for x in data.columns]
        return data


    def save(self, data, destination, index_col=True):
        """Save data to local storage."""
        if destination.endswith('.csv'):
            data.to_csv(destination, index_col=index_col)
        else:
            data.to_excel(destination, index_col=index_col)


    def upload(self, data, dataset, id_field='id', refs=None):
        """Upload a dataset to a Firebase Firestore NoSQL database."""
        items = data.to_dict(orient='records')
        # list(map(lambda x:Reading(h=x[0],p=x[1]),df.values.tolist()))
        if refs is None:
            refs = [f"data/ccrs/{dataset}/{x[id_field]}" for x in items]
        update_documents(refs, items, database=self.db)
        return items


    def get(
            self,
            dataset,
            base='https://cannlytics.com/api',
            limit=None,
    ):
        """Get a dataset from the Cannlytics API.
        Reads Cannlytics API key from `CANNLYTICS_API_KEY` environmet variable."""
        api_key = os.environ['CANNLYTICS_API_KEY']
        headers = {
            'Authorization': 'Bearer %s' % api_key,
            'Content-type': 'application/json',
        }
        url = f'{base}/data/ccrs/{dataset}'
        response = requests.get(url, headers=headers)
        return response.json()


    def get_lab_results(self):
        """Get lab results from the Cannlytics API."""
        return self.get('lab_results')
