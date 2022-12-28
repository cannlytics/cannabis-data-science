"""
CCRS Client | Cannlytics
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 4/10/2022
Updated: 4/17/2022
License: <https://github.com/cannlytics/cannlytics/blob/main/LICENSE>
"""
# Standard imports.
import csv

# External imports.
import pandas as pd

# # Internal imports.
from cannlytics.utils.utils import snake_case
from .constants import analytes, analyses, datasets


class CCRS(object):
    """An instance of this class handles CCRS data."""


    def __init__(self, test=True):
        """Initialize a CCRS client."""
        self.state = 'WA'
        self.test = test


    def read_lab_results(self, data_dir, limit=None):
        """Read lab results into a well-formatted DataFrame.
        Future work: Allow users to specify which fields to read.
        """
        dataset = 'LabResult_0'
        datafile = f'{data_dir}/{dataset}/{dataset}.csv'
        data = pd.read_csv(
            datafile,
            usecols=datasets['lab_results']['fields'],
            parse_dates=datasets['lab_results']['date_fields'],
            low_memory=False,
            nrows=limit,
        )
        parsed_analyses = data['TestName'].map(analytes).values.tolist()
        data = data.join(pd.DataFrame(parsed_analyses))
        data['type'] = data['type'].map(analyses)
        data.columns = [snake_case(x) for x in data.columns]
        # TODO: Exclude any test lab results.
        return data


    def read_licensees(self, data_dir):
        """Read licensee data into a well-formatted DataFrame.
            1. If a row has a value in cell 22, shift 2 to the left,
            condensing column 4 to 3 and column 6 to 5.
            2. If a row has a value in cell 21, shift 1 to the left,
            condensing column 4 to 3.
        
        Future work: Allow users to specify which fields to read.
        """
        dataset = 'Licensee_0'
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
    

    def read_transfers(self, data_dir, limit=None):
        """Read transfers into a well-formatted DataFrame.
        Future work: Allow users to specify which fields to read.
        """
        dataset = 'Transfers_0'
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
    

    # TODO: Upload to Firebase!

    # TODO: Import from Cannlytics API!
