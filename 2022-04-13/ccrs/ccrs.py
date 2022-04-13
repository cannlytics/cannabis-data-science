"""
CCRS Client | Cannlytics
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 4/10/2022
Updated: 4/12/2022
License: <https://github.com/cannlytics/cannlytics-engine/blob/main/LICENSE>
"""
# Standard imports.
import csv

# External imports.
import pandas as pd

# # Internal imports.
from cannlytics.utils.utils import snake_case
from .constants import analytes, analysis_map, model_names


class CCRS(object):
    """An instance of this class handles CCRS data."""

    def __init__(self, test=True):
        """Initialize a CCRS client."""
        self.state = 'wa'
        self.test = test


    def read_lab_results(self, data_dir, folder):
        """Read lab results into a well-formatted DataFrame."""
        dataset = 'LabResult_0'
        lab_results = pd.read_csv(
            f'{data_dir}/{folder}/{dataset}/{dataset}.csv',
            usecols=model_names['lab_results']['fields'],
            parse_dates=model_names['lab_results']['date_fields'],
            low_memory=False,
        )
        parsed_analyses = lab_results['TestName'].map(analytes).values.tolist()
        lab_results = lab_results.join(pd.DataFrame(parsed_analyses))
        lab_results['type'] = lab_results['type'].map(analysis_map)
        lab_results.columns = [snake_case(x) for x in lab_results.columns]
        # TODO: Exclude any test lab results.
        return lab_results


    def read_licensees(self, data_dir, folder):
        """Read licensee data into a well-formatted DataFrame.
            1. If a row has a value in cell 22, shift 2 to the left,
            condensing column 4 to 3 and column 6 to 5.
            2. If a row has a value in cell 21, shift 1 to the left,
            condensing column 4 to 3.
        """
        dataset = 'Licensee_0'
        datafile = f'{data_dir}/{folder}/{dataset}/{dataset}.csv'
        csv_list = []
        with open(datafile, 'r', encoding='latin1') as f:
            for line in csv.reader(f):
                csv_list.append(line)
        headers = csv_list[:1][0]
        data = pd.DataFrame(csv_list[1:])
        csv_list = []
        for _, values in data.iterrows():
            # FIXME: Some rows are even longer due to addresses.
            if values[22]:
                values[5] = values[5] + values[6]
                values[3] = values[3] + values[4]
                values.pop(6)
                values.pop(4)
            elif values[21]:
                values[3] = values[3] + values[4]
                values.pop(4)
            csv_list.append(values)
        licensees = pd.DataFrame(csv_list)
        licensees.columns = headers + [''] * (len(licensees.columns) - len(headers))
        licensees.drop('', axis=1, inplace=True)
        licensees.columns = [snake_case(x) for x in licensees.columns]
        licensees['license_issue_date'] = pd.to_datetime(licensees['license_issue_date'], errors='ignore')
        licensees['license_expiration_date'] = pd.to_datetime(licensees['license_expiration_date'], errors='ignore')
        licensees['created_date'] = pd.to_datetime(licensees['created_date'], errors='ignore')
        licensees['updated_date'] = pd.to_datetime(licensees['updated_date'], errors='ignore')
        licensees['license_number'] = licensees['license_number'].str.strip()
        licensees['name'] = licensees['name'].str.title()
        licensees['dba'] = licensees['dba'].str.title()
        licensees['city'] = licensees['city'].str.title()
        licensees['county'] = licensees['county'].str.title()
        return licensees
