"""
Massachusetts Cannabis Control Commission (CCC) Open Data API Wrapper
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 4/5/2022
Updated: 4/5/2022
License: MIT License <https://opensource.org/licenses/MIT>

Data sources:

    - Massachusetts Cannabis Control Commission Data Catalog
    https://masscannabiscontrol.com/open-data/data-catalog/

TODO: Create a data guide.

"""
import os
import pandas as pd
from requests import Session
from simplejson import JSONDecodeError


BOOLEAN = [
    'activity_date',
    'app_create_date',
    'facilityisexpired',
    'facilityismedical',
    'facilityisretail',
    'facilityiscultivation',
    'facilityisdispensaryorstore',
    'facilityisinfusedmanufacturer',
    # is_abutters_notified
    'not_a_dbe',
    'priority',
]
DATETIME = [
    'activitysummarydate',
    'sale_period',
    'saledate',
]
NUMERIC = [
    'abutters_count',
    'application_fee',
    'average_spent',
    'countbasedtotal',
    'dollarcountbasedtotal',
    'dollarweightbasedtotal',
    'gross_sales',
    'harvestactivecount',
    'latitude',
    'lic_fee_amount',
    'longitude',
    'percent_total',
    'plantdestroyedcount',
    'plantfloweringcount',
    'plantharvestedcount',
    'plantvegetativecount',
    'price_per_ounce',
    'quantity',
    'square_footage_establishment',
    'strainactivecount',
    'total',
    'total_dollars',
    'totalprice',
    'total_units',
    'units',
    'weightbasedtotal',
]
RENAME = {
    '= 1 oz': 'price_per_ounce',
}


class APIError(Exception):
    """A primary error raised by the Open Data API."""


    def __init__(self, response):
        message = self.get_response_messages(response)
        super().__init__(message)
        self.response = response


    def get_response_messages(self, response):
        """Extract error messages from a Open Data API response.
        Args:
            response (Response): A request response from the Metrc API.
        Returns:
            (str): Returns any error messages.
        """
        try:
            return response.json()
        except JSONDecodeError:
            return response.text


class CCC(object):
    """An instance of this class communicates with the
    Cannabis Control Commission of the Commonwealth of Massachusetts'
    Open Data catalog."""


    def __init__(self):
        """Initialize a CCC API client."""
        self.base = 'https://masscannabiscontrol.com/resource/'
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1667.0 Safari/537.36'}
        self.session = Session()
        self.endpoints = {
            'agent-gender-stats': 'hhjg-atjk',
            'agent-ethnicity-stats': 'pt2c-wb44',
            'licensees': 'albs-all',
            'licensees-approved': 'hmwt-yiqy',
            'licensees-pending': 'piib-tj3f',
            'licensees-demographics': '5dkg-e39p',
            'licensees-under-review-stats': 'pebi-jpc4',
            'licensees-application-stats': 'n6qz-us6r',
            'retail-sales-stats': '87rp-xn9v',
            'retail-sales-weekly': 'dt9b-i6ds',
            'retail-price-per-ounce': 'rqtv-uenj',
            'medical-stats': 'g5mj-5pg3',
            'plants': 'meau-plav',
            'sales': 'fren-z7jq',
        }


    def get(self, endpoint, params=None):
        """Make a request to an API."""
        url = os.path.join(self.base, f'{endpoint}.json')
        try:
            response = self.session.get(url, headers=self.headers, params=params)
        except ConnectionError:
            self.session = Session()
            response = self.session.get(url, headers=self.headers, params=params)
        if response.status_code == 200:
            data = pd.DataFrame(response.json())
            data.columns = map(str.lower, data.columns)
            data.columns = [x.replace('$', 'dollars').replace('%', 'percent') for x in data.columns]
            data.rename(columns=RENAME, inplace=True)
            for key in list(set(BOOLEAN).intersection(data.columns)):
                data[key] = data[key].astype(bool, errors='ignore')
            for key in list(set(DATETIME).intersection(data.columns)):
                data[key] = pd.to_datetime(data[key], errors='ignore')
            for key in list(set(NUMERIC).intersection(data.columns)):
                data[key] = data[key].astype(float, errors='ignore')
            return data
        else:
            raise APIError(response)


    def get_agents(self, dataset='gender-stats'):
        """Get agent statistics.
        Args:
            dataset (str): An optional dataset filter:
                * `gender-stats` (default)
                * `ethnicity-stats`
        """
        key = 'agent'
        if dataset:
            key += '-' + dataset
        endpoint = self.endpoints[key]
        return self.get(endpoint)


    def get_licensees(
            self,
            dataset='',
            limit=10_000,
            order_by='app_create_date',
            ascending=False,
    ):
        """Get Massachussetts licensee data and statistics.
        Args:
            dataset (str): An optional dataset filter:
                * `approved`
                * `pending`
                * `demographics`
                * `under-review-stats`
                * `application-stats`
            limit (int): A limit to the number of returned observations.
            order_by (str): The field to order the results, `app_create_date` by default.
            ascending (bool): If ordering results, ascending or descending. Descending by default.
        """
        key = 'licensees'
        if dataset:
            key += '-' + dataset
        endpoint = self.endpoints[key]
        params = {
            '$limit': limit,
            '$order': order_by,
        }
        if not ascending:
            params['$order'] += ' DESC'
        return self.get(endpoint, params=params)


    def get_retail(
            self,
            dataset='sales-stats',
            limit=10_000,
            order_by='date',
            ascending=False,
    ):
        """Get Massachusetts retail data and statistics.
        Args:
            dataset (str): An optional dataset filter:
                * `sales-stats` (default)
                * `sales-weekly`
                * `price-per-ounce`
            limit (int): A limit to the number of returned observations.
            order_by (str): The field to order the results, `app_create_date` by default.
            ascending (bool): If ordering results, ascending or descending. Descending by default.
        """
        key = 'retail'
        key += '-' + dataset
        endpoint = self.endpoints[key]
        params = {
            '$limit': limit,
            '$order': order_by,
        }
        if not ascending:
            params['$order'] += ' DESC'
        return self.get(endpoint, params=params)


    def get_medical(self, dataset='stats'):
        """Get Massachussetts medical stats."""
        key = 'medical'
        key += '-' + dataset
        endpoint = self.endpoints[key]
        return self.get(endpoint)


    def get_plants(
            self,
            limit=10_000,
            order_by='activitysummarydate',
            ascending=False,
    ):
        """Get Massachusetts cultivation data and statistics.
        Args:
            limit (int): A limit to the number of returned observations.
            order_by (str): The field to order the results, `app_create_date` by default.
            ascending (bool): If ordering results, ascending or descending. Descending by default.
        """
        key = 'plants'
        endpoint = self.endpoints[key]
        params = {
            '$limit': limit,
            '$order': order_by,
        }
        if not ascending:
            params['$order'] += ' DESC'
        return self.get(endpoint, params=params)


    def get_sales(
            self,
            limit=10_000,
            order_by='activitysummarydate',
            ascending=False,
    ):
        """Get Massachusetts sales data.
        Args:
            limit (int): A limit to the number of returned observations.
            order_by (str): The field to order the results, `app_create_date` by default.
            ascending (bool): If ordering results, ascending or descending. Descending by default.
        """
        key = 'sales'
        endpoint = self.endpoints[key]
        params = {
            '$limit': limit,
            '$order': order_by,
        }
        if not ascending:
            params['$order'] += ' DESC'
        return self.get(endpoint, params=params)
