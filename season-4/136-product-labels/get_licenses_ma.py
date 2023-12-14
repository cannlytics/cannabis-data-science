"""
Cannabis Licenses | Get Massachusetts Licenses
Copyright (c) 2022-2023 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
    Candace O'Sullivan-Sutherland <https://github.com/candy-o>
Created: 9/29/2022
Updated: 9/19/2023
License: <https://github.com/cannlytics/cannlytics/blob/main/LICENSE>

Description:

    Collect Massachusetts cannabis license data.

Data Source:

    - Massachusetts Cannabis Control Commission Data Catalog
    URL: <https://masscannabiscontrol.com/open-data/data-catalog/>

"""
# Standard imports.
from datetime import datetime
import os
from typing import Optional

# External imports.
from cannlytics.data import OpenData
from cannlytics.utils import snake_case


# Specify where your data lives.
DATA_DIR = './data/ma'

# Specify state-specific constants.
STATE = 'MA'
MASSACHUSETTS = {
    'licensing_authority_id': 'MACCC',
    'licensing_authority': 'Massachusetts Cannabis Control Commission',
    'licenses': {
        'columns': {
            'mailing_address_1': 'mailing_address_1',
            'zip': 'premise_zip_code',
            'ee_priority_number_from_account': 'ee_priority_number_from_account',
            'establishment_city': 'premise_city',
            'rmd_priority_number_from_account': 'rmd_priority_number_from_account',
            'cultivation_environment': 'cultivation_environment',
            'application_classification_split_2': 'application_classification_split_2',
            'mailing_city': 'mailing_city',
            'latitude': 'premise_latitude',
            'ee_priority_number_from_application': 'ee_priority_number_from_application',
            'business_address_2': 'business_address_2',
            'establishment_zipcode': 'establishment_zipcode',
            'submitted_date': 'submitted_date',
            'county_lat': 'county_lat',
            'mailing_address_2': 'mailing_address_2',
            'approved_license_type': 'approved_license_type',
            'longitude': 'premise_longitude',
            'license_type': 'license_type',
            'application_number': 'application_number',
            'first_compliance_review_date': 'first_compliance_review_date',
            'establishment_address_2': 'establishment_address_2',
            'cccupdatedate': 'cccupdatedate',
            'license_number': 'license_number',
            'business_name': 'business_legal_name',
            'business_state': 'business_state',
            'lic_fee_amount': 'lic_fee_amount',
            'region': 'region',
            'business_city': 'business_city',
            'fee_waiver_ee_or_se_number': 'fee_waiver_ee_or_se_number',
            'geocoded_county': 'geocoded_county',
            'business_zipcode': 'business_zipcode',
            'application_approved_date': 'application_approved_date',
            'town': 'town',
            'cultivation_tier': 'cultivation_tier',
            'application_classification': 'application_classification',
            'application_status': 'application_status',
            'ein_tin': 'ein_tin',
            'priority': 'priority',
            'dbe': 'dbe',
            'rmd_priority_status': 'rmd_priority_status',
            'lic_status': 'license_status',
            'lic_fee_payment_submitted_date': 'lic_fee_payment_submitted_date',
            'establishment_state': 'establishment_state',
            'business_address_1': 'business_address_1',
            'ee_priority_status': 'ee_priority_status',
            'cnb_deemed_complete_date': 'cnb_deemed_complete_date',
            'mailing_zipcode': 'mailing_zipcode',
            'lic_expiration_date': 'expiration_date',
            'commence_operations_date': 'commence_operations_date',
            'establishment_county': 'establishment_county',
            'pmt_amount': 'pmt_amount',
            'fee_waiver_request': 'fee_waiver_request',
            'lic_start_date': 'lic_start_date',
            'county': 'premise_county',
            'application_classification_split_1': 'application_classification_split_1',
            'establishment_address': 'establishment_address',
            'county_long': 'county_long',
            'mailing_state': 'mailing_state',
            'approved_license_stages': 'approved_license_stages',
            'application_created_date': 'issue_date',
            'rmd_priority_number_from_application': 'rmd_priority_number_from_application',
            'geocoded_address': 'geocoded_address',
            'cnb_dt_of_final_licensure': 'cnb_dt_of_final_licensure',
        }
    }
}


def get_licenses_ma(
        data_dir: Optional[str] = None,
        **kwargs,
    ):
    """Get Massachusetts cannabis license data."""

    # Get the licenses data.
    ccc = OpenData()
    licenses = ccc.get_licensees('approved')

    # Standardize the fields.
    columns = MASSACHUSETTS['licenses']['columns']
    licenses.columns = [snake_case(col) for col in licenses.columns]
    licenses.rename(columns=columns, inplace=True)
    licenses['business_dba_name'] = licenses['business_legal_name']

    # Remove duplicates.
    licenses.drop_duplicates(subset=['business_legal_name'], inplace=True)

    # Standardize the licenses data.
    licenses = licenses.assign(
        premise_state=STATE,
        licensing_authority_id=MASSACHUSETTS['licensing_authority_id'],
        licensing_authority=MASSACHUSETTS['licensing_authority'],
        # business_dba_name=licenses['business_legal_name'],
        business_structure=None,
        business_email=None,
        business_owner_name=None,
        parcel_number=None,
        business_image_url=None,
        business_website=None,
        business_phone=None,
    )

    # Optional: Look-up business websites for each license.

    # Save and return the data.
    if data_dir is not None:
        if not os.path.exists(data_dir): os.makedirs(data_dir)
        date = datetime.now().strftime('%Y-%m-%d')
        retailers = licenses.loc[licenses['license_type'].str.contains('Retailer')]
        retailers.to_csv(f'{data_dir}/retailers-{STATE.lower()}-{date}.csv', index=False)
        licenses.to_csv(f'{data_dir}/licenses-{STATE.lower()}-{date}.csv', index=False)
        licenses.to_csv(f'{data_dir}/licenses-{STATE.lower()}-latest.csv', index=False)
    return licenses


# === Test ===
# [✓] Tested: 2023-09-19 by Keegan Skeate <keegan@cannlytics>
if __name__ == '__main__':

    # Support command line usage.
    import argparse
    try:
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('--d', dest='data_dir', type=str)
        arg_parser.add_argument('--data_dir', dest='data_dir', type=str)
        args = arg_parser.parse_args()
    except SystemExit:
        args = {'d': DATA_DIR}

    # Get licenses, saving them to the specified directory.
    # FIXME: This results in a different number of licenses each time!
    data_dir = args.get('d', args.get('data_dir'))
    data = get_licenses_ma(data_dir)
    print(len(data))
