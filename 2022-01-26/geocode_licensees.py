"""
GeoCode Licensees | Cannlytics
Copyright (c) 2021-2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: Wed May 12 18:50:27 2021
Updated: 1/25/2022
License: MIT License <https://opensource.org/licenses/MIT>
"""
# Standard imports.
import json
from time import sleep

# External imports.
import pandas as pd
from dotenv import dotenv_values
from googlemaps import Client


def geocode_addresses(df, api_key):
    """Geocode addresses in a dataframe given that the dataframe has fields:
         - street
         - city
         - state
         - zip
    A 1 second pause is applied between requests to honor Google's usage limits.
    https://developers.google.com/maps/documentation/javascript/usage-and-billing#other-usage-limits
    """
    gmaps = Client(key=api_key)
    for index, item in df.iterrows():
        address = f'{item.street}, {item.city}, {item.state} {item.zip}'
        geocode_result = gmaps.geocode(address)
        sleep(1)
        if geocode_result:
            location = geocode_result[0]['geometry']['location']
            formatted_address = geocode_result[0]['formatted_address']
            df.at[index, 'formatted_address'] = formatted_address
            df.at[index, 'latitude'] = location['lat']
            df.at[index, 'longitude'] = location['lng']
            for info in geocode_result[0]['address_components']:
                key = info['types'][0]
                if key == 'administrative_area_level_2':
                    df.at[index, 'county'] = info['long_name']
        else:
            print('Failed to geocode:', index)
    return df

#-----------------------------------------------------------------------------
# Import the data.
#-----------------------------------------------------------------------------

if __name__ == '__main__':

    # Specify where the data lives.
    directory ='../.datasets'

    # Read in the licensee data.
    file_name = f'{directory}\Licensees_0.csv'
    licensee_data = pd.read_csv(
        file_name,
        sep='\t',
        encoding='utf-16',
    )

    # Geocode each address to identify the county.
    config = dotenv_values('../.env')
    google_maps_api_key = config['GOOGLE_MAPS_API_KEY']
    licensee_data['street'] = licensee_data['address1']
    licensee_data['state'] = 'WA'
    licensee_data['zip'] = licensee_data['postal_code']
    licensee_data = geocode_addresses(licensee_data, google_maps_api_key)

    # Save the data to csv, xlsx, and json.
    licensee_data.to_excel('../.datasets/geocoded_licensee_data.xlsx')
    licensee_data.to_csv('../.datasets/geocoded_licensee_data.csv')
    with open('../.datasets/geocoded_licensee_data.json', 'w') as outfile:
        data = licensee_data.where(pd.notnull(licensee_data), None)
        data = json.loads(json.dumps(list(data.T.to_dict().values())))
        json.dump(data, outfile)

