"""
Logistics Module | Cannabis Data Science

Author: Keegan Skeate
Created: Wed Feb 24 07:25:27 2021

License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Description:
    Geocode addresses with the Google Maps API.

Resources:
    https://developers.google.com/maps/documentation/javascript/get-api-key
    https://console.cloud.google.com/project/_/apiui/credential
    https://hackersandslackers.com/preparing-data-for-mapbox-geocoding/
    https://hackersandslackers.com/series/mapping-data-with-mapbox/

"""
# Standard library imports
import environ
import urllib
from time import sleep

# External imports
from googlemaps import Client


def get_api_key():
    """Get your Google Maps API key."""
    env = environ.Env()
    env.read_env("../.env")
    return urllib.parse.quote_plus(env("GOOGLE_MAPS_API_KEY"))


def geocode_addresses(df):
    """Geocode addresses in a dataframe.
    Args:
        df (DataFrame): A DataFrame with street, city, state, and zip columns.
    Returns:
        df (DataFrame): The input DataFrame with latitude, longitude, and county columns.
    """
    api_key = get_api_key()
    gmaps = Client(key=api_key)
    for index, item in df.iterrows():
        address = f"{item.street}, {item.city}, {item.state} {item.zip}"
        geocode_result = gmaps.geocode(address)
        if geocode_result:
            df.at[index, "formatted_address"] = geocode_result[0]["formatted_address"]
            location = geocode_result[0]["geometry"]["location"]
            print("Geocoded:", item.name, "-->", location)
            df.at[index, "latitude"] = location["lat"]
            df.at[index, "longitude"] = location["lng"]
            for info in geocode_result[0]["address_components"]:
                key = info["types"][0]
                if key == "administrative_area_level_2":
                    df.at[index, "county"] = info["long_name"]
        sleep(2)  # Prevents spamming Google's servers (necessary?)
    return df

