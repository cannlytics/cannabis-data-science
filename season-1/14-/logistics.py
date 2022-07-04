# -*- coding: utf-8 -*-
"""
Logistics | Cannlytics
Copyright © 2021 Cannlytics
Author: Keegan Skeate <keegan@cannlytics.com>
Created: 1/10/2021

License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
"""
from googlemaps import Client, places
from firebase_admin import initialize_app, firestore
from time import sleep


def get_google_maps_api_key():
    """Get Google Maps API key."""
    db = firestore.client()
    admin = db.collection("admin")
    google = admin.document("google").get()
    google_data = google.to_dict()
    return google_data["google_maps_api_key"]


def geocode_addresses(df):
    """Geocode addresses in a dataframe."""
    try:
        initialize_app()
    except ValueError:
        pass
    api_key = get_google_maps_api_key()
    gmaps = Client(key=api_key)
    for index, item in df.iterrows():
        # FIXME: Handle existing lat and long more elegantly.
        # try:
        #     if item.latitude and item.longitude:
        #         continue
        # except:
        #     pass
        address = f"{item.street}, {item.city}, {item.state} {item.zip}"
        geocode_result = gmaps.geocode(address)
        if geocode_result:
            df.at[index, "formatted_address"] = geocode_result[0]["formatted_address"]
            location = geocode_result[0]["geometry"]["location"]
            print(item.name, "-->", location)
            df.at[index, "latitude"] = location["lat"]
            df.at[index, "longitude"] = location["lng"]
            # TODO: Round latitude and longitude (4-6 decimal places?)
            for info in geocode_result[0]["address_components"]:
                key = info["types"][0]
                if key == "administrative_area_level_2":
                    df.at[index, "county"] = info["long_name"]

        sleep(2)  # Prevents spamming Google's servers (necessary?)
    return df


def search_for_address(query, api_key=None, fields=["formatted_address"]):
    """Search for the address of a given name."""
    if api_key is None:
        api_key = get_google_maps_api_key()
    gmaps = Client(key=api_key)
    place = places.find_place(gmaps, query, "textquery", fields=fields)
    return place["candidates"]


def get_place_details(query, api_key=None, fields=[]):
    """Get the place details for a given a name."""
    if api_key is None:
        api_key = get_google_maps_api_key()
    if not fields:
        fields = [
            "formatted_address",
            "photo",
            "opening_hours",
            "website",
        ]
    gmaps = Client(key=api_key)
    search = places.find_place(gmaps, query, "textquery")
    place_id = search["candidates"][0]["place_id"]
    place = places.place(gmaps, place_id, fields=fields)
    return place["result"]
