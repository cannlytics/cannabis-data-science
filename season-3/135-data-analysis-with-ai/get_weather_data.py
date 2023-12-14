"""
Get Producer Weather Data
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 11/22/2023
Updated: 11/22/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
from datetime import datetime, timedelta
from dotenv import dotenv_values
from googlemaps import Client
import pandas as pd
import requests


def fetch_weather(api_key, lat, lon, start_date, end_date):
    url = 'https://history.openweathermap.org/data/3.0/history/timemachine'
    weather_data = []
    while start_date <= end_date:
        params = {
            'lat': lat,
            'lon': lon,
            'dt': int(start_date.timestamp()),
            'appid': api_key
        }
        response = requests.get(url, params=params)
        data = response.json()
        weather_data.append({
            'temp': data['current']['temp'],
            'humidity': data['current']['humidity']
        })
        start_date += timedelta(days=1)
    return weather_data


if __name__ == '__main__':

    # Read TerpLife Labs COA data.
    results = pd.read_excel('data/terplife-labs-coa-data-2023-11-21.xlsx')
    print('Read {} rows of data.'.format(len(results)))

    # Assign date columns.
    results['date'] = pd.to_datetime(results['date_tested'])
    results['month_year'] = results['date'].dt.to_period('M')
    print('Starting date: {}'.format(results['date'].min()))
    print('Ending date: {}'.format(results['date'].max()))


    # === Augment the data. ===

    # Identify all unique cultivation facilities.
    locations = list(results['producer_address'].unique())

    # Geocode licenses.
    googlemaps_api_key = dotenv_values(env_file)['GOOGLE_MAPS_API_KEY']
    gmaps = Client(key=googlemaps_api_key)
    address_coordinates = {}
    for address in locations:
        geocode_result = gmaps.geocode(address)
        location = geocode_result[0]['geometry']['location']
        address_coordinates[address] = (location['lat'], location['lng'])
        
    # Add latitude and longitude to the dataframe
    results['producer_latitude'] = results['producer_address'].apply(lambda x: address_coordinates[x][0] if x in address_coordinates else None)
    results['producer_longitude'] = results['producer_address'].apply(lambda x: address_coordinates[x][1] if x in address_coordinates else None)

    # API key for OpenWeatherMap
    env_file = '../../.env'
    open_weather_maps_api_key = dotenv_values(env_file)['OPEN_WEATHER_MAP_API_KEY']

    # FIXME: Find average temperature and humidity for the past 60 days
    # from the date_produced for each sample.
    # Get unique date produced by coordinate
    date_by_coordinate = results.groupby(['producer_latitude', 'producer_longitude'])['date_produced'].unique()
    for (lat, lon), unique_dates in date_by_coordinate.iteritems():
        for date_produced in unique_dates:
            # Calculate start and end date for the 60-day period
            end_date = pd.to_datetime(date_produced)
            start_date = end_date - timedelta(days=60)

            # Fetch weather data
            weather_data = fetch_weather(open_weather_maps_api_key, lat, lon, start_date, end_date)

            # Calculate average temperature and humidity
            average_temp = sum([data['temp'] for data in weather_data]) / len(weather_data)
            average_humidity = sum([data['humidity'] for data in weather_data]) / len(weather_data)

            print(f"Average temperature for {lat}, {lon} on {date_produced}: {average_temp} K")
            print(f"Average humidity for {lat}, {lon} on {date_produced}: {average_humidity} %")