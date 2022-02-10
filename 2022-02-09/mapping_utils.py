"""
Mapping Functions
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 2/9/2022
Updated: 2/9/2022
License: MIT License <https://opensource.org/licenses/MIT>
"""
# Standard imports.
from datetime import datetime
from typing import Any, Optional, Tuple

# External imports.
from dotenv import dotenv_values
import googlemaps


def initialize_googlemaps(env_file: Optional[str] = './.env') -> Any:
    """Initialize the Google Maps client.
    Args:
        env_file (str): A file path to a .env file with a `GOOGLE_MAPS_API_KEY`
            environment variable.
    Returns:
        (Client): A googlemaps API client.
    """
    config = dotenv_values(env_file)
    google_maps_api_key = config['GOOGLE_MAPS_API_KEY']
    client = googlemaps.Client(key=google_maps_api_key)
    return client


def get_transfer_distance(
        client,
        start,
        end,
        mode='driving',
) -> Tuple[int, int]:
    """Get the distance and duration of a transfer.
    Args:
        client (Client): A googlemaps API client.
        start (string): The starting point, either lat,long as a string or an address.
        end (string): The ending point, either lat,long as a string or an address.
        mode (string): The transporation method, driving by default.
    Returns:
        (int, int): Returns a tuple of the distance in kilometers and the
            duration in seconds.
    """
    driving_distances = client.distance_matrix(start, end, mode=mode)
    elements = driving_distances['rows'][0]['elements'][0]
    km = elements['distance']['value']
    duration = elements['duration']['value']
    return km, duration


def get_transfer_route(
        client,
        start,
        end,
        departure_time=None,
        mode='driving',
) -> str:
    """Get the route of a transfer.
    Args:
        client (Client): A googlemaps API client.
        start (string): The starting point, either lat,long as a string or an address.
        end (string): The ending point, either lat,long as a string or an address.
        departure_time (datetime): The time of departure, defaults to now (optional).
        mode (string): The transporation method, driving by default (optional).
    Returns:
        (str): Returns the route as a polyline string.
    """
    if departure_time is None:
        departure_time = datetime.now()
    driving_directions = client.directions(
        start,
        end,
        mode=mode,
        departure_time=departure_time
    )
    m = driving_directions[0]['legs'][0]['distance']['value']
    min = driving_directions[0]['legs'][0]['duration']['value']
    polyline = driving_directions[0]['overview_polyline']['points']
    return m, min, polyline
