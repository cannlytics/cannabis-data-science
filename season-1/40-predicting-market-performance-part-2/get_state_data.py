"""
Get State-specific Data
Copyright (c) 2021 Cannlytics

Author: Keegan Skeate <keegan@cannlytics.com>
Created: 11/22/2021
Updated: 11/22/2021
License: MIT License <https://github.com/cannlytics/cannlytics-ai/blob/main/LICENSE>
"""

# External imports.
from fredapi import Fred


def get_state_current_population(state, api_key=None):
    """Get a given state's latest population from the Fed Fred API,
    getting the number in 1000's and returning the absolute value.
    Args:
        (str): The state abbreviation for the state to retrieve population
            data. The abbreviation can be upper or lower case.
        (str): A Fed FRED API key. You can sign up for a free API key at
            http://research.stlouisfed.org/fred2/. You can also pass `None`
            and set the environment variable 'FRED_API_KEY' to the value of
            your API key.
    Returns:
        (dict): Returns a dictionary with population values and source.
    """
    fred = Fred(api_key=api_key)
    state_code = state.upper()
    population_source_code = f'{state_code}POP'
    population = fred.get_series(population_source_code)
    real_population = int(population.iloc[-1] * 1000)
    population_date = population.index[-1].isoformat()[:10]
    return {
        'population': real_population,
        'population_formatted': f'{real_population:,}',
        'population_source_code': population_source_code,
        'population_source': f'https://fred.stlouisfed.org/series/{population_source_code}',
        'population_at': population_date,
    }
