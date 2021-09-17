"""
Licensees per Capita in Oklahoma | Cannabis Data Science Meetup

Author: Keegan Skeate <keegan@cannlytics.com>
Created: 9/1/2021
Updated: 9/1/2021
License: MIT License
"""
# External imports
import pandas as pd
from dotenv import dotenv_values
from fredapi import Fred

# Internal imports
from metadata_ok import counties

#--------------------------------------------------------------------------
# Get the data.
#--------------------------------------------------------------------------

# Get licensees in Oklahoma (See Cannlytics data collection tools).
# https://github.com/cannlytics/cannlytics/blob/main/tools/data/state_data/get_data_OK.py
licensees = pd.read_excel('data/licensees_OK_2021-09-01.xlsx')

# Get the population of Oklahoma.

# Read in a Fred API key.
config = dotenv_values('../.env')

# Get the effective federal funds rate from Fred.
fred = Fred(api_key=config['FRED_API_KEY'])
population = fred.get_series('OKPOP', observation_start='1/1/2020')

# Get the population of each county in Oklahoma.

#--------------------------------------------------------------------------
# Calculate statistics.
#--------------------------------------------------------------------------

# Calculate licensees per capita in Oklahoma.
licensees_per_capita = len(licensees) / (population.iloc[0] * 1000)
print('Licensees per capita:', licensees_per_capita)

# Calculate licensees per capita for each county in Oklahoma.
licensees_per_capita = {}
for county in counties:

    # Find all licensees in that county.
    county_name = county['name']
    county_licensees = licensees.loc[licensees.county == county_name]

    # Get the population of that county.
    county_population = fred.get_series(county['pop_ref'], observation_start='1/1/2020')

    # Calculate the licensees per capita in that county.
    county_licensees_per_capita = len(county_licensees) / (county_population.iloc[0] * 1000)

    # Keep track of the data.
    # key = 'OK_' + county_name.replace(' ', '_')
    # licensees_per_capita[county_name] = {'licensees_per_capita': county_licensees_per_capita}
    licensees_per_capita[county_name] = county_licensees_per_capita

    # Print out for sanity sake.
    # print('County:', county_name)
    # print('Population:', population)
    # print('Licensees:', len(county_licensees))
    # print('Licensees per capita:', county_licensees_per_capita)
    # print('---------------------------------')

#--------------------------------------------------------------------------
# Look at the data (chloropleth of Oklahoma.)
#--------------------------------------------------------------------------
# import matplotlib.pyplot as plt
# labels = []
# sizes = []
# for license_type in list(licensees.license_type.unique()):
#     license_type_data = licensees.loc[licensees.license_type == license_type]
#     porportion = round(len(license_type_data) / len(licensees) * 100, 2)
#     print(f'Total number of {license_type}:', len(license_type_data))
#     print('Percent of total:', porportion, '\n')
#     sizes.append(porportion)
#     labels.append(license_type)
# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, labels=labels, autopct='%1.1f%%',)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.show()

# External imports
import pandas as pd
from bokeh.io import show
from bokeh.models import LogColorMapper
from bokeh.palettes import Viridis6 as palette
from bokeh.plotting import figure
from bokeh.sampledata.us_counties import data as counties

# Washington counties
state = "ok"
counties = {
    code: county for code, county in counties.items() if county["state"] == state
}
county_names = [county['name'] for county in counties.values()]


# # Get counties.
# licenses = licenses
# licenses["street"] = licenses["address1"]
# licenses["state"] = licenses["state_code"]
# licenses["zip"] = licenses["postal_code"]
# # sample = geocode_addresses(licenses)

# Create Bokeh chloropleth
# from random import randint
# values = [randint(0, 100) for p in range(0, len(counties))]
# bokeh_chloropleth(state, values)

# Define county shapes.
counties = {
    code: county for code, county in counties.items() if county["state"] == state
}
county_xs = [county["lons"] for county in counties.values()]
county_ys = [county["lats"] for county in counties.values()]
county_names = [county['name'] for county in counties.values()]

# Define county values
# county_rates = [unemployment[county_id] for county_id in counties]

# Partition our values.
values = []
for county_name in county_names:
    value = licensees_per_capita.get(county_name, 0)
    values.append(value)

color_mapper = LogColorMapper(palette=palette)

data=dict(
    x=county_xs,
    y=county_ys,
    name=county_names,
    value=values,
)

TOOLS = "pan,wheel_zoom,reset,hover,save"

p = figure(
    title="Oklahoma Cannabis Licensees per Capita",
    tools=TOOLS,
    x_axis_location=None,
    y_axis_location=None,
    # tooltips=[
    #     ("Name", "@name"),
    #     ("Number", "@value%"),
    #     ("(Long, Lat)", "($x, $y)")
    # ]
    #     plot_width=800,
    # plot_height=500
    )
p.grid.grid_line_color = None
p.hover.point_policy = "follow_mouse"

p.patches(
    'x',
    'y',
    source=data,
    fill_color={'field': 'value', 'transform': color_mapper},
    fill_alpha=0.7,
    line_color="white",
    line_width=0.5,
)
show(p)
