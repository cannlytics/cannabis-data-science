"""
Title | Project

Author: Keegan Skeate
Created: Wed Feb 24 08:49:07 2021

License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Description:
    

Resources:

"""

# External imports
import pandas as pd
from bokeh.sampledata.us_counties import data as counties

# Internal imports
from logistics import geocode_addresses
from bokeh_chloropleth import bokeh_chloropleth


# Washington counties
state = "wa"
counties = {
    code: county for code, county in counties.items() if county["state"] == state
}
county_names = [county['name'] for county in counties.values()]


# Read in licensees.
licenses = pd.read_excel("./data/active_licenses.xlsx")

# Get counties.
licenses = licenses
licenses["street"] = licenses["address1"]
licenses["state"] = licenses["state_code"]
licenses["zip"] = licenses["postal_code"]
# sample = geocode_addresses(licenses)

# Create Bokeh chloropleth
from random import randint
values = [randint(0, 100) for p in range(0, len(counties))]
bokeh_chloropleth(state, values)


