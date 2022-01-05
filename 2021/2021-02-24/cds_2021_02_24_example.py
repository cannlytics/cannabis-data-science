"""
Get API Data | Cannabis Data Science Meetup Group

Author: Keegan Skeate
Created: 2/21/2021

License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Description:
    Mock Leaf API endpoints.

Resources:
    https://data-wadnr.opendata.arcgis.com/datasets/12712f465fc44fb58328c6e0255ca27e_11
    
    https://github.com/cannlytics/cannlytics-api
"""

# External imports
import pandas as pd
from bokeh.sampledata.us_counties import data as counties

# Internal imports
from logistics import geocode_addresses
from bokeh_chloropleth import bokeh_chloropleth

#----------------------------------------------------------------------------#
# Data Science
#----------------------------------------------------------------------------#

# Read in licensees.
licenses = pd.read_excel("./data/active_licenses.xlsx")

# Get counties.
licenses = licenses
licenses["street"] = licenses["address1"]
licenses["state"] = licenses["state_code"]
licenses["zip"] = licenses["postal_code"]
sample = geocode_addresses(licenses)

# Save to Excel
licenses["county"] = licenses["county"].str.replace(" County", "")
licenses.to_excel("./data/licenses.xlsx")

# Count licensees by county.
licenses = pd.read_excel("./data/licenses.xlsx")

# Map (chloropleth / heatmap) of cannabis licensees.
state = "wa"
counties = {
    code: county for code, county in counties.items() if county["state"] == state
}
county_names = [county['name'] for county in counties.values()]
from random import randint
values = [randint(0, 100) for p in range(0, len(counties))]
bokeh_chloropleth(state, values)


# Read in inventory adjustments.


# Determine how much waste is produced.


# Aggregate waste by licensee and by day, week, month


# Determine location of licensee.


# Map (chloropleth / heatmap) of cannabis waste.



#----------------------------------------------------------------------------#
# Mock Leaf API
#----------------------------------------------------------------------------#

# import requests

# url = 'http://www.ichangtou.com/#company:data_000008.html'
# headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

# response = requests.get(url, headers=headers)
# print(response.content)

