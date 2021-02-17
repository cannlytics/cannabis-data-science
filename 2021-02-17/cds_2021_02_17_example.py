"""
Get API Data | Cannabis Data Science Meetup Group

Author: Keegan Skeate
Created: Tue Feb 16 18:23:03 2021

License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Description:
    Get as many cannabis lab results as possible from public APIs
    and save them to Excel.

Resources:
    https://github.com/cannlytics/cannlytics-api
    https://dev.socrata.com/foundry/data.oregon.gov/hybm-xi2e
"""
import requests
import pandas as pd

#----------------------------------------------------------------------------#
# Oregon Pesticides API
#----------------------------------------------------------------------------#

# Get pesticide list from Oregon.
pesticides_endpoint = "https://data.oregon.gov/resource/b8ki-p9ef.json"
response = requests.get(pesticides_endpoint)
pesticides = response.json()
print(pesticides[0])

# Turn pesticides data into a DataFrame
pesticides_data = pd.DataFrame(pesticides)

# Turn data to title case
# https://stackoverflow.com/questions/39512002/convert-whole-dataframe-from-lower-case-to-upper-case-with-pandas
pesticides_data = pesticides_data.apply(lambda x: x.astype(str).str.title())

# Save pesticide data to Excel
pesticides_data.to_excel("data/pesticides.xlsx")

#----------------------------------------------------------------------------#
# Cannlytics Labs API
#----------------------------------------------------------------------------#

# Get cannabis-testing labs from the Cannlytics API
labs_endpoint = "https://api.cannlytics.com/v1/labs"
response = requests.get(labs_endpoint)
labs = response.json()
print(labs["data"][0])

# Turn labs data into a DataFrame.
labs_data = pd.DataFrame(labs["data"])

# Save labs to Excel
labs_data.to_excel("./data/labs.xlsx")

