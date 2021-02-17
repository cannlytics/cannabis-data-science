"""
Leaf Data Systems API Intro | Cannabis Data Science

Author: Keegan Skeate
Created: Wed Feb 17 08:55:55 2021

License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Description:
    Akerna (MJ Freeway) (Leaf Data Systems) -> Traceability in Washington State

Resources:
    https://lcb.wa.gov/sites/default/files/publications/Marijuana/traceability/WALeafDataSystems_UserManual_v1.37.5_AddendumC_LicenseeUser.pdf

"""

import requests

# Define base API endpoint.
base = "https://watest.leafdatasystems.com/api/v1"

# Define headers with authentication
headers = {
    "x-mjf-key": "key123", # Get your API key.
    "x-mjf-mme-code": "WAWA1.MM1VC", # Get company's MME global ID.
     "Content-Type": "application/json"    
}

# Get MME IDs.
response = requests.get(base + "/users", headers=headers)

# Get users.
response = requests.get(base + "/users", headers=headers)

# Create an area.
data = {
    "area": [
        {
            "name": "Flower Room 1",
            "type": "quarantine", # quarantine, non-quarantine
            "external_id": "flower_room_1",
        }    
    ]
}
response = requests.post(base + "/areas", data=data, headers=headers)

