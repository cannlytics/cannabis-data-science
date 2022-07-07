"""
Title | Project

Author: Keegan Skeate
Created: Wed Mar  3 09:06:37 2021

License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Description:
    

Resources:

"""

import requests

base = "https://api.cannlytics.com"
# base = "http://127.0.0.1:8000"

endpoint = "/test/leaf/mmes"

response = requests.get(base + endpoint + '?limit=2')
print(response.status_code)
data = response.json()


response = requests.get(base + "/test/leaf/lab_results" + '?limit=2')
print(response.status_code)
lab_results = response.json()

