import requests

url = 'https://sandbox-api-or.metrc.com/facilities/v1/'
params = None
auth = ('vendor_api_key', 'user-api-key')
response = requests.get(url, params=params, auth=auth)
data = response.json()


assert(response.status_code == 200)
