"""
Title | Project

Author: Keegan Skeate
Contact: <keegan@cannlytics.com>
Created: 
Updated: 
License: MIT License <https://opensource.org/licenses/MIT>
"""
# Standard imports.
from pathlib import Path

# TODO: Otherwise download the data.

# Create a `data` directory if one doesn't already exist.
# Path('./data').mkdir(parents=True, exist_ok=True)

# # Download the Washington State sales by month by retailer data.
# filename = './data/washington_monthly_sales.xlsx'
# url = 'https://lcb.wa.gov/sites/default/files/publications/Marijuana/sales_activity/2021-12-06-MJ-Sales-Activity-by-License-Number-Traceability-Contingency-Reporting-Retail.xlsx'
# response = requests.get(url)
# with open(filename, 'wb') as f:
#     f.write(response.content)