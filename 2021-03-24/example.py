"""
Herfindahl–Hirschman Index | Cannabis Data Science

Author: Keegan Skeate
Created: Wed Mar 24 2021

License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Description:
    Estimate the HHI for cannabis cultivators in the Washington state
    cannabis market over time in 2020.

Resources:
    https://www.justice.gov/atr/herfindahl-hirschman-index
    https://data.openthc.org/b2b
"""
import pandas as pd

#-----------------------------------------------------------------------------
# Import the data.
#-----------------------------------------------------------------------------

data = pd.read_excel(
    './data/wholesale_sales.xlsx',
     sheet_name='Panel',
     col=0
)

#-----------------------------------------------------------------------------
# Calculate total sales by month.
#-----------------------------------------------------------------------------

sales_by_month = {}
months = data.month.unique()
for month in months:
    total_sales = data.loc[data.month == month].sales.sum()
    sales_by_month[month] = total_sales

#-----------------------------------------------------------------------------
# Calculate market share for each wholesaler by month. (Refactor!)
#-----------------------------------------------------------------------------

market_shares = []
for index, values in data.iterrows():
    market_share = values.sales / sales_by_month[values.month]
    market_shares.append(market_share)
data['market_share'] = pd.Series(market_shares)
    

#-----------------------------------------------------------------------------
# Calculate the HHI by month.
#-----------------------------------------------------------------------------

hhi_by_month = {}
for month in months:
    month_data = data.loc[data.month == month]
    hhi = 0
    for index, values in month_data.iterrows():
        hhi += values.market_share ** 2
    hhi_by_month[month] = hhi

#-----------------------------------------------------------------------------
# Plot the HHI
#-----------------------------------------------------------------------------

hhi_data = pd.Series(hhi_by_month)
hhi_data = hhi_data.sort_index()
hhi_data.plot()
