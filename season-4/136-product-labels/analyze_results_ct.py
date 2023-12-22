"""
Analyze Connecticut Cannabis Lab Results
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 12/14/2023
Updated: 12/14/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# Standard imports:
import matplotlib.pyplot as plt
import seaborn as sns

# External imports:
import pandas as pd
import statsmodels.api as sm


# === Setup ===

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})


# === Read the data ===

# Read the lab results.
datafile = './data/ct-coa-data-2023-12-14-13-53-12.xlsx'
results = pd.read_excel(datafile)
results['date'] = pd.to_datetime(results['date_tested'])
results['year_month'] = results['date'].dt.to_period('M')


# === Chemical Analysis ===


# === Lab Analysis ===

# Visualize the number of tests by lab by month.
monthly_counts = results.groupby(['lab', 'year_month']).size().reset_index(name='counts')
pivot_table = monthly_counts.pivot(index='year_month', columns='lab', values='counts')
pivot_table.plot(kind='line', figsize=(10, 6))
plt.title('Number of Results Tested by Each Lab by Month')
plt.xlabel('Month')
plt.ylabel('Number of Tests')
plt.show()


# === Market Share Analysis ===

# Visualize market share by lab over time.
total_per_month = monthly_counts.groupby('year_month')['counts'].sum()
monthly_counts = monthly_counts.set_index('year_month')
monthly_counts['total_per_month'] = total_per_month
monthly_counts['market_share'] = (monthly_counts['counts'] / monthly_counts['total_per_month']) * 100
pivot_table = monthly_counts.pivot(columns='lab', values='market_share')
pivot_table.plot(kind='line', figsize=(10, 6))
plt.title('Market Share of Lab Results by Lab Over Time')
plt.xlabel('Month')
plt.ylabel('Market Share (%)')
plt.show()
