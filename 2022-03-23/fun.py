"""
Epidemeology Exercise
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 3/21/2022
Updated: 3/22/2022
License: MIT License <https://opensource.org/licenses/MIT>
"""
import matplotlib.pyplot as plt
import pandas as pd

# Read the Table of Casualties
# Source:  
graunt = pd.read_excel('../.datasets/graunt.xlsx', index_col=0)

# Identify the years.
years = list(graunt.columns)

# Identify casualty of interest.
count = graunt.loc["King's Evil"].values

# Plot the casualties over time.
plt.plot(years[:14], count[:14], label="King's Evil")

# Plot a set of diseases.
plt.figure(figsize=(16, 12))
malaties = [
    # "King's Evil",
    # 'Cut of the Stone',
    # 'Poysoned',
    'Rickets',
    'Spleen',
    'Livergrown, Spleen, and Rickets',
]
for ailment in malaties:
    count = graunt.loc[ailment].values[:14]
    relative_difference = (1 + (count - count.mean()) / count.mean()) * 100
    plt.plot(years[:14], relative_difference, label=ailment)
plt.legend()
plt.show()
