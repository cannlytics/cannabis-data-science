"""
Inventory Lab Results Analysis
Copyright (c) 2023 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 1/18/2023
Updated: 1/18/2023
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Data Sources:

    - Washington State Liquor and Cannabis Board (WSLCB)
    URL: <https://lcb.box.com/s/xseghpsq2t4i1musxj6mgd7b8rhxe7bm>

    - Curated CCRS Inventory Lab Results
    URL: <https://cannlytics.page.link/ccrs-inventory-lab-results-2022-12-07>

References:

    - Making timelines with Python
    URL: <https://dadoverflow.com/2021/08/17/making-timelines-with-python/>

"""
# Standard imports:
import ast
from datetime import timedelta
import gc
import os

# External imports:
from cannlytics.utils import camel_to_snake
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm


#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

# Specify where your data lives.
DATA_DIR = '../../.datasets/washington'

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 21,
})


#------------------------------------------------------------------------------
# Get the data.
#------------------------------------------------------------------------------

# Read in the data.
filename = 'ccrs-inventory-lab-results-2022-12-07.xlsx'
lab_results = pd.read_excel(os.path.join(DATA_DIR, filename))


#------------------------------------------------------------------------------
# Look at the data.
#------------------------------------------------------------------------------

# Find the timeline beginning and end.
start = lab_results['created_date'].min()
end = lab_results['updated_date'].max()

# Plot all lab results.
daily_lab_tests = lab_results.groupby('created_date')['inventory_id'].count()
daily_lab_tests.plot()

# Restrict to lab results created in 2022.
daily_lab_tests = lab_results.loc[
    (lab_results['created_date'] >= pd.to_datetime('2022-01-01')) &
    (lab_results['created_date'] <= pd.to_datetime('2022-12-31'))
].groupby('created_date')['inventory_id'].count()
daily_lab_tests.plot()


#------------------------------------------------------------------------------
# Analyze the data.
#------------------------------------------------------------------------------

# Find all Runtz varieties.
variety = 'runtz'
strain_tests = lab_results.loc[
    lab_results['strain_name'].str.contains(variety, case=False)
]
varieties = list(strain_tests['strain_name'].unique())

# Plot Runtz tests over 2022.
plt.figure(figsize=(8, 6))
daily_strain_tests = strain_tests.groupby('created_date')['inventory_id'].count()
daily_strain_tests.groupby(pd.Grouper(freq='M')).sum().plot()
plt.title(f'Number of "{variety}" tests in WA in 2022')
plt.show()


#------------------------------------------------------------------------------
# Model the data.
#------------------------------------------------------------------------------

def timeline(
        dates,
        labels,
        title,
        min_date=None,
        max_date=None,
        alternate=False,
        hide_yaxis=False,
        offset=None,
    ):
    """Create a vertical timeline."""

    # Set up the timeline and points.
    fig, ax = plt.subplots(
        figsize=(6, 10), 
        constrained_layout=True,
    )
    _ = ax.set_xlim(-5, 20)
    _ = ax.set_ylim(min_date, max_date)
    _ = ax.axvline(
        0,
        ymin=0.05,
        ymax=0.95,
        c='deeppink',
        zorder=1,
    )
    _ = ax.scatter(
        np.zeros(len(dates)),
        dates,
        s=120,
        c='palevioletred',
        zorder=2,
    )
    _ = ax.scatter(
        np.zeros(len(dates)),
        dates,
        s=30,
        c='darkmagenta',
        zorder=3,
    )

    # Add labels.
    label_offsets = np.repeat(2.0, len(dates))
    label_offsets[1::2] = -2.0
    for i, (l, d) in enumerate(zip(labels, dates)):
        if offset:
            d = d - timedelta(days=offset)
        align = 'left'
        if i % 2 == 0 and alternate:
            align = 'left'
        _ = ax.text(
            label_offsets[i] + 5,
            d,
            l,
            ha=align,
            fontfamily='serif',
            fontweight='bold',
            color='royalblue',
            fontsize=12
        )

    # Draw stems.
    stems = np.repeat(5.0, len(dates))
    if alternate:
        stems[1::2] *= -1.0
    x = ax.hlines(dates, 0, stems, color='darkmagenta')

    # Clean the chart.
    _ = ax.set_xticks([])
    if hide_yaxis:
        _ = ax.set_yticks([])
    for spine in ['left', 'top', 'right', 'bottom']:
        _ = ax.spines[spine].set_visible(False)

    # Format the plot.
    _ = ax.set_title(
        title,
        fontweight='bold',
        fontfamily='serif',
        fontsize=16, 
        color='royalblue'
    )
    plt.show()


#------------------------------------------------------------------------------
# Draw insights from the data.
#------------------------------------------------------------------------------

# Specify timeframe.
min_date = pd.to_datetime('2022-01-01')
max_date = pd.to_datetime('2022-12-31')

# Find when each variety first appeared.
variety = 'runtz'
strain_tests = lab_results.loc[
    lab_results['strain_name'].str.contains(variety, case=False)
]
genesis = strain_tests.groupby('strain_name')['created_date'].min()

# Create a timeline of a sample of the varieties.
sample = genesis.loc[(genesis >= min_date) & (genesis <= max_date)]
sample = sample.sample(15, random_state=420)
title = f'Timeline of "{variety}" tests in WA in 2022'
timeline(sample, sample.index, title, min_date, max_date)

# Identify the creator of a particular strain.
strain = 'Greasy Runtz'
inventor = lab_results.loc[lab_results['strain_name'] == strain]
inventor = inventor.loc[inventor['created_date'] == inventor['created_date'].min()]
inventor_name = inventor.iloc[0]["retailer_dba"]
print(f'Inventor of {strain}: {inventor_name}')
