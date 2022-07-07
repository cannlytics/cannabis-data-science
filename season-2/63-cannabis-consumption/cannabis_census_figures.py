"""
Cannabis-Related Census Data
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 4/25/2022
Updated: 4/28/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description: This script creates figures of cannabis use in the
USA by state from a 2019-2020 Census survey.

Data Source:

    - SAMHSA, Center for Behavioral Health Statistics and Quality, National Survey on Drug Use and Health, 2019 and Q1 and Q4 2020.
    https://www.samhsa.gov/data/data-we-collect/nsduh-national-survey-drug-use-and-health

    - National Survey on Drug Use and Health (NSDUH) Population Data
    https://www.datafiles.samhsa.gov/dataset/national-survey-drug-use-and-health-2020-nsduh-2020-ds0001

    - National Survey on Drug Use and Health (NSDUH) State-Level Adult Totals
    https://cannlytics.page.link/nsduh-state-level-adult-totals

Setup:

    1. Download data from https://cannlytics.page.link/nsduh-state-level-adult-totals
    to `../.datasets/census` or your directory of choice.
    You will also need a `../.datasets/census/figures` folder.

"""
import pandas as pd
import matplotlib.pyplot as plt
import requests
import seaborn as sns

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 24})
colors = sns.color_palette('Set2', n_colors=10)
primary_colors = sns.color_palette('husl', n_colors=10)


def change_width(ax, new_value):
    """Modify patches width and x coordinates for positioning.
    Author: jsgounot <https://stackoverflow.com/users/5016055/jsgounot>
    License: CC BY-SA 3.0 <https://creativecommons.org/licenses/by-sa/3.0/>
    """
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value
        patch.set_width(new_value)
        patch.set_x(patch.get_x() + diff * .5)

# Read in the data.
datafile = '../.datasets/census/stats/nsduh-state-level-adult-totals.xlsx'
state_data = pd.read_excel(datafile, sheet_name='NSDUH State Totals')

# Get state regulatory statuses from the Cannlytics API!
url = 'https://cannlytics.com/api/data/state'
response = requests.get(url)
regulations = pd.DataFrame(response.json()['data'])

# Plot percent of first time users by state with medical / adult-use status.
adult_use = regulations.loc[regulations['recreational'] == True]
medical_only = regulations.loc[
    (regulations['recreational'] != True) &
    (regulations['medicinal'] == True)
]
prohibited = regulations.loc[
    (~regulations['state'].isin(adult_use['state'].unique())) &
    (~regulations['state'].isin(medical_only['state'].unique()))
]

# Look at percent of first-time consumers by state.
state_data.index = state_data['state_name']
series = state_data['first_use_of_cannabis_in_the_past_year'] / state_data['population']
state_data['percent_first_time_users'] = series * 100
state_data = state_data.sort_values(by='percent_first_time_users', ascending=False)

# Code `regulation` variable.
state_data.loc[state_data.state.isin(prohibited.state.unique()), 'regulation'] = 'Prohibited'
state_data.loc[state_data.state.isin(medical_only.state.unique()), 'regulation'] = 'Medical Only'
state_data.loc[state_data.state.isin(adult_use.state.unique()), 'regulation'] = 'Adult Use'
state_data.drop('US', inplace=True)
state_data['state_name'].replace({'District of Columbia': 'D.C.'}, inplace=True)

# Create the figure
fig, ax = plt.subplots(figsize=(18, 11.5))
sns.barplot(
    x='state_name',
    y='percent_first_time_users',
    hue='regulation',
    palette=sns.color_palette('gist_earth_r', n_colors=3),
    data=state_data,
    dodge=False,
)
plt.title('First Time Cannabis Users as a Percent of Population by State in 2020')
plt.ylabel('Percent')
plt.xlabel('State')
plt.xticks(fontsize=16)
plt.xticks(rotation=90)
plt.legend(title='', loc='upper right')
plt.text(
    0,
    -0.45,
    """Authors: Cannlytics and the Cannabis Data Science Team
Data Source: SAMHSA, Center for Behavioral Health Statistics and Quality,
National Survey on Drug Use and Health, 2019 and Q1 and Q4 2020.""",
    transform=ax.transAxes,
    fontsize=16,
)
change_width(ax, 0.5)
fig.tight_layout(pad=0.5)
plt.savefig(
    '../.datasets/census/figures/nsduh-first-time-users-by-state.png',
    dpi=96,
    bbox_inches='tight',
)
plt.show()
