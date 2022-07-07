"""
Cannabis-Related Census Data
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 4/25/2022
Updated: 4/27/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description: We have found the holy grail folks 👑
This script curates variables related to cannabis in the
USA by state and nationally from a 2019-2020 Census survey.

Data Source:

    - SAMHSA, Center for Behavioral Health Statistics and Quality, National Survey on Drug Use and Health, 2019 and Q1 and Q4 2020.
    https://www.samhsa.gov/data/data-we-collect/nsduh-national-survey-drug-use-and-health

    - National Survey on Drug Use and Health (NSDUH) Population Data
    https://www.datafiles.samhsa.gov/dataset/national-survey-drug-use-and-health-2020-nsduh-2020-ds0001

Setup:

    1. pip install cannlytics

    2. Get a Fed FRED API key and save to `../.env`
    as `FRED_API_KEY`. URL: https://fred.stlouisfed.org/docs/api/api_key.html

    3. Download data to `../.datasets/census` or your directory of choice.

"""
# Standard imports.
import json
import os
import re

# External imports.
from cannlytics.utils.utils import snake_case
from dotenv import dotenv_values
import matplotlib.pyplot as plt
import pandas as pd
import pdfplumber
import seaborn as sns
from statadict import parse_stata_dict # pip install statadict

# Internal imports
from constants import states, state_names
from data import (
    format_millions,
    format_thousands,
    get_state_population,
    sentence_case,
)


# Specify where your data lives.
DATA_DIR = '../.datasets/census'

# Specify the year of the study.
YEAR = 2020

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 24})
colors = sns.color_palette('Set2', n_colors=10)
primary_colors = sns.color_palette('husl', n_colors=10)


#-----------------------------------------------------------------------
# Get the population for each state.
#-----------------------------------------------------------------------

# Read FRED API key.
# Get a key: https://fred.stlouisfed.org/docs/api/api_key.html
config = dotenv_values('../.env')
fred_api_key = config['FRED_API_KEY']

# Get the population for each state and the US, using abbreviations.
populations = {}
abbreviations = list(states.keys())
for state in abbreviations:
    population = get_state_population(
        api_key=fred_api_key,
        state=state,
        obs_start=f'{YEAR}-01-01',
        obs_end='2020-01-01',
    )
    populations[state] = population / 1000 # Turn to thousands.

# Get the US total population.
us_population = get_state_population(
    api_key=fred_api_key,
    state='',
    district='TOTUSA647NWDB',
    obs_start=f'{YEAR}-01-01',
    obs_end=f'{YEAR}-01-01',
    multiplier=1,
)
populations['US'] = us_population / 1_000


#-----------------------------------------------------------------------
# Read the data: 2019-2020 National Surveys on Drug Use and Health -
# Model-Based Estimated Totals (in Thousands)
# (50 States and the District of Columbia)
#-----------------------------------------------------------------------

# Define the totals table.
file_name = 'NSDUHsaeTotals2020.xlsx'

# Optional: Keep notes and source from rows that with NOTE, Source, '--'

# Read all of the tables.
state_data = pd.DataFrame()
for i in range(1, 34):

    # Get the data.
    table = f'Table {i}'
    data = pd.read_excel(
        os.path.join(DATA_DIR , file_name),
        sheet_name=table,
        header=None,
    )

    # Get the field.
    title = data.iloc[0][0]
    title = title.replace(table + '. ', '')
    title = title.split(':')[0]
    field = snake_case(title)
    field = field.replace('marijuana', 'cannabis')

    # Skip the notes and source to get to the data.
    columns = []
    for index, row in data.iterrows():
        cell = row[0]
        if cell == 'Order':
            columns = row.values
            columns = [x.replace('\n', ' ') for x in columns]
            data = data[index + 1:]
            break
    data.columns = columns
    data.index = data['State']

    # Read `18+ Estimate` values.
    try:
        data = data[['18+ Estimate']]
        data.columns = [field]
    except KeyError:
        # Note: Table 16 (alcohol use) is formatted differently.
        data = data[[
            'Alcohol Use in Past Month Estimate',
            'Binge Alcohol Use in Past Month Estimate',
        ]]
        data.columns = [snake_case(x).replace('_estimate', '') for x in data.columns]
    state_data = pd.concat([state_data, data], axis=1)

# Drop regions from states: Northeast, Midwest, South, West
drop = ['Northeast', 'Midwest', 'South', 'West']
state_data.drop(drop, inplace=True)

# Keep `Total U.S.` as `US`.
state_data.rename({'Total U.S.': 'US'}, inplace=True)


#-----------------------------------------------------------------------
# Analyze total state consumers.
#-----------------------------------------------------------------------

# Estimate the number of cannabis consumers in the US! and by state!
# Optional: Also look at month, `cannabis_use_in_the_past_month``.
us_consumers = state_data.loc['US']['cannabis_use_in_the_past_year'] * 1000
print(
    'Estimated number of US consumers in 2020:',
    format_millions(us_consumers)
)

# Look at the states with the most consumers.
key = 'cannabis_use_in_the_past_year'
state_data = state_data.sort_values(key, ascending=False)
series = (state_data[key] * 1000).apply(format_millions)
print(series)

# Look at first-time consumers.
us_first_consumers = state_data.loc['US']['first_use_of_cannabis_in_the_past_year'] * 1000
print(
    'Estimated number of people who first used cannabis in 2020:',
    format_millions(us_first_consumers)
)

# Look at first-time consumers by state.
key = 'first_use_of_cannabis_in_the_past_year'
state_data = state_data.sort_values(key, ascending=False)
series = (state_data[key] * 1000).apply(format_thousands)
print(series)

# Augment population.
state_data.index.name = 'state_name'
for index, state in state_data.iterrows():
    if index == 'US':
        state_data.loc[index, 'population'] = us_population
        state_data.loc[index, 'state'] = 'US'
        continue
    abbreviation = state_names[index]
    population = populations[abbreviation]
    state_data.loc[index, 'population'] = population * 1000
    state_data.loc[index, 'state'] = abbreviation

# Look at percent of first-time consumers by state.
series = (state_data['first_use_of_cannabis_in_the_past_year'] * 1000 / state_data['population'])
series = series.sort_values(ascending=False) * 100
fig, ax = plt.subplots(figsize=(12, 8))
series.plot(kind='bar')
plt.title('Percent of Population that were First Time Users in 2020 by State')
plt.ylabel('Percent')
plt.xticks(fontsize=16)
plt.show()

#-----------------------------------------------------------------------
# Read NSDUH percentages from their PDFs with PDF plumber!
# Source: https://www.samhsa.gov/data/report/2019-2020-nsduh-state-specific-tables
#-----------------------------------------------------------------------

# Someone call a plumber! The data dam just burst!
# The Cannabis Data Science team is all hands on deck serving
# you up the holy grail of cannabis data: cannabis use rates.
# Get them while they're hot!!!


def split_at_digit(string):
    """Split a string at the first digit."""
    return re.split(r'\d+', string)


# Future work: Download and extract the PDFs programatically.

# Define the folder.
directory = os.path.join(DATA_DIR, 'NSDUHsaeStateTables2020/NSDUHsaeStateTables2020/')

# Define nuisance columns.
drop = ['PAST YEAR SUBSTANCE USE DISORDER AND TREATMENT',
        'TOBACCO PRODUCTS', 'ALCOHOL', 'ILLICIT DRUGS',
        'PAST YEAR MENTAL HEALTH ISSUES']
extra = ['Marijuana Use', 'Beverage Once or Twice a Week']

# Read all of the PDFS.
percentages = pd.DataFrame()
for doc in os.listdir(directory):

    # Get the state name.
    state_name = doc.replace('NSDUHsae', '').replace(f'{YEAR}.pdf', '')
    state_name = snake_case(state_name).replace('_', ' ').title()

    # Read the licensees PDF.
    pdf = pdfplumber.open(directory + doc)

    # Get all of the table data.
    for page in pdf.pages[2:]:

        # Get the table data.
        table = page.extract_table()
        columns = table[0]
        data = [x.split(' \n') for x in table[1]]

        # Identify fields, droping extraneous labels and handling
        # fields that take up an extra row.
        fields = data[0]
        for i, field in enumerate(fields):
            key = split_at_digit(field.strip())[0]
            if key in extra:
                fields[i - 1] += '_' + key
                fields.pop(i)
                continue
            fields[i] = key
            if key in drop:
                fields.pop(i)

        # Clean the fields.
        fields = [snake_case(split_at_digit(x)[0]) for x in fields]

        # Keep the 18+ measure.
        measures = data[1:][-1]
        measures = [x.strip(' ') for x in measures]
        measures = [ele for ele in measures if ele.strip()]

        # Record the percentages (any way to do this faster?).
        for i, field in enumerate(fields):
            percentages.loc[state_name, field] = measures[i]

# Clean the percentages.
# TODO: Replace '--' with 0?
percentages = percentages.apply(pd.to_numeric, errors='ignore')

# Replace marijuana with cannabis
percentages.columns = [
    x.replace('marijuana', 'cannabis') for x in percentages.columns
]

# Remove regions.
regions = ['North', 'Northeast', 'East', 'South', 'West', 'Midwest']
percentages.drop(index=regions, inplace=True, errors='ignore')

# Augment population (just in case).
percentages.index.name = 'state_name'
percentages.rename(index={
    'District Of Col': 'District of Columbia',
    'National': 'US',
}, inplace=True)
for index, state in percentages.iterrows():
    if index == 'US':
        percentages.loc[index, 'population'] = us_population
        percentages.loc[index, 'state'] = 'US'
        continue
    abbreviation = state_names[index]
    population = populations[abbreviation]
    percentages.loc[index, 'population'] = population * 1000
    percentages.loc[index, 'state'] = abbreviation


#-----------------------------------------------------------------------
# Analyze state consumption rates.
#-----------------------------------------------------------------------

# See the states where people consume the most frequently.
key = 'past_year_cannabis_use'
fig, ax = plt.subplots(figsize=(12, 8))
values = percentages.sort_values(key, ascending=False)
values[key].plot(kind='bar')
plt.title('Cannabis consumers as a percent of state population in 2020')
plt.ylabel('Percent')
plt.xticks(fontsize=16)
plt.show()

# See the states where people consume the most frequently.
key = 'past_month_cannabis_use'
fig, ax = plt.subplots(figsize=(12, 8))
values = percentages.sort_values(key, ascending=False)
values[key].plot(kind='bar')
plt.title('Monthly cannabis consumers as a percent of state population')
plt.ylabel('Percent')
plt.xticks(fontsize=16)
plt.show()

# Perceptions about monthly cannabis use risk.
key = 'perceptions_of_great_risk_from_smoking_cannabis_once_a_month'
fig, ax = plt.subplots(figsize=(12, 8))
values = percentages.sort_values(key, ascending=False)
values[key].plot(kind='bar')
plt.title('Perception of great risk from monthly cannabis use by state in 2020')
plt.ylabel('Percent')
plt.xticks(fontsize=16)
plt.show()


#-----------------------------------------------------------------------
# Get individual cannabis use data.
# https://www.datafiles.samhsa.gov/dataset/national-survey-drug-use-and-health-2020-nsduh-2020-ds0001
#-----------------------------------------------------------------------

def parse_codings(data_dir, filename='codings'):
    """Parse codings from Stata .do file into a dictionary,
    then output as JSON :)"""

    # FIXME: Handle extraneous .do elements and get only
    # `#delimit ;` to `label values`

    # Parse codings
    title = filename.split('.')[0]
    with open(os.path.join(data_dir, f'{title}.txt'), 'r') as doc:
        labels = doc.read()

    # Split each label and record the codings.
    codings = {}
    field_codings = labels.split('\n\n')
    for coding in field_codings:
        if coding.startswith('label define'):
            field = coding.split('\n')[0].replace('label define ', '')
        parts = coding.split('\n')[1:]
        key = field.lower()
        codings[key] = {}
        for part in parts:
            code = int(part.split(' "')[0])
            description = part.replace(f'{code} "{code} - ', '').replace('"', '')
            codings[key][code] = description

    # Save as JSON
    with open(os.path.join(data_dir, f'{title}.json'), 'w+') as doc:
        json.dump(codings, doc, sort_keys=True, indent=4)

    # Return the codings.
    return codings

codings = parse_codings(DATA_DIR)

# Read all of the field details.
folder = 'NSDUH-2020-DS0001-bndl-data-ascii-setup-to-stata_v1'
filename = os.path.join(DATA_DIR, folder, 'NSDUH_2020_Setup_ASCII_to_STATA.dct')
stata_dict = parse_stata_dict(file=filename)
survey_fields = pd.DataFrame({
    'column': stata_dict.column_numbers,
    'description': stata_dict.comments,
    'key': stata_dict.names,
    'format': stata_dict.formats,
    'type': stata_dict.types,
})

# Format the fields
survey_fields['description'] = survey_fields['description'].apply(sentence_case)
survey_fields['key'] = survey_fields['key'].apply(str.lower)

# Optional: Only read in the variables of interest.
variables_of_interest = []

# Read TSV.
filenmae = 'NSDUH-2020-DS0001-bndl-data-tsv_v1/NSDUH_2020_Tab.txt'
panel = pd.read_csv(
    os.path.join(DATA_DIR, filenmae),
    low_memory=False,
    sep='\t'
)
panel.columns = [x.lower() for x in panel.columns]


#-----------------------------------------------------------------------
# Analyze consumer characteristics.
#-----------------------------------------------------------------------

# Look at the age where people first used cannabis.
invalid_ages = list(codings['mjage'].keys())
know_age_first_consumed = panel.loc[~panel['mjage'].isin(invalid_ages)]
know_age_first_consumed['mjage'].hist(bins=100)
plt.title('Age of First Cannabis Use of US Consumers')
plt.show()

# Find percent of population who consumed in last 30 days.
past_30_day_consumers = panel.loc[panel['mjrec'] == 1]
proportion = len(past_30_day_consumers) / len(panel)
print('Proportion who consumed in last 30 days: %.2f%%' % (proportion * 100))

past_year_consumers = panel.loc[( panel['mjrec'] == 1) |  (panel['mjrec'] == 2)]
proportion = len(past_year_consumers) / len(panel)
print('Proportion who consumed in 2020: %.2f%%' % (proportion * 100))

# Find the percent that have used medical cannabis in the past year.
past_year_patients = panel.loc[panel['medmjyr'] == 1]
proportion = len(past_year_patients) / len(panel)
print('Proportion with medical cannabis perscription in 2020: %.2f%%' % (proportion * 100))

#-----------------------------------------------------------------------

# Male vs female 30 day consumers.
female_consumers = past_30_day_consumers.loc[
    past_30_day_consumers['irsex'] == 2
]
proportion = len(female_consumers) / len(past_30_day_consumers)
print('Proportion of monthly female consumers: %.2f%%' % (proportion * 100))

# Male vs female past year.
female_consumers = past_year_consumers.loc[
    past_year_consumers['irsex'] == 2
]
proportion = len(female_consumers) / len(past_year_consumers)
print('Proportion of female consumers in 2020: %.2f%%' % (proportion * 100))


#-----------------------------------------------------------------------
# Analyze reported effects of consumption.
#-----------------------------------------------------------------------

def calculate_ratio(data, field):
    return len(data.loc[data[field] == 1]) /  len(data.loc[(data[field] == 1) | (data[field] == 2)])

def show_ratio(data, fields, field):
    ratio = calculate_ratio(data, field)
    description = fields.loc[fields['key'] == field].iloc[0]['description']
    print(f'{description}: ' +  '%.2f%%' % (ratio * 100))
    return ratio


# Those that "Needed more mj to get same effect pst 12 mos".
mrjndmor = show_ratio(panel, survey_fields, 'mrjndmor')

# Those that "Using same amt mj had less effect past 12 mos".
mrjlsefx = show_ratio(panel, survey_fields, 'mrjlsefx')

# Those that "Want/try to cut down/stop using mj pst 12 mos".
mrjcutdn = show_ratio(panel, survey_fields, 'mrjcutdn')

# Those "Able to cut/stop using mj every time pst 12 mos"
mrjcutev = show_ratio(panel, survey_fields, 'mrjcutev')

# Those where "Mj cause prbs with emot/nerves past 12 mos".
mrjemopb = show_ratio(panel, survey_fields, 'mrjemopb')

# Those with "Any phys prbs caused/worsnd by mj pst 12 mos"
mrjphlpb = show_ratio(panel, survey_fields, 'mrjphlpb')

# Contd to use cannabis despite phys prbs
mrjphctd = show_ratio(panel, survey_fields, 'mrjphctd')

# Less activities b/c of mj use past 12 mos
mrjlsact = show_ratio(panel, survey_fields, 'mrjlsact')

# Mj cause sers prbs at home/work/sch pst 12 mos
mrjserpb = show_ratio(panel, survey_fields, 'mrjserpb')

# Using mj and do dangerous activities pst 12 mos
mrjpdang = show_ratio(panel, survey_fields, 'mrjpdang')

# Using mj cause prbs with law past 12 mos
mrjlawtr = show_ratio(panel, survey_fields, 'mrjlawtr')

# Using mj cause prbs w/family/friends pst 12 mos
mrjfmfpb = show_ratio(panel, survey_fields, 'mrjfmfpb')

# Contd to use mj despite prbs w/ fam/frnds
mrjfmctd = show_ratio(panel, survey_fields, 'mrjfmctd')


#-----------------------------------------------------------------------
# Analyze consumer behavior.
#-----------------------------------------------------------------------

# See how many may have driven under the influence.
dui = past_year_consumers.loc[
    past_year_consumers['drvinmarj'] == 1
]
print('DUI:', len(dui) / len(past_year_consumers))


# At job or business last time bought cannabis
# mmbatjob1

# Buy last cannabis from store or dispensary
# mmbtdisp

# Give away any cannabis got last time for free
# mmfgive

# Give away any cannabis last time you grew it
# mmggive

#-----------------------------------------------------------------------
# Characteristics of consumption.
#-----------------------------------------------------------------------

# Use mj or hashish even once yr before last
mrjyrbfr = show_ratio(panel, survey_fields, 'mrjyrbfr')

# Age of last use.
invalid_ages = list(codings['mrjaglst'].keys())
consumers_with_ages = panel.loc[~panel['mrjaglst'].isin(invalid_ages)]
print('How old were you the last time used cannabis:')
consumers_with_ages['mrjaglst'].hist(bins=100)
plt.title('Ages of US Cannabis Consumers')
plt.show()

# Optional: Look at time of last use for people who used
# longer than a year ago.
old_time_users = panel.loc[
    (panel['mjever'] == 1) &
    (panel['drvinmarj'] == 2)
]
proportion = len(old_time_users) / len(panel)
print('Proportion who used to consume before 2019: %.2f%%' % (proportion * 100))

# Look how people last used cannabis.
# ("How did you get last cannabis used").
coding = codings['mmgetmj']
users = panel.loc[panel['mmgetmj'].isin([1, 2, 3, 4])]
users['source'] = users['mmgetmj'].map(coding)

# Visualize how people get their cannabis.
fig, ax = plt.subplots(figsize=(8,8), facecolor='silver')
users['source'].value_counts().plot(kind='pie')
plt.show()

# Calculate percent of past 30 day consumers who grew their own.
homegrowers = past_30_day_consumers.loc[past_30_day_consumers['mmgetmj'] == 4]
proportion = len(homegrowers) / len(past_30_day_consumers)
print('Proportion who consumed their own homegrow in the last 30 days: %.2f%%' % (proportion * 100))

homegrowers = past_year_consumers.loc[past_year_consumers['mmgetmj'] == 4]
proportion = len(homegrowers) / len(past_year_consumers)
print('Proportion who consumed their own homegrow in the last year: %.2f%%' % (proportion * 100))

# Estimate the number of homegrowers in the US!
proportion = len(homegrowers) / len(panel)
us_homegrowers = proportion * us_population
print('Number of US homegrowers:', format_millions(us_homegrowers))


#-----------------------------------------------------------------------
# Analyze policy effects on consumption.
#-----------------------------------------------------------------------

# How difficult is it for consumers to get cannabis?
coding = codings['difgetmrj']
panel['difficulty'] = panel['difgetmrj'].map(coding)
state_medical = panel.loc[panel['medmjpa2'] == 1]
state_not_medical = panel.loc[panel['medmjpa2'] == 2]

# Visualize how hard it is for people in medical states to get cannabis.
fig, ax = plt.subplots(figsize=(8,8), facecolor='silver')
users = state_medical.loc[state_medical['difgetmrj'].isin([1, 2, 3, 4, 5])]
users['difficulty'].value_counts().plot(kind='pie')
plt.title('Reported difficulty to get cannabis in medical states')
plt.ylabel('')
plt.show()

# Visualize how hard it is for people in non-medical states to get cannabis.
fig, ax = plt.subplots(figsize=(8,8), facecolor='silver')
users = state_not_medical.loc[state_not_medical['difgetmrj'].isin([1, 2, 3, 4, 5])]
users['difficulty'].value_counts().plot(kind='pie')
plt.title('Reported difficulty to get cannabis in non-medical states')
plt.ylabel('')
plt.show()


#-----------------------------------------------------------------------
# Analytics: Look at correlation with other interesting fields.
#-----------------------------------------------------------------------

# Look at service members who partake in cannabis consumption.
service = panel.loc[panel['service'] == 1]
service_consumers = service.loc[
    (service['mjrec'] == 1) |
    (service['medmjyr'] == 1)
]
proportion = len(service_consumers) / len(service)
print('Proportion of service members who partake: %.2f%%' % (proportion * 100))

#-----------------------------------------------------------------------

# Look at consumption rates of people who work vs. unemployed.
# wrkhadjob | wrkdpstwk | wrknjbwks
employed = panel.loc[panel['wrkdpstwk'] == 1]
employed_consumers = employed.loc[employed['mjrec'] == 1]
proportion = len(employed_consumers) / len(employed)
print('Proportion of employed who partake: %.2f%%' % (proportion * 100))

unemployed = panel.loc[panel['wrkdpstwk'] == 2]
unemployed_consumers = unemployed.loc[unemployed['mjrec'] == 1]
proportion = len(unemployed_consumers) / len(unemployed)
print('Proportion of unemployed who partake: %.2f%%' % (proportion * 100))

#-----------------------------------------------------------------------

# Estimate the number of days consumers consume, who consumed in last 30 days.
coding = codings['mr30est']
options = [1, 2, 3, 4, 5, 6]
users = past_30_day_consumers.loc[past_30_day_consumers['mr30est'].isin(options)]
users['frequency'] = users['mr30est'].map(coding)

# Visualize the frequency.
fig, ax = plt.subplots(figsize=(8,8), facecolor='silver')
users['frequency'].value_counts().plot(
    kind='pie',
    autopct=lambda pct: "{:.1f}%".format(pct)
)
plt.title('Reported frequency of cannabis use in past 30 days of US Consumers')
plt.ylabel('')
plt.show()


#-----------------------------------------------------------------------
# Analyze sales statistics.
#-----------------------------------------------------------------------

import numpy as np
from scipy.stats import norm

# Amount paid for last cannabis joints bought
key = 'mmjnpctb1'
coding = codings[key]
options = range(1, 5)
users = past_year_consumers.loc[past_year_consumers[key].isin(options)]
users['amount'] = users[key].map(coding)
fig, ax = plt.subplots(figsize=(8,8), facecolor='silver')
users['amount'].value_counts().plot(
    kind='pie',
    autopct=lambda pct: "{:.1f}%".format(pct)
)
plt.title('Amount paid for last cannabis joints bought by US Consumers')
plt.ylabel('')
plt.show()

# Amount paid for cannabis bought last time
key = 'mmlspctb1'
coding = codings[key]
options = range(1, 13)
users = past_year_consumers.loc[past_year_consumers[key].isin(options)]
users['amount'] = users[key].map(coding)
fig, ax = plt.subplots(figsize=(8, 8), facecolor='silver')
users['amount'].value_counts(normalize=True).mul(100).plot(kind='bar')
plt.title('Amount paid for cannabis bought last time by US Consumers')
plt.ylabel('Percent (%)')
plt.show()

# Amount worth last cannabis joints traded for.
key = 'mmtjwrcb1'
coding = codings[key]
options = range(1, 13)
users = past_year_consumers.loc[past_year_consumers[key].isin(options)]
users['amount'] = users[key].map(coding)
fig, ax = plt.subplots(figsize=(8, 8), facecolor='silver')
users['amount'].value_counts(normalize=True).mul(100).plot(kind='bar')
plt.title('Amount worth last cannabis joints traded by US Consumers')
plt.ylabel('Percent (%)')
plt.show()

# Amount the last cannabis traded for was worth
key = 'mmtlwrcb1'
coding = codings[key]
options = range(1, 13)
users = past_year_consumers.loc[past_year_consumers[key].isin(options)]
users['amount'] = users[key].map(coding)
fig, ax = plt.subplots(figsize=(8, 8), facecolor='silver')
users['amount'].value_counts(normalize=True).mul(100).plot(kind='bar')
plt.title('Amount the last cannabis traded for was worth by US Consumers')
plt.ylabel('Percent (%)')
plt.show()

# Optional: Price category of last cannabis joints bought
# mmjnpcat1

# Optional: Number of cannabis joints bought last time
# mmjntnum1

# Amount of cannabis bought last time (grams).
key = 'mmlsgms1'
coding = codings[key]
options = [1, 2, 3]
users = past_year_consumers.loc[past_year_consumers[key].isin(options)]
users['amount'] = users[key].map(coding)
fig, ax = plt.subplots(figsize=(8, 8))
users['amount'].value_counts(normalize=True).mul(100).plot(kind='bar')
plt.title('Amt of cannabis bought last time - grams by US Consumers')
plt.ylabel('Percent (%)')
plt.show()
print('N:', len(users))

# Estimate the demand for sales in the US.
prior_max = 28
spend_coding = {
    1: {'min': 1, 'max': 5},
    2: {'min': 5, 'max': 10},
    3: {'min': 10, 'max': prior_max},
}
users['amount_min'] = users[key].apply(lambda x: spend_coding[x]['min'])
users['amount_max'] = users[key].apply(lambda x: spend_coding[x]['max'])

# Future work: Estimate price per gram and compare with prior data.

# Future work: Number of days bought cannabis past 30 days.
# mmbt30dy

# Plot prior belief of spend per year.
# Future work: Add sales per capita calculations.
fig, ax = plt.subplots(figsize=(12, 8))
mean = 500 ; sd = 200
prior = np.arange(mean - 3 * sd, mean + 3 * sd)
density = pd.Series(norm.pdf(prior, mean, sd))
plt.plot(prior, density)
plt.title('Prior Belief of Annual Consumer Cannabis Spend')
plt.xlabel('Annual Spend (Dollars)')
plt.xlim(0)
plt.show()

# Plot prior belief of demand in the US.
annual_spend = pd.Series(prior)
proportion = len(past_year_consumers) / len(panel)
estimated_us_demand = annual_spend * proportion * us_population
estiamted_us_demand_billions = estimated_us_demand.mul(10**(-9))
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(estiamted_us_demand_billions, density.mul(100))
plt.xlabel('Demand (Billions of Dollars)')
plt.ylabel('Probability')
plt.title('Predicted Demand for Cannabis in the US')
plt.xlim(0)
plt.show()

# Saturday Morning Statistics teaser:
# Create probability of an interval.
minimum = 20 ; maximum = 50
interval = estiamted_us_demand_billions.loc[
    (estiamted_us_demand_billions > minimum) &
    (estiamted_us_demand_billions < maximum)
]

# Visualize probability interval with prediction.
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(estiamted_us_demand_billions, density.mul(100))
plt.fill_between(
    interval,
    density.loc[interval.index].mul(100),
    color='#0b559f',
    alpha=0.2
)
plt.xlabel('Demand (Billions of Dollars)')
plt.ylabel('Probability')
plt.title('Predicted Demand for Cannabis in the US in 2020')
plt.xlim(0)
plt.show()

# Probability of falling in the interval
probability = density.loc[interval.index].mul(100).sum().round(2)
print(f'Probability of demand being between ${minimum}B and ${maximum}B, given prior: {probability}%')

# Estimate satiated demand.
# Future work: Add our prior estimate of US sales in 2020!
us_legal_sales = 25 
satiated = estiamted_us_demand_billions.loc[
    (estiamted_us_demand_billions > 0) &
    (estiamted_us_demand_billions < us_legal_sales)
]
proportion = density.loc[satiated.index].mul(100).sum().round(2)
amount = round(satiated.max())
print(f'Satiated demand, given priors: {proportion}% (${amount}B)')


# Future work: Look at sales per capita from prior weeks and update prior!


#-----------------------------------------------------------------------
# Upcoming Analytics in Saturday Morning Statistics!
#-----------------------------------------------------------------------

# Correlate the various variables and see which variables move together.

# Get economic variables (e.g. from Fed FRED) and
# correlate state economic variables, such as cannabis sales,
# with participation rates.

# Heckman model to predict sales, consistently!


#-----------------------------------------------------------------------
# Create a beautiful data visualization!
#-----------------------------------------------------------------------

# Create a choropleth of cannabis use rates by state in the USA!!!

#-----------------------------------------------------------------------
# Save the data!
#-----------------------------------------------------------------------

# Save state-level totals.
state_data.to_excel(f'{DATA_DIR}/stats/nsduh-state-level-adult-totals.xlsx')

# Save percentages.
percentages.to_excel(f'{DATA_DIR}/stats/nsduh-state-level-adult-percentages.xlsx')

# Save survey fields.
survey_fields.to_excel(f'{DATA_DIR}/stats/nsduh-survey-fields.xlsx')
