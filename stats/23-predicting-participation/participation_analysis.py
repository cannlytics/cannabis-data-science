"""
Cannabis Participation Analysis
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 4/29/2022
Updated: 4/30/2022
License: MIT License <https://opensource.org/licenses/MIT>

Data Sources:

    - National Survey on Drug Use and Health (NSDUH) State-Level Adult Totals
    https://cannlytics.page.link/nsduh-state-level-adult-totals

    - National Survey on Drug Use and Health (NSDUH) Survey Fields
    https://cannlytics.page.link/nsduh-survey-fields

    - National Survey on Drug Use and Health (NSDUH) Survey Codings
    https://cannlytics.page.link/nsduh-codings

    - National Survey on Drug Use and Health (NSDUH) Panel Data
    https://www.datafiles.samhsa.gov/dataset/national-survey-drug-use-and-health-2020-nsduh-2020-ds0001

Setup:

    1. Get a Fed FRED API key and save to `../.env`
    as `FRED_API_KEY`. URL: https://fred.stlouisfed.org/docs/api/api_key.html

    2. Download data to `../.datasets/census/nsduh`
    or your directory of choice.

"""
# Standard imports.
import json
import os

# External imports.
from dotenv import dotenv_values
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels as sm

# Internal imports.
from data import get_state_population
from heckman import Heckman


# Specify where your data lives.
DATA_DIR = '../.datasets/census'

# Specify the year of the study.
YEAR = 2020

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})
colors = sns.color_palette('Set2', n_colors=10)
primary_colors = sns.color_palette('husl', n_colors=10)


#-----------------------------------------------------------------------
# Get individual cannabis use data.
#-----------------------------------------------------------------------

# Read field codings.
codings_file = os.path.join(DATA_DIR, 'nsduh', 'nsduh-codings.json')
with open(codings_file, 'r') as doc:
    codings = json.loads(doc.read())

# Read all field details.
survery_fields_file = os.path.join(DATA_DIR, 'nsduh', 'nsduh-survey-fields.xlsx')
survey_fields = pd.read_excel(survery_fields_file, sheet_name='Data')

# Optional: Only read in the variables of interest.
variables_of_interest = []

# Read TSV.
panel_file = os.path.join(DATA_DIR, 'nsduh', 'NSDUH_2020_Tab.txt')
panel = pd.read_csv(panel_file, sep='\t', low_memory=False)
panel.columns = [x.lower() for x in panel.columns]

# Get the US population.
config = dotenv_values('../.env')
fred_api_key = config['FRED_API_KEY']
us_population = get_state_population(
    api_key=fred_api_key,
    state='',
    district='TOTUSA647NWDB',
    obs_start=f'{YEAR}-01-01',
    obs_end=f'{YEAR}-01-01',
    multiplier=1,
)


#-----------------------------------------------------------------------
# Determine ages from use / age at past use.
# Problem: This introduces bias because it appears people who
# report using other substances are also more likely to report
# using cannabis.
#-----------------------------------------------------------------------

age_fields = [
    {'key': 'mrjaglst', 'indicator': 'mjrec'},
    {'key': 'cigaglst', 'indicator': 'cigrec'},
    {'key': 'smkaglast', 'indicator': 'smklssrec'},
    {'key': 'cgraglst', 'indicator': 'cigarrec'},
    {'key': 'alcaglst', 'indicator': 'alcrec'},
    {'key': 'cigaglst', 'indicator': 'cigrec'},
    {'key': 'cocaglst', 'indicator': 'cocrec'},
    {'key': 'crkaglst', 'indicator': 'crakrec'},
    {'key': 'heraglst', 'indicator': 'herrec'},
    {'key': 'hallaglst', 'indicator': 'hallucrec'},
    {'key': 'lsdaglst', 'indicator': 'lsdrec'},
    {'key': 'pcpaglst', 'indicator': 'pcprec'},
    {'key': 'ecstmoagl', 'indicator': 'ecstmorec'},
    {'key': 'inhlaglst', 'indicator': 'inhalrec'},
    {'key': 'methaglst', 'indicator': 'methamrec'},    
]


def determine_age_from_field(data, codings, key, indicator):
    """Get the age of people who used cannabis in the past month or year.
    Remove invalid ages and returning a series of ages with only valid ages.
    """
    field = data.loc[(data[indicator] == 1) | (data[indicator] == 2)]
    invalid_ages = list(codings[key].keys())
    invalid_ages = [int(x) for x in invalid_ages]
    field = field.loc[~field[key].isin(invalid_ages)]
    return field[key]


# Attempt to identify the age of the sample.
panel['age'] = 0
for age_field in age_fields:
    key = age_field['key']
    indicator = age_field['indicator']
    ages = determine_age_from_field(panel, codings, key, indicator)
    panel.loc[ages.index, 'age'] = ages

proportion = len(panel.loc[panel['age'] > 0]) / len(panel)
print('Identified ages for %.2f%% of the sample.' % proportion)

#-----------------------------------------------------------------------
# Future work: Parse codings.
#-----------------------------------------------------------------------

# Age cohort: `age2` (looks the best!)
# Other age fields: `catage`, `sexage`

# def code_field():
# coding = codings[key]
# users = past_year_consumers.loc[past_year_consumers[key].astype(str).isin(options)]
# series= sample[key].astype(str).map(coding)


def clean_field(data, codings, key, valid=[]):
    """Remove observations with"""
    options = list(codings[key].keys())
    sample = data.loc[~data[key].astype(str).isin(options)]
    series = sample[key]
    return series

# Parse the individuals weight and height.
panel['weight'] = clean_field(panel, codings, 'wtpound2')
panel['height'] = clean_field(panel, codings, 'htinche2')

# Count the percent with valid weight and height.
valid_weight = panel['weight'].loc[~panel['weight'].isnull()]
valid_height = panel['height'].loc[~panel['height'].isnull()]
print('Percent of sample with valid weight: %.2f%%' % (len(valid_weight) / len(panel)))
print('Percent of sample with valid height: %.2f%%' % (len(valid_height) / len(panel)))

# Income bracket: `income`

# Urban: `coutyp4` (Large Metro, Small Metro, Nonmetro)

# Population density: `pden10`

# Education level: `eduhighcat`


#-----------------------------------------------------------------------
# Analyze consumer characteristics.
#-----------------------------------------------------------------------

# Find percent of population who consumed in last 30 days.
past_30_day_consumers = panel.loc[panel['mjrec'] == 1]
proportion = len(past_30_day_consumers) / len(panel)
print('Proportion who consumed in last 30 days: %.2f%%' % (proportion * 100))

past_year_consumers = panel.loc[(panel['mjrec'] == 1) | (panel['mjrec'] == 2)]
proportion = len(past_year_consumers) / len(panel)
print('Proportion who consumed in 2020: %.2f%%' % (proportion * 100))

# Find the percent that have used medical cannabis in the past year.
past_year_patients = panel.loc[panel['medmjyr'] == 1]
proportion = len(past_year_patients) / len(panel)
print('Proportion with medical cannabis perscription in 2020: %.2f%%' % (proportion * 100))

#-----------------------------------------------------------------------

# Look at the age where people first used cannabis.
invalid_ages = list(codings['mjage'].keys())
invalid_ages = [int(x) for x in invalid_ages]
first_use_age = panel.loc[~panel['mjage'].isin(invalid_ages)]
ax = sns.displot(data=first_use_age, x='mjage', bins=100)
plt.title('Age of First Cannabis Use of US Consumers')
plt.xlabel('Age')
plt.gcf().set_size_inches(18.5, 10.5)
plt.vlines(25, ymin=0, ymax=2000)
plt.show()

# Probability of first-use being greater than 25?
older_25 = first_use_age.loc[first_use_age['mjage'] > 25]
proportion = len(older_25) / len(first_use_age)
print('Probability of first-time use being greater than 25: %.2f%%' % proportion)

#-----------------------------------------------------------------------

# Assign age cohort.
panel['older_25'] = False
panel.loc[panel['mrjaglst'] > 25, 'older_25'] = True

# Identify the ages of people who used in the past year.
past_year_ages = panel.loc[
    (panel['mjrec'] == 1) |
    (panel['mjrec'] == 2)
]
invalid_ages = list(codings['mrjaglst'].keys())
invalid_ages = [int(x) for x in invalid_ages]
past_year_ages = past_year_ages.loc[~past_year_ages['mrjaglst'].isin(invalid_ages)]

# Plot known ages of known consumers.
ax = sns.displot(
    data=past_year_ages,
    x='mrjaglst',
    hue='older_25',
    bins=100,
    legend=False,
)
plt.legend(labels=['Over 25', 'Under 25'])
plt.title('Age of Cannabis Users in the US in 2020')
plt.xlabel('Age')
plt.gcf().set_size_inches(18.5, 10.5)
plt.vlines(25, ymin=0, ymax=150)
plt.show()
print('N:', len(past_year_ages))

# Probability of a cannabis consumer being older than 25?
older_25 = past_year_ages.loc[past_year_ages['mrjaglst'] > 25]
proportion = len(older_25) / len(past_year_ages)
print('Probability of cannabis consumer being greater than 25: %.2f%%' % proportion)


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

# Estimate the number of days consumers consume, who consumed in last 30 days.
coding = codings['mr30est']
options = ['1', '2', '3', '4', '5', '6']
users = past_30_day_consumers.loc[past_30_day_consumers['mr30est'].astype(str).isin(options)]
users['frequency'] = users['mr30est'].astype(str).map(coding)

# Visualize the frequency.
fig, ax = plt.subplots(figsize=(8,8), facecolor='silver')
users['frequency'].value_counts().plot(
    kind='pie',
    autopct=lambda pct: "{:.1f}%".format(pct)
)
plt.title('Reported frequency of cannabis use in past 30 days of US Consumers')
plt.ylabel('')
plt.show()
print('N:', len(users))

# Amount of cannabis bought last time (grams).
key = 'mmlsgms1'
coding = codings[key]
options = ['1', '2', '3']
users = past_year_consumers.loc[past_year_consumers[key].astype(str).isin(options)]
users['amount'] = users[key].astype(str).map(coding)
fig, ax = plt.subplots(figsize=(8, 8))
users['amount'].value_counts(normalize=True).mul(100).plot(kind='bar')
plt.title('Amt of cannabis bought last time - grams by US Consumers')
plt.ylabel('Percent (%)')
plt.show()
print('N:', len(users))

# Estimate annual consumption by multiplying amount times period of use.
# E.g. Monthly consumers are mean amount times 12 and annual consumers are
# simply the average amount of cannabis last bought. 
key = 'mmlsgms1'
prior_max = 28
spend_coding = {
    1: {'min': 1, 'max': 5},
    2: {'min': 5, 'max': 10},
    3: {'min': 10, 'max': prior_max},
}
users['amount_min'] = users[key].apply(lambda x: spend_coding[x]['min'])
users['amount_max'] = users[key].apply(lambda x: spend_coding[x]['max'])
users['grams_bought'] = (users['amount_min'] + users['amount_max']).div(2)

# Assign annual consumers annual consumption.
criterion = users.loc[(users['mjrec'] != 1) & (users['mjrec'] == 2)]
users.loc[criterion.index, 'annual_consumption'] = users['grams_bought']

# Assign monthly consumers annual consumption by the
# number of days bought cannabis past 30 days multipled by the number of grams by 12 months.
key = 'mmbt30dy'
coding = codings[key]
options = list(coding.keys())
users['monthly_consumer'] = 0
criterion = users.loc[users['mjrec'] == 1]
users.loc[criterion.index, 'monthly_consumer'] = 1
frequent_users = users.loc[
    (~users[key].astype(str).isin(options)) &
    (users['monthly_consumer'] == 1)
]
users.loc[
    (users[key].astype(str).isin(options)) &
    (users['monthly_consumer'] == 1),
    'grams_bought'
] = users['grams_bought'] * 12
monthly_amount = frequent_users['grams_bought'] * frequent_users[key]
users.loc[frequent_users.index, 'grams_bought'] = monthly_amount
users.loc[frequent_users.index, 'annual_consumption'] = users['grams_bought'] * 12

# Visualize the estimated amount of grams consumed per year by cannabis consumers.
sample = users.loc[users.annual_consumption <= 1500]
ax = sns.displot(data=sample, x='annual_consumption', bins=100)
plt.title('Estimated Total Grams Consumed per Cannabis User in the US in 2020')
plt.xlabel('Amount (Grams)')
plt.gcf().set_size_inches(18.5, 10.5)
plt.show()
print('N:', len(frequent_users))

# Predict the probability of consuming more than 100g a year.
moderate_users = users.loc[
    (users['annual_consumption'] >= 100) &
    (users['annual_consumption'] <= 1000)
]
heavy_users = users.loc[users['annual_consumption'] > 1000]
proportion_moderate = len(moderate_users) / len(users)
proportion_heavy = len(heavy_users) / len(users)
print('Proportion of consumers who consume 100g-1000g per year: %.2f%%' % (proportion_moderate * 100))
print('Proportion of consumers who consume more than 1000g per year: %.2f%%' % (proportion_heavy * 100))

#-----------------------------------------------------------------------

# FIXME: Code and remove missing values:
# - mrjaglst
# - service
# - height
# - weight



# sample.loc[sample['age'] 'older_25']

# Correlate consumption with various factors.
explanatory_variables = [
    'annual_consumption',
    'monthly_consumer',
    'mrjaglst',
    'height',
    'weight',
    'service',
    # 'older_25',
]
correlation = sample[explanatory_variables].corr()
sns.heatmap(correlation, annot=True, cmap='vlag_r')
plt.show()


#-----------------------------------------------------------------------
# Statistical Model: Heckman Model
#-----------------------------------------------------------------------

# FIXME: `users` is a small subsample. Perhaps sample approximately 4x
# non-consumers to use for creating a model?

# # Assign consumption amount from the users.
# panel['annual_consumption'] = 0
# panel.loc[users.index, 'annual_consumption'] = users['annual_consumption']

# # Correlate the various variables and see which variables move together.
# panel['past_year_consumer'] = 0
# criterion = panel['annual_consumption'] > 0
# panel.loc[criterion, 'past_year_consumer'] = 1

# TODO: Code these variables.
# Identify factors that predict probability of use:
# - service
# - income (bracket)
# - coutyp4 (Large Metro, Small Metro, Nonmetro)
# - pden10 (population density)
# - eduhighcat
# - catage (bracket)

# TODO: Estimate a probit model.
# Y = pd.get_dummies(panel['past_year_consumer'])
# X = panel[explanatory_variables]
# model = sm.Probit(Y, X).fit()
# print(model.summary())

# TODO: Estimate the probit model.


# TODO: Code these variables.
# Identify factors that predict amount of use:
# - mrjaglst
# - income (bracket)
# - htinche2 (height in in.)
# - wtpound2 (weight in pounds)
# - cadrlast (# of drinks in the past month)


# TODO: Estimate a Heckman model to consumption consistently!
# heckman_model = Heckman(
#     endog=
#     exog=
#     exog_select=
# )


#-----------------------------------------------------------------------
# Use the model for prediction!
#-----------------------------------------------------------------------

# Specify a target for prediction.
observation = {
    'age': 30,
    'eduhighcat': 'College graduate',
    'metro': 'Small Metro',
    'older_25': True,
    'service': False,
    'medmjpa2': True, # state_medical
}

# Predict the probability of being a 1st time consumer.

# Predict the probability of being a yearly user.

# Predict the probability of being a monthly user.

# Predict the amount of cannabis consumed per month, given probability
# of being a user (Heckman model!).

# Just for fun: Predict the probability of being a homegrower.


#-----------------------------------------------------------------------
# Consumption analysis
#-----------------------------------------------------------------------

# Correlate probability of consuming and consumption amount with
# various factors.

# Predict the amount consumed from survey responses and compare with the
# estimated actual.


#-----------------------------------------------------------------------
# Health analysis
#-----------------------------------------------------------------------

# Question: Does consuming cannabis or the amount of cannabis consumed
# corretate with a greater or lesser likelihood of adverse health events?
