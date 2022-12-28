"""
Cannabis Participation Analysis
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 4/29/2022
Updated: 5/7/2022
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

Resources:

    - The Heckman Sample Selection Model by Rob Hicks
    https://rlhick.people.wm.edu/stories/econ_407_notes_heckman.html

    - Type 1 and Type 2 Errors
    https://en.wikipedia.org/wiki/Type_I_and_type_II_errors

Setup:

    1. pip install cannlytics

    2. Get a Fed FRED API key and save to `../.env`
    as `FRED_API_KEY`. URL: https://fred.stlouisfed.org/docs/api/api_key.html

    3. Download data to `../.datasets/census/nsduh`
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
from scipy.stats import norm
import statsmodels as sm

# Internal imports.
from heckman import Heckman
try:
    from cannlytics.utils.utils import snake_case
except ImportError:
    from utils import snake_case


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


#-----------------------------------------------------------------------
# Clean the data by parsing codings.
#-----------------------------------------------------------------------

def code_field(data, codings, key, new_key=''):
    """Code a field for observations given codings."""
    if not new_key: new_key = key
    coding = codings[key]
    series = data[key].astype(str).map(coding)
    data.loc[data.index, new_key] = series
    return data


def clean_field(data, codings, key, valid=[]):
    """Remove observations with invalid coding options"""
    invalid = list(codings[key].keys())
    invalid = list(set(invalid) - set(valid))
    sample = data.loc[~data[key].astype(str).isin(invalid)]
    series = sample[key]
    return series


# Parse the individuals weight and height.
panel['weight'] = clean_field(panel, codings, 'wtpound2')
panel['height'] = clean_field(panel, codings, 'htinche2')

# Count the percent with valid weight and height.
valid_weight = panel['weight'].loc[~panel['weight'].isnull()]
valid_height = panel['height'].loc[~panel['height'].isnull()]
print('Sample with valid weight: %.2f%%' % (len(valid_weight) / len(panel) * 100))
print('Sample with valid height: %.2f%%' % (len(valid_height) / len(panel) * 100))


#-----------------------------------------------------------------------
# Code a myriad of other potential predictors.
#-----------------------------------------------------------------------

# See which `int` fields have a high response rate.
percent_valid = {}
int_fields = survey_fields.loc[survey_fields['type'] == 'int']
for index, field in int_fields.iterrows():
    try:
        key = field['key']
        description = field['description']
        series = clean_field(panel, codings, key)
        valid = series.loc[~series.isnull()]
        percentage = round(len(valid) / len(panel) * 100, 2)
        print(f'Valid {description} ({key}): {percentage}%')
        percent_valid[key] = percentage
    except:
        pass

percent_valid = pd.Series(percent_valid)
percent_valid.sort_values(inplace=True, ascending=False)

# Find predictors that apply to many observations.
predictors = {
    'htinche2': 'height', # Height in inches without shoes - recode
    'wtpound2': 'weight', # Weight in pounds - recode
    # 'nmertmt2': 'emerg_rookm', # # of times been treated in emer room past 12 mos
    # 'nmvsopt2': 'out_patient', # # outpatient visits past 12 mos - recode
    # 'iralcyfu': 'first_use_alcohol', # Alcohol year of first use - imputation revised
    # 'iralcage': 'first_use_alcohol_revised', # Alcohol age of first use - imputation revised
    # 'alctry': 'age_first_alcohol', # Age when first drank alcoholic beverage
    # 'iralcfy': 'alcohol frequency', # Alcohol frequency past year - imputation revised
    # 'alcyrtot': 'alcohol_days', # Total # of days used alcohol in past 12 mos
    # 'impydays': 'unable_to_work', # How many day in past yr you were unable to work
}
sample_index = panel.index
for key, value in predictors.items():
    series = clean_field(panel, codings, key)
    panel[value] = series
    sample_index = list(set(sample_index) & set(series.index))

# Identify the portion of the panel with all explanatory variables.
sample = panel.loc[sample_index]
percent_of_panel = round(len(sample) / len(panel) * 100, 2)
print('n =', len(sample), f'| {percent_of_panel}% of the panel.')

# Code income bracket from `income`.
sample = code_field(sample, codings, 'income', 'income_bracket')

# Code urban from `coutyp4` (Large Metro, Small Metro, Nonmetro).
sample = code_field(sample, codings, 'coutyp4', 'urban')

# Code population density from `pden10`.
custom_codings = {
    'pden10': {
        "1": "1 million or more persons",
        "2": "Fewer than 1 million persons",
        "3": "Fewer than 10,000 persons"
    }
}
sample = code_field(sample, custom_codings, 'pden10', 'population_density')

# Code education level from `eduhighcat`.
sample = code_field(sample, codings, 'eduhighcat', 'education')

# Code age level from `catag6`.
sample = code_field(sample, codings, 'catag6', 'age')

# Code service.
sample['armed_forces'] = 0
series = sample.loc[sample['service'] == 1]
sample.loc[series.index, 'armed_forces'] = 1

# Optional: Add more categorical variables.
# - health2 overall health (Excellent, Very Good, Good, Fair/Poor)
# - coldyr2 Other cough/cold med used past 12 mos - specify 2


#-----------------------------------------------------------------------
# Analyze consumer characteristics.
#-----------------------------------------------------------------------

# Determine monthly use from `irmjfm`.
coding = codings['irmjfm']
invalid = list(coding.keys())
consume_monthly = sample.loc[~sample['irmjfm'].astype(int).astype(str).isin(invalid)]
consume_less_than_monthly = sample.loc[~sample.index.isin(consume_monthly.index)]

# Look at percent of past 30 day consumers by factor.
percent = round(len(consume_monthly) / len(sample) * 100, 2)
print('Percent of sample monthly consumers:', percent)

# Optional: Estimate bias.
panel_prop = len(panel.loc[~panel['irmjfm'].astype(int).astype(str).isin(invalid)]) / len(panel)
sample_prop = len(consume_monthly) / len(sample)
bias = (sample_prop - panel_prop) / panel_prop
if bias >= 0: bias += 1
if bias < 0: bias -= 1


def group_plot(codings, series, numeraire, group_by, key, title='', bias=1):
    """"""
    count = series.groupby(group_by)[key].count()
    null_count = numeraire.groupby(group_by)[key].count()
    total = count + null_count
    percent = count / total * 100 * (bias**(-1))
    percent.index = percent.index.astype(str).map(codings[group_by])
    percent.plot(kind='bar')
    if not title: title = group_by.title()
    plt.xlabel(title)
    plt.ylabel('Percent')
    plt.title('Percent of Sample Consuming Monthly')
    plt.show()


# Look at the percent of 30 day users by education (eduhighcat).
group_plot(
    codings,
    consume_monthly,
    consume_less_than_monthly,
    group_by='eduhighcat',
    key='mjrec',
    title='Education',
    bias=bias,
)

# Look at the percent of 30 day users by urban (coutyp4).
group_plot(
    codings,
    consume_monthly,
    consume_less_than_monthly,
    group_by='coutyp4',
    key='mjrec',
    title='Urban Area',
    bias=bias,
)

# Look at the percent of 30 day users by population density (pden10).
group_plot(
    codings,
    consume_monthly,
    consume_less_than_monthly,
    group_by='pden10',
    key='mjrec',
    title='Population Density',
    bias=bias,
)

# Look at the percent of 30 day users by income bracket (income).
group_plot(
    codings,
    consume_monthly,
    consume_less_than_monthly,
    group_by='income',
    key='mjrec',
    title='Income',
    bias=bias,
)

# Look at the percent of 30 day users by age (catag6).
group_plot(
    codings,
    consume_monthly,
    consume_less_than_monthly,
    group_by='catag6',
    key='mjrec',
    title='Age',
    bias=bias,
)

# Look at the percent of 30 day users by service (armed_forces).
custom_codings = {
    'armed_forces': {
        '0': 'No',
        '1': 'Yes',
    }
}
group_plot(
    custom_codings,
    consume_monthly,
    consume_less_than_monthly,
    group_by='armed_forces',
    key='mjrec',
    title='Served in the Armed Forces',
    bias=bias,
)


#-----------------------------------------------------------------------
# Estimate the amount of cannabis consumed by consumers in 2020.
#-----------------------------------------------------------------------

# Visualize the frequency.
consume_monthly['frequency']  = consume_monthly['irmjfm'].astype(int)
frequencies = consume_monthly.groupby('frequency')['frequency'].count() / len(consume_monthly)
frequencies.index = frequencies.index.astype(int)
frequencies.plot(kind='bar')
plt.gcf().set_size_inches(18.5, 10.5)
plt.title('Frequency of cannabis use for monthly consumers')
plt.xlabel('Number of Days')
plt.ylabel('Percent')
plt.show()
print('N:', len(consume_monthly))

# Amount of cannabis bought last time (grams).
key = 'mmlsgms1'
coding = codings[key]
options = ['1', '2', '3']
users = sample.loc[sample[key].astype(str).isin(options)]
users['amount'] = users[key].astype(str).map(coding)
fig, ax = plt.subplots(figsize=(12, 8))
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

# Assumption: If amount bought not specified, then assign the median amount.
median_bought = users['grams_bought'].median()
consume_monthly.loc[consume_monthly.index.isin(users.index), 'grams_bought'] = users.loc[users.index.isin(consume_monthly.index)]['grams_bought']
consume_monthly.loc[~consume_monthly.index.isin(users.index), 'grams_bought'] = median_bought

# Assign monthly consumers annual consumption by the
# number of days bought cannabis past 30 days multipled
# by the number of grams bought last time by 12 months.
key = 'mmbt30dy'
coding = codings[key]
options = list(coding.keys())
frequent_users = consume_monthly.loc[
    ~consume_monthly[key].astype(str).isin(options)
]
monthly_amount = frequent_users['grams_bought'] * frequent_users[key]
annual_amount = monthly_amount * 12
consume_monthly.loc[consume_monthly.index.isin(frequent_users.index), 'grams_bought'] = monthly_amount
consume_monthly.loc[consume_monthly.index.isin(frequent_users.index), 'consumption'] = annual_amount

# Assumption: If not specified, monthly consumers only purchased once in the past month.
consume_monthly.loc[
    consume_monthly[key].astype(str).isin(options),
    'consumption'
] = consume_monthly['grams_bought'] * 12

# Assign annual consumers annual consumption.
consume_annually = sample.loc[(sample['mjrec'] != 1) & (sample['mjrec'] == 2)]
consume_annually.loc[consume_annually.index.isin(users.index), 'grams_bought'] = users.loc[users.index.isin(consume_annually.index)]['grams_bought']
consume_annually.loc[~consume_annually.index.isin(users.index), 'grams_bought'] = median_bought
consume_annually.loc[consume_annually.index, 'consumption'] = consume_annually['grams_bought']

# Assign consumption to the sample.
sample['consumption'] = 0
sample.loc[consume_monthly.index, 'consumption'] = consume_monthly['consumption']
sample.loc[consume_annually.index, 'consumption'] = consume_annually['consumption']


#-----------------------------------------------------------------------
# Analyze consumption.
#-----------------------------------------------------------------------

# Estimate amount monthly consumption for consumers.
upper_limit = sample.loc[sample['consumption'] > 0]['consumption'].quantile(0.95)
subsample = sample.loc[
    (sample['consumption'] <= upper_limit) &
    (sample['consumption'] > 0)
]
percent = round(len(subsample) / len(sample) * 100, 2)
print('Consumers:', len(subsample), f'({percent}%)')
print('Mean:', round(subsample['consumption'].mean() / 28 / 12, 2), 'oz. / mo.')
print('Median:', round(subsample['consumption'].median() / 28 / 12, 2), 'oz. / mo.')

# Predict the probability of consuming more than 100g a year.
heavy_users = subsample.loc[subsample['consumption'] >= 100]
proportion_heavy = len(heavy_users) / len(sample)
print('Sample who consume more than 100g per year: %.2f%%' % (proportion_heavy * 100))

# Visualize moderate user consumption
heavy_users['ounces'] = heavy_users['consumption'].div(12)
ax = sns.displot(data=heavy_users, x='ounces', bins=100)
plt.title('Estimated annual consumption for heavy consumers')
plt.xlabel('Amount (Ounces)')
plt.gcf().set_size_inches(18.5, 10.5)
plt.show()
percent = round(len(heavy_users) / len(sample) * 100, 2)
print('Heavy Users:', len(heavy_users), f'({percent}%)')
print('Mean:', round(heavy_users['consumption'].mean() / 28 / 12, 2), 'oz. / mo.')
print('Median:', round(heavy_users['consumption'].median() / 28 / 12, 2), 'oz. / mo.')


#-----------------------------------------------------------------------
# Statistical Model: Heckman Model
#-----------------------------------------------------------------------

# Assign consumption indicator.
sample['consumer'] = 0
sample.loc[sample['consumption'] > 0, 'consumer'] = 1

# Exclude outliers
subsample = sample.loc[sample['consumption'] < upper_limit]

# Define explanatory variables.
categorical_variables = [
    'age',
    'education',
    'income_bracket',
    # 'armed_forces',
]
continuous_variables = [
    'height',
    'weight',
]

# Create dummy variables for each categorical variable.
for variable in categorical_variables:
    dummies = pd.get_dummies(subsample[variable])
    dummies.columns = [variable + '_' + snake_case(x) for x in dummies.columns]
    subsample = pd.concat([subsample, dummies], axis=1)

# Define explanatory variables
explanatory_variables = continuous_variables
for variable in categorical_variables:
    columns = [x for x in subsample.columns if x.startswith(variable + '_')]
    print('Excluding:', columns[0])
    columns = columns[1:] # Exclude 1st category.
    explanatory_variables += columns

#-----------------------------------------------------------------------

# Estimate a probit model.
Y = subsample['consumer']
X = subsample[explanatory_variables]
model = sm.discrete.count_model.Probit(Y, X).fit()
print(model.summary())

# See how well the probit predict the sample.
predictions = model.predict(X)
subsample = subsample.assign(
    predicted_consumer=0,
    predicted_value=predictions,
)
threshold = subsample['consumer'].mean()
percentile =  predictions.quantile(1 - threshold)
subsample.loc[predictions >= percentile, 'predicted_consumer'] = 1

# Assign predicted consumers.
actuals = subsample['consumer']
predicted = subsample['predicted_consumer']
print(predicted.mean())

# Visualize predictions.
predictions.hist(bins=100)
sns.displot(
    subsample,
    x='predicted_value',
    hue='predicted_consumer'
)
plt.gcf().set_size_inches(18.5, 10.5)
plt.vlines(percentile, ymin=0, ymax=1000)
plt.show()


def confusion_matrix(data, actual, predicted):
    """Create a confusion matrix for a binary prediction.
    |                 | actual true | actual false |
    | predicted true  | true_pos    | false_pos    |
    | predicted false | false_neg   | true_neg     |
    """
    true_pos = len(data.loc[(data[actual] == 1) & (data[predicted] == 1)])
    true_neg = len(data.loc[(data[actual] == 0) & (data[predicted] == 0)])
    false_pos = len(data.loc[(data[actual] == 0) & (data[predicted] == 1)])
    false_neg = len(data.loc[(data[actual] == 1) & (data[predicted] == 0)])
    return [[true_pos, false_pos], [false_neg, true_neg]]


# Create confusion matrix.
# Note Type 1 and Type 2 errors.
cm = confusion_matrix(subsample, 'consumer', 'predicted_consumer')
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]
accuracy = (tp + tn) / (tp + tn + fp + fn)
true_negative_rate = tn / (tn + fp)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
print('Accuracy: %.2f%%' % (accuracy * 100))
print('True negative rate: %.2f%%' % (true_negative_rate * 100))
print('Sensitivity: %.2f%%' % (sensitivity * 100))
print('Specificity: %.2f%%' % (specificity * 100))
print('Precision: %.2f%%' % (precision * 100))


#-----------------------------------------------------------------------
# Estimate a Heckman model to predict consumption consistently!
# Assumptions: 
#   - Education affects if you consume and NOT how much you consume.
#-----------------------------------------------------------------------

# Format the variables for the Heckman model.
exog_variables = [x for x in explanatory_variables if not x.startswith('educ')]
select = subsample[explanatory_variables]
exog = subsample[exog_variables]
exog = sm.tools.tools.add_constant(exog)
endog = subsample['consumption']
endog.loc[endog == 0] = np.nan

# Fit the model.
heckman_model = Heckman(
    endog=endog.values,
    exog=exog.values,
    exog_select=select.values,
)
results = heckman_model.fit()
print(results.summary())

#-----------------------------------------------------------------------

# Look at predictions of 1st stage.
predicted_consumers = pd.Series(results.select_res.predict())
percentile =  predicted_consumers.quantile(1 - threshold)
pd.Series(predicted_consumers).hist(bins=100)
plt.gcf().set_size_inches(18.5, 10.5)
plt.vlines(percentile, ymin=0, ymax=1000)
plt.show()

#-----------------------------------------------------------------------

# Predict if a given individual will consume.
observation = {
    'height': 12 * 5 + 9,
    'weight': 150,
    'age_18_25_years_old': 1,
    'age_26_34_years_old': 0,
    'age_35_49_years_old': 0,
    'age_50_64_years_old': 0,
    'age_65_or_older': 0,
    'education_college_graduate': 1,
    'education_high_school_grad': 0,
    'education_less_high_school': 0,
    'education_some_colltoassoc_dg': 0,
    'income_bracket_dollars_50_000_dollars_74_999': 0,
    'income_bracket_dollars_75_000_or_more': 0,
    'income_bracket_less_than_dollars_20_000': 0,
}

# Step 1. Predict if observation will consume.
z_hat = np.array(list(observation.values()))
predicted_value = results.select_res.predict(z_hat)[0]
print('Predicted value:', predicted_value)
if predicted_value < percentile:
    print('Predicted non-consumer.')

# Step 2. If they consume, predict how much they consume.
if predicted_value > percentile:
    print('Predicted consumer.')

    # Get parameters from the 1st stage.
    im = norm.pdf(predicted_value) / norm.cdf(predicted_value)
    im_param = results.params_inverse_mills
    rho = results.corr_eqnerrors
    sigma = np.sqrt(results.var_reg_error)

    # Estimate 2nd stage.
    fields = {k: v for k, v in observation.items() if not k.startswith('educ')}
    x_hat = np.array([1] + list((fields.values()))) # Adding a constant.
    xb = np.dot(x_hat, results.params)
    # or
    # xb = results.predict(x_hat)
    predicted_consumption = xb + im_param * im
    # or
    # predicted_consumption = xb + rho * sigma * im
    print('Predicted annual consumption (grams): %.2f' % predicted_consumption)

    # Future work: Double-check by OLS on X and rho * sigma_u * lambda(Z * gamma)


#-----------------------------------------------------------------------
# Make predictions for the sample.
#-----------------------------------------------------------------------

# Predict consumption for sample and compare to actual estimates.
first_stage_params = results.select_res.params
im_param = results.params_inverse_mills
xb = results.predict(exog.values)
im = []
predicted_consumers = []
for value in select.values:

    # Calculate inverse mills ratio.
    z_times_gamma = np.dot(value, first_stage_params)
    imr = norm.pdf(z_times_gamma) / norm.cdf(z_times_gamma)
    im.append(imr)

    # Calculate probability of consuming.
    prob_consumer = norm.cdf(z_times_gamma)
    if prob_consumer < percentile:
        predicted_consumers.append(0)
    else:
        predicted_consumers.append(1)

# Predict!
y_hat = xb + np.dot(im_param, im)

# Record predicted participation and consumption.
subsample['predicted_consumption'] = pd.Series(y_hat)
subsample['predicted_consumer'] = pd.Series(predicted_consumers)


#-----------------------------------------------------------------------
# Visualize the model results.
#-----------------------------------------------------------------------

# Visualize moderate user consumption
criterion = subsample['predicted_consumer'] == 0
subsample.loc[criterion]['predicted_consumption'].hist(
    bins=100,
    density=True,
    alpha=0.7,
    label='Non-consumers',
)
criterion = subsample['predicted_consumer'] == 1
subsample.loc[criterion]['predicted_consumption'].hist(
    bins=100,
    density=True,
    alpha=0.7,
    label='Consumers',
)
subsample.loc[subsample['consumption'] < 200]['consumption'].hist(bins=100, density=True)
plt.legend(loc='best')
plt.title('Estimated annual consumption for consumers')
plt.xlabel('Amount (Grams)')
plt.gcf().set_size_inches(12, 8)
plt.show()

# Estimated participation rate.
percent = round(subsample['predicted_consumer'].mean() * 100, 2)
print('Predicted consumers: %.2f%%' % percent)

# Estimated monthly consumption of users.
predicted_consumers = subsample.loc[subsample['predicted_consumer'] == 1]
print('Predicted Mean:', round(predicted_consumers['predicted_consumption'].mean() / 28 / 12, 2), 'oz. / mo.')
print('Predicted Median:', round(predicted_consumers['predicted_consumption'].median() / 28 / 12, 2), 'oz. / mo.')
print('Actual Estimated Mean:', round(subsample['consumption'].mean() / 28 / 12, 2), 'oz. / mo.')
print('Actual Estimated Median:', round(subsample['consumption'].median() / 28 / 12, 2), 'oz. / mo.')

# Measure prediction accuracy.
# |                 | actual true | actual false |
# | predicted true  | true_pos    | false_pos    |
# | predicted false | false_neg   | true_neg     |
cm = confusion_matrix(subsample, 'consumer', 'predicted_consumer')
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]
accuracy = (tp + tn) / (tp + tn + fp + fn)
true_negative_rate = tn / (tn + fp)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
print('Accuracy: %.2f%%' % (accuracy * 100))
print('True negative rate: %.2f%%' % (true_negative_rate * 100))
print('Sensitivity: %.2f%%' % (sensitivity * 100))
print('Specificity: %.2f%%' % (specificity * 100))
print('Precision: %.2f%%' % (precision * 100))
