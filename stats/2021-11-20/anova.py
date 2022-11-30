"""
ANOVA applied to Massachusetts Cannabis Data
Copyright (c) 2021 Cannlytics and the Cannabis Data Science Meetup Group

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 11/20/2021
Updated: 11/20/2021
License: MIT License <https://opensource.org/licenses/MIT>

Objective:
    
    Count the number of DBE licenses by cultivation tier.
    Perform ANOVA to see if there is a difference in cultivation size
    measured by square feet between DBE cultivations and non-DBE cultivations.
    
Data Sources:
    
    MA Cannabis Control Commission
    - Approved Massachusetts Licensees: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/hmwt-yiqy
    - Plant Activity and Volume: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/j3q7-3usu

Reading:
    
    ANOVA Theory
    https://en.wikipedia.org/wiki/Survival_function
    https://www.dummies.com/education/math/statistics/finding-the-power-of-a-hypothesis-test/
    https://en.wikipedia.org/wiki/Power_of_a_test
    https://en.wikipedia.org/wiki/Analysis_of_variance
    
    ANOVA Python Code
    https://raphaelvallat.com/pingouin.html
    https://github.com/raphaelvallat/pingouin/blob/master/notebooks/01_ANOVA.ipynb
    https://github.com/raphaelvallat/pingouin
    https://towardsdatascience.com/anova-t-test-and-other-statistical-tests-with-python-e7a36a2fdc0c
    https://stackoverflow.com/questions/60495977/difference-in-means-test-on-pandass-summary-statistics
    https://campus.datacamp.com/courses/statistical-simulation-in-python/resampling-methods?ex=13
    
    ANOVA Examples
    https://www.statsmodels.org/dev/generated/statsmodels.stats.power.tt_ind_solve_power.html
    https://www.statsmodels.org/stable/anova.html
    https://www.statsmodels.org/stable/examples/notebooks/generated/interactions_anova.html
"""
from dotenv import dotenv_values
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

# Internal imports
from utils import (
    reverse_dataframe,
)

#--------------------------------------------------------------------------
# Get MA public cannabis data.
#--------------------------------------------------------------------------

# Setup Socrata API, setting the App Token for request headers.
config = dotenv_values('../.env')
app_token = config.get('APP_TOKEN', None)
headers = {'X-App-Token': app_token}
base = 'https://opendata.mass-cannabis-control.com/resource'

# Get licensees data.
url = f'{base}/hmwt-yiqy.json'
params = {'$limit': 10000,  '$order': 'app_create_date DESC'}
response = requests.get(url,  headers=headers, params=params)
licensees = pd.DataFrame(response.json(), dtype=float)

# Get production stats (total employees, total plants, etc.) j3q7-3usu
url = f'{base}/j3q7-3usu.json'
params = {'$limit': 2000, '$order': 'activitysummarydate DESC'}
response = requests.get(url,  headers=headers, params=params)
production = pd.DataFrame(response.json(), dtype=float)
production = reverse_dataframe(production)
production['date'] = pd.to_datetime(production['activitysummarydate'])
production.set_index('date', inplace=True)

# Calculate sales difference, coding outliers and negatives as 0.
production['sales'] = production['salestotal'].diff()
outlier = production.loc[production.sales >= 10000000]
production.at[outlier.index, 'sales'] = 0
negatives = production.loc[production.sales < 0]
production.at[negatives.index, 'sales'] = 0

#--------------------------------------------------------------------------
# Find conditional data.
#--------------------------------------------------------------------------

# Identify licensees with final licenses.
# These are the licenses that are assumed to be currently operating.
final_licensees = licensees.loc[
    (licensees['approved_license_type'] == 'FINAL LICENSE')
]

#--------------------------------------------------------------------------
# Re-look at weekly averages using only licensees with final licenses.
#--------------------------------------------------------------------------

# Create weekly series.
weekly_sales = production.sales.resample('W-SUN').sum()
weekly_plants = production['total_plantfloweringcount'].resample('W-SUN').mean()
weekly_employees = production.total_employees.resample('W-SUN').mean()

# Create total licensees series.
production['total_retailers'] = 0
production['total_cultivators'] = 0
production['total_licensees'] = 0
for index, _ in production.iterrows():
    timestamp = index.isoformat()
    production.at[index, 'total_retailers'] = len(licensees.loc[
        (licensees.license_type == 'Marijuana Retailer') &
        (licensees['cnb_dt_of_final_licensure'] <= timestamp)
    ])
    production.at[index, 'total_cultivators'] = len(licensees.loc[
        (licensees.license_type == 'Marijuana Cultivator') &
        (licensees['cnb_dt_of_final_licensure'] <= timestamp)
    ])
    production.at[index, 'total_licensees'] = len(licensees.loc[
        (licensees['cnb_dt_of_final_licensure'] <= timestamp)
    ])

# Create weekly averages.
weekly_total_retailers = production['total_retailers'].resample('W-SUN').mean()
weekly_total_cultivators = production['total_cultivators'].resample('W-SUN').mean()
weekly_total_licensees = production['total_licensees'].resample('W-SUN').mean()

# Estimate sales per retailer.
sales_per_retailer = weekly_sales / weekly_total_retailers
(sales_per_retailer / 1000).plot()
plt.show()

# Estimate plants per cultivator.
plants_per_cultivator = weekly_plants / weekly_total_cultivators
plants_per_cultivator.plot()
plt.show()

# Estimate employees per licensee.
employees_per_license = weekly_employees / weekly_total_licensees
employees_per_license.plot()
plt.show()

#--------------------------------------------------------------------------
# Find conditional data and perform ANOVA of tier size by license type.
#--------------------------------------------------------------------------

# Identify DBEs (disadvantaged business enterprise).
def identify_dbe(row):
    """Idenfity if an observed licensee is a DBE."""
    if 'Not a DBE' in row['dbe']:
        value = 0
    else:
        value = 1
    return value

licensees['dbe_license'] = licensees.apply(identify_dbe, axis=1)

# Restirct analysis to cultivators.
cultivators = licensees.loc[
    licensees['license_type'] == 'Marijuana Cultivator'
]

# Restrict analysis to cultivators with a final license.
operating_cultivators = cultivators.loc[
    cultivators['cnb_dt_of_final_licensure'] >= production.index.min().isoformat()
]

# Calculate percent of operating cultivators that are DBE.
dbe_cultivators = operating_cultivators.loc[
    operating_cultivators['dbe_license'] == 1.0
]
percent_dbe_cultivators = len(dbe_cultivators) / len(operating_cultivators) * 100
print('Percent of DBE cultivators: %.2f' % percent_dbe_cultivators)

# Plot distributions of sq feet of establishment.
cultivators['square_footage_establishment'].hist(bins=40)
dbe_cultivators['square_footage_establishment'].hist(bins=40)
plt.show()

# Looking at the mean and standard deviation of square_footage_establishment.
tiers = cultivators['cultivation_tier'].unique()
tiers = [str(x) for x in tiers]
tiers.sort()
print('|----------------------------------------------------|')
print('| Final licenses by tier and DBE application status. |')
print('|----------------------------------------------------|')
print('| Tier | Number of Cultivations | Number of DBE Cultivations |')
print('------ | ---------------------- | -------------------------- |')
for tier in tiers:
    tier_cultivations = cultivators.loc[
        cultivators['cultivation_tier'] == tier
    ]
    tier_dbe_cultivations = dbe_cultivators.loc[
        dbe_cultivators['cultivation_tier'] == tier
    ]
    print('|', tier, '|' , len(tier_cultivations), '|' , len(tier_dbe_cultivations), '|')


print('|----------------------------------------------------|')
print('| Building Size by tier and DBE application status. |')
print('|----------------------------------------------------|')
print('| Tier | Avg. Size of Cultivation | Avg Size of DBE Cultivation |')
print('------ | ------------------------ | --------------------------- |')
for tier in tiers:
    tier_cultivations = cultivators.loc[
        cultivators['cultivation_tier'] == tier
    ].square_footage_establishment.mean()
    tier_dbe_cultivations = dbe_cultivators.loc[
        dbe_cultivators['cultivation_tier'] == tier
    ].square_footage_establishment.mean()
    print('|', tier, '|' , round(tier_cultivations, 2), '|' , round(tier_dbe_cultivations, 2), '|')
   
#--------------------------------------------------------------------------
# Perform ANOVA to determine if DBE and non-DBE have statistically
# different means or variances in cultivation size.
#--------------------------------------------------------------------------

# FIXME: Perform ANOVA using non-DBE and DBE series.

import pingouin as pg

# Calculate the correlation between two columns.
pg.corr(
        x=cultivators['dbe_license'],
        y=cultivators['square_footage_establishment']
)

# Visualize the correlation.
sns.set(style='white', font_scale=1.2)
g = sns.JointGrid(
    data=cultivators,
    x='dbe_license',
    y='square_footage_establishment',
    height=5
)
g = g.plot_joint(sns.regplot, color="xkcd:muted blue")
g = g.plot_marginals(sns.distplot, kde=False, bins=12, color="xkcd:bluey grey")
plt.tight_layout()

# Run a one-way ANOVA.
aov = pg.anova(
    data=cultivators,
    dv='square_footage_establishment',
    between='dbe_license',
    detailed=True
)
print(aov)

# Calculate the power of the test.
k = cultivators['dbe_license'].nunique()  # Number of groups
n = cultivators.shape[0] / k  # Number of observations per group
achieved_power = pg.power_anova(eta=aov.loc[0, 'np2'], k=k, n=n, alpha=0.05)
print('Achieved power: %.4f' % achieved_power)

# Check normality.
pg.normality(
    cultivators,
    group='dbe_license',
    dv='square_footage_establishment'
)
