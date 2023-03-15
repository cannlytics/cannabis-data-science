"""
Cannabis Quality Control
Copyright (c) 2023 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 3/8/2023
Updated: 3/8/2023
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Data Sources:

    - Washington State Liquor and Cannabis Board (WSLCB)
    URL: <https://lcb.box.com/s/xseghpsq2t4i1musxj6mgd7b8rhxe7bm>

    - Curated CCRS Inventory Lab Results
    URL: <https://cannlytics.page.link/ccrs-inventory-lab-results-2022-12-07>

"""
# Standard imports:
import os

# External imports:
import pandas as pd
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

# Specify where your data lives.
DATA_DIR = '.datasets/'
FILENAME = 'ccrs-inventory-lab-results-2022-12-07.xlsx'

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 21,
})


#------------------------------------------------------------------------------
# Get the data.
#------------------------------------------------------------------------------

# Read lab results.
data = pd.read_excel(os.path.join(DATA_DIR, FILENAME))


#------------------------------------------------------------------------------
# Determine standards.
#------------------------------------------------------------------------------

# Calculate the average failure rate.
avg_failure_rate = data['status'].value_counts()['Fail'] / len(data)


# TODO: Calculate the average detection rate of pesticides.


# TODO: Calculate the average detection rate of residual solvents.


#------------------------------------------------------------------------------
# Quantify quality control by lab.
#------------------------------------------------------------------------------

# Get passes / failures by lab.
lab_rates = {}
lab_statuses = data.groupby('lab_licensee_id')['status'].value_counts()
for keys, value in lab_statuses.items():
    lab_licensee_id = keys[0]
    key = keys[1]
    print(f'Lab {lab_licensee_id} {key}:', value)
    rates = lab_rates.get(lab_licensee_id, {})
    rates[key] = value
    lab_rates[lab_licensee_id] = rates

# Calculate average failure rate by lab.
for lab_licensee_id, values in lab_rates.items():
    number_pass = values['Pass']
    number_fail = values.get('Fail', 0)
    failure_rate = number_fail / (number_pass + number_fail)
    lab_rates[lab_licensee_id]['failure_rate'] = failure_rate
    print(f'Lab {lab_licensee_id} failure rate: {round(failure_rate * 100, 2)}%')

# Calculate mean of lab averages.
failure_data = pd.DataFrame.from_dict(lab_rates, orient='index')
mean_failure_rate = failure_data['failure_rate'].mean()
print(f'Mean lab failure rate: {round(mean_failure_rate * 100, 2)}%')
# failure_data['Fail'].sum() / (failure_data['Pass'].sum() + failure_data['Fail'].sum())

# Calculate standard deviation of lab averages.
failure_rate_std = failure_data['failure_rate'].std()
print(f'Lab failure rate std. deviation: {round(failure_rate_std * 100, 2)}%')

# Rank each lab against the standard.
criterion = failure_data['failure_rate'] < mean_failure_rate
failure_data['below_average'] = 0
failure_data.loc[criterion, 'below_average'] = 1
print(failure_data.sort_values(by='failure_rate', ascending=False))

# Insight: What proportion of samples are
# tested at labs with below avg. failure rates?
passed_samples_from_below_avg = failure_data.loc[criterion]['Pass'].sum()
passed_samples_from_above_avg = failure_data.loc[~criterion]['Pass'].sum()


#------------------------------------------------------------------------------
# Quantify quality control by licensee.
#------------------------------------------------------------------------------

# Get passes / failures by licensee.
licensee_rates = {}
licensee_statuses = data.groupby('licensee_id')['status'].value_counts()
for keys, value in licensee_statuses.items():
    _id = keys[0]
    key = keys[1]
    rates = licensee_rates.get(_id, {})
    rates[key] = value
    licensee_rates[_id] = rates

# Calculate average failure rate by licensee.
for _id, values in licensee_rates.items():
    number_pass = values.get('Pass', 0)
    number_fail = values.get('Fail', 0)
    try:
        failure_rate = number_fail / (number_pass + number_fail)
    except ZeroDivisionError:
        failure_rate = None
    licensee_rates[_id]['failure_rate'] = failure_rate

# Calculate mean of licensee averages.
licensee_failure_data = pd.DataFrame.from_dict(licensee_rates, orient='index')
licensee_mean_failure_rate = licensee_failure_data['failure_rate'].mean()
print(f'Mean licensee failure rate: {round(licensee_mean_failure_rate * 100, 2)}%')

# Calculate standard deviation of licensee averages.
licensee_failure_rate_std = licensee_failure_data['failure_rate'].std()
print(f'Licensee failure rate std. deviation: {round(licensee_failure_rate_std * 100, 2)}%')

# Rank each licensee against the standard.
criterion = licensee_failure_data['failure_rate'] < licensee_mean_failure_rate
licensee_failure_data['below_average'] = 0
licensee_failure_data.loc[criterion, 'below_average'] = 1
ranking = licensee_failure_data.sort_values(by='failure_rate', ascending=True)

# Insight: Identify licensees with lowest failure rates.
print('Lowest Failure Rates:')
print(ranking.head(10))

# Insight: Identify licensees with highest failure rates.
print('Highest Failure Rates:')
print(ranking.tail(20))

# Plot failure rate of licensees who are failing.
licensee_failure_data['failure_rate'].loc[
    (licensee_failure_data['failure_rate'] > 0) &
    (licensee_failure_data['failure_rate'] < 1)
].mul(100).hist(bins=100, figsize=(12, 8))
plt.xlabel('Failure Rate (%)')
plt.ylabel('Density')
plt.title('Failure Rates of Licensees with Failing Samples in WA in 2022')

# Insight: What proportion of samples are from
# licensees with below avg. failure rates?
passed_samples_from_below_avg_licensees = licensee_failure_data.loc[criterion]['Pass'].sum()
passed_samples_from_above_avg_licensees = licensee_failure_data.loc[~criterion]['Pass'].sum()
rate = (passed_samples_from_above_avg_licensees / \
       (passed_samples_from_below_avg_licensees + passed_samples_from_above_avg_licensees))
print(f'{round(rate * 100, 2)}% of passing samples from licensees with above avg. failure rates.')
