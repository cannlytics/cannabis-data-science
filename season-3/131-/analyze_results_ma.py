"""
Analyze Results from MA PRR
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 10/17/2023
Updated: 10/17/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

References:

    - MA microbe limits: <https://www.mass.gov/doc/exhibit-6-analysis-requirements-for-microbiological-contaminants-and-mycotoxins-in-medical/download>
    - MA sampling protocol: <https://masscannabiscontrol.com/wp-content/uploads/20200724_Testing-Protocol-for-Comment.pdf>

"""
# External imports:
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib import cm
import numpy as np
import pandas as pd
import seaborn as sns

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})


# === Get the data ===

# Read MA lab results.
ma_results = pd.read_csv('./data/TestingTHC-THCA-YeastMold-Apr-Dec2021-FINAL.csv')


# === Clean the data ===

# Rename certain columns.
ma_results = ma_results.rename(columns={
    'ProductCategory': 'product_type',
    'PackageLabel': 'label',
    'TestType': 'test_type',
    'TestResult': 'test_result',
    'TestingLabId': 'lab',
    'TestPerformedDate': 'date_tested',
})

# Add a date column.
ma_results['date'] = pd.to_datetime(ma_results['date_tested'])
ma_results['month_year'] = ma_results['date'].dt.to_period('M')


# === Analysis of re-tests. ===

# Grouping by label and filtering groups with multiple test dates
multi_test_labels = ma_results.groupby('label').filter(lambda x: x['date_tested'].nunique() > 1)
print(multi_test_labels[['label', 'test_type', 'test_result', 'lab', 'date_tested']])

# Creating a pivot table
pivot_df = multi_test_labels.pivot_table(
    index=['label', 'date_tested', 'lab'],
    columns='test_type',
    values='test_result',
    aggfunc='first',
).reset_index()
pivot_df.columns.name = None  
pivot_df.columns = ['label', 'date_tested', 'lab', 'delta_9_thc', 'thca', 'yeast_and_mold']
print(pivot_df)

# Group by labels and then get the first and last THC/THCA values.
pivot_df['date'] = pd.to_datetime(pivot_df['date_tested'])
pivot_df = pivot_df.sort_values(by='date')
result = pivot_df.groupby('label').agg(
    first_tested=pd.NamedAgg(column='date_tested', aggfunc='first'),
    last_tested=pd.NamedAgg(column='date_tested', aggfunc='last'),
    delta_9_thc_first=pd.NamedAgg(column='delta_9_thc', aggfunc='first'),
    delta_9_thc_last=pd.NamedAgg(column='delta_9_thc', aggfunc='last'),
    thca_first=pd.NamedAgg(column='thca', aggfunc='first'),
    thca_last=pd.NamedAgg(column='thca', aggfunc='last'),
    lab=pd.NamedAgg(column='lab', aggfunc='first'),
).reset_index()
print(result)

# Calculate differences
result['delta_9_thc_diff'] = result['delta_9_thc_last'] - result['delta_9_thc_first']
result['thca_diff'] = result['thca_last'] - result['thca_first']

# Calculate total_thc values
result['total_thc_first'] = result['delta_9_thc_first'] + result['thca_first'] * 0.877
result['total_thc_last'] = result['delta_9_thc_last'] + result['thca_last'] * 0.877
result['total_thc_diff'] = result['total_thc_last'] - result['total_thc_first']
print(result)

# Look at the difference by lab.
result.groupby('lab', as_index=False)[['delta_9_thc_diff', 'thca_diff', 'total_thc_diff']].count()
result.groupby('lab', as_index=False)[['delta_9_thc_diff', 'thca_diff', 'total_thc_diff']].mean()

# Plot the distribution of differences for a given lab.
result.loc[result['lab'] == 'LabB']['total_thc_diff'].hist(bins=100)


# === Analysis of delta-9 THC and THCA ===

# Creating a pivot table
pivot_df = ma_results.pivot_table(
    index=['label', 'date_tested', 'lab'],
    columns='test_type',
    values='test_result',
    aggfunc='first',
).reset_index()
pivot_df.columns.name = None  
pivot_df.columns = ['label', 'date_tested', 'lab', 'delta_9_thc', 'thca', 'yeast_and_mold']
pivot_df['date'] = pd.to_datetime(pivot_df['date_tested'])
print(pivot_df)

# Drop duplicates.
pivot_df.drop_duplicates(subset=['label'], inplace=True)

# What percentage THCA is unreasonable?
pivot_df['thca'].hist(bins=100)
print(pivot_df['thca'].quantile(0.99))

# What percentage delta-9 THC is unreasonable?
pivot_df['delta_9_thc'].hist(bins=100)
print(pivot_df['delta_9_thc'].quantile(0.99))

# What percentage total THC is unreasonable?
pivot_df['total_thc'] = pivot_df['delta_9_thc'] + pivot_df['thca'] * 0.877
pivot_df['total_thc'].hist(bins=100)
print(pivot_df['total_thc'].quantile(0.99))


# === Total THC analysis ===

# Set date_tested as index
pivot_df.set_index('date', inplace=True)

# Resampling data by month and calculating the mean of total_thc
monthly_thc = pivot_df['total_thc'].resample('M').mean()

# Plotting total_thc by month
monthly_thc.plot()
plt.title('Average Total THC by Month in MA in 2021')
plt.xlabel('Month')
plt.ylabel('Average Total THC (%)')
plt.xticks(rotation=45)
plt.ylim(15, 20)
plt.savefig(f'figures/ma-total-thc-by-month.png', bbox_inches='tight', dpi=300)
plt.show()

# Plotting total_thc by month by lab
monthly_thc_by_lab = pivot_df.groupby('lab')['total_thc'].resample('M').mean().unstack('lab')
sns.set_palette(sns.color_palette('plasma_r', 9))
ax = monthly_thc_by_lab.plot(alpha=0.7)
plt.title('Average Total THC by Month by Lab in MA in 2021')
plt.xlabel('Month')
plt.ylabel('Average Total THC (%)')
plt.xticks(rotation=45)
plt.legend().set_visible(False)
for i, lab in enumerate(monthly_thc_by_lab.columns):
    y = monthly_thc_by_lab[lab].dropna().iloc[-1]
    x = monthly_thc_by_lab[lab].dropna().index[-1]
    ax.text(
        x,
        y,
        lab,
        color=sns.color_palette('plasma_r', 9)[i],
        fontsize=24,
        ha='left',
        va='center',
    )
plt.savefig(f'figures/ma-total-thc-by-lab-by-month.png', bbox_inches='tight', dpi=300)
plt.show()


# === Microbe analysis ===

# Histogram
filtered_df = pivot_df.dropna(subset=['yeast_and_mold'])
filtered_df.loc[filtered_df['yeast_and_mold'] > 10_000]['yeast_and_mold'].hist(bins=100, alpha=0.75)
plt.axvline(10_000, color='r', linestyle='dashed', linewidth=1)
plt.xlabel('Yeast and Mold Counts')
plt.ylabel('Frequency')
plt.title('Histogram of Yeast and Mold Detections above 10,000')
plt.legend(['State Limit (10,000)', 'Yeast and Mold Counts'])
x_ticks = np.arange(10000, max(filtered_df['yeast_and_mold'])+1000, 1000)
ax.set_xticks(x_ticks)
ax.set_xticklabels([f'{int(x):,}' for x in x_ticks], rotation=45)
plt.show()
fails = filtered_df.loc[filtered_df['yeast_and_mold'] > 10_000]
print(fails[['label', 'date_tested', 'lab', 'yeast_and_mold']])

# Visualize failure rates.
pivot_df['fail'] = pivot_df['yeast_and_mold'] > 10_000
fail_counts = pivot_df['fail'].value_counts()
fail_percentages = (fail_counts / fail_counts.sum()) * 100
colors = cm.coolwarm(pivot_df['fail'].value_counts(normalize=True))
ax = pivot_df['fail'].value_counts().plot(
    kind='bar',
    color=[colors[-1], colors[0]]
)
ax.get_yaxis().set_major_formatter(StrMethodFormatter('{x:,.0f}'))
plt.xticks(
    ticks=[0, 1],
    labels=['Below 10,000 CFU/g', 'Above 10,000 CFU/g'],
    rotation=0,
)
for i, (count, percentage) in enumerate(zip(fail_counts, fail_percentages)):
    ax.text(i, count, f'{percentage:.1f}%', color='black', ha='center', va='bottom')
plt.ylabel('Number of Samples')
plt.title('Total Yeast and Mold Detections in MA in 2021', pad=24)
plt.xlabel('Pass/Fail')
plt.savefig(f'figures/ma-yeast-and-mold-failure-rate-2021.png', bbox_inches='tight', dpi=300)
plt.show()
failure_rate = len(fails) / len(pivot_df)
print('Failure rate: %0.2f%%' % (failure_rate * 100))

# Visualize failure rate by lab.
samples_tested_by_lab = pivot_df['lab'].value_counts()
failures_by_lab = pivot_df.groupby('lab')['fail'].sum()
failure_rate_by_lab = pivot_df.groupby('lab')['fail'].mean()
failure_rate_by_lab = failure_rate_by_lab.sort_values()
plt.figure(figsize=(18, 16/1.618))
ax = sns.barplot(
    x=failure_rate_by_lab.index,
    y=failure_rate_by_lab.values * 100,
    palette='coolwarm'
)
for i, p in enumerate(ax.patches):
    lab = failure_rate_by_lab.index[i]
    ax.annotate(
        f'{failures_by_lab[lab]:,.0f} / {samples_tested_by_lab[lab]:,.0f}',
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center',
        va='bottom',
        fontsize=24,
        color='black',
        xytext=(0, 3),
        textcoords='offset points'
    )
plt.ylabel('Failure Rate (%)', fontsize=28, labelpad=10)
plt.xlabel('')
plt.title('Total Yeast and Mold Failure Rate by Lab in MA in 2021', fontsize=34)
plt.xticks(rotation=45)
plt.figtext(
    0,
    -0.075,
    'Note: Statistics are calculated from 31,613 package lab tests for total yeast and mold performed between 4/1/2021 and 12/31/2021 in Massachusetts. The number of tests above the state limit, 10,000 CFU/g, and the total number of tests are shown for each lab.',
    ha='left',
    fontsize=24,
    wrap=True
)
plt.tight_layout()
plt.savefig(f'figures/ma-yeast-and-mold-failure-rate-by-lab-2021.png', bbox_inches='tight', dpi=300)
plt.show()


# === Future work: See if there is a correlation between THC and yeast and mold. ===

import scipy.stats

# Correlate total THC and yeast and mold.
clean_data = pivot_df.dropna(subset=['total_thc', 'yeast_and_mold'])
correlation, _ = scipy.stats.pearsonr(clean_data['total_thc'], clean_data['yeast_and_mold'])
print("Correlation between total THC and total yeast and mold: %.2f" % correlation)


import statsmodels.api as sm

# Defining the dependent variable (fail) and the independent variable (total_thc)
X = clean_data['total_thc']
X = sm.add_constant(X)  # Adding a constant
y = clean_data['fail'].astype(int)  # Converting boolean to int

# Performing logistic regression
model = sm.Logit(y, X).fit()
print(model.summary())

# Calculate the marginal effect of a 1% increase in total_thc
params = model.params
mean_thc = clean_data['total_thc'].mean()
def calculate_probability(thc, params):
    return 1 / (1 + np.exp(-(params['const'] + params['total_thc'] * thc)))
prob_at_mean = calculate_probability(mean_thc, params)
prob_at_mean_plus_1 = calculate_probability(mean_thc * 1.01, params)
marginal_effect = prob_at_mean_plus_1 - prob_at_mean
print(f"Marginal effect of a 1% increase in total THC on the probability of failure is {marginal_effect * 100:.2f}%.")

# === Future work: Estimate cannabinoid consumption in MA ===
