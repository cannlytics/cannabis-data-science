"""
Total Yeast and Mold
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 9/26/2023
Updated: 9/27/2023
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    This script begins to analyze the Maryland cannabis lab result data. 

Data Source:

    - Maryland Test Results 2023-06-30
    URL: <>

"""
# Standard imports:

# External imports:
import random
import matplotlib.pyplot as plt
import pandas as pd


# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})


#-----------------------------------------------------------------------
# Get the data!
# URL: <>
#-----------------------------------------------------------------------

# Read in lab result data.
results = pd.read_csv('data/md-lab-results-2023-09-27.csv')
print('Data shape:', results.shape)


#-----------------------------------------------------------------------
# Augment the data.
#-----------------------------------------------------------------------

# Calculate total THC.
results['total_thc'] = results['THCA'].mul(0.877) + results['THC']
results['total_thc'].describe()


#-----------------------------------------------------------------------
# Look at the data
#-----------------------------------------------------------------------

# Isolate various product types.
r_and_d = results.loc[results['product_subtype'] == 'R&D Testing']
flower = results.loc[results['product_subtype'] == 'Raw Plant Material']
concentrates = results.loc[results['product_subtype'] == 'Inhalable/Vape Concentrate']
stability_t1 = results.loc[results['product_subtype'] == 'Stability T1']
stability_t2 = results.loc[results['product_subtype'] == 'Stability T2']
stability_t3 = results.loc[results['product_subtype'] == 'Stability T3']

# Look at total THC.
sample = flower[flower['total_thc'].notna()]
sample = flower[(flower['total_thc'] > 0) & (flower['total_thc'] < 100)]
sample['total_thc'].hist(bins=100, range=(0, 40))
plt.title('Total THC in Maryland Cannabis from 2017 to 2023')
plt.xlabel('Percent (%)')
plt.ylabel('Count')
plt.show()
sample['total_thc'].describe()

# Look at total yeast and mold.
key = 'Total Yeast and Mold Count'
sample = flower[flower[key].notna()]
sample = flower[(flower[key] > 0) & (flower[key] < 50_000)]
sample[key].hist(bins=100, range=(0, 100_000))
plt.title('Total Yeast and Mold in Maryland Cannabis from 2017 to 2023')
plt.xlabel('CFU/g')
plt.ylabel('Count')
plt.show()
sample[key].describe()


#-----------------------------------------------------------------------
# Analyze the data: See if specific strains have statistically
# different levels of yeast and mold.
#-----------------------------------------------------------------------

# Identify all the unique strains.
strains = list(flower['StrainName'].unique())
print('Unique strains:', len(strains))

cultivar = 'Jack Herer'

from scipy.stats import ttest_ind, shapiro, levene

# Filtering datasets for Jack Herer and others
jack_herer_thc = flower[(flower['StrainName'] == cultivar) & flower['total_thc'].notna()]['total_thc']
other_flowers_thc = flower[(flower['StrainName'] != cultivar) & flower['total_thc'].notna()]['total_thc']
jack_herer_mold = flower[(flower['StrainName'] == cultivar) & flower[key].notna()][key]
other_flowers_mold = flower[(flower['StrainName'] != cultivar) & flower[key].notna()][key]

# Perform t-test (assuming unequal variances by default)
t_stat_thc, p_val_thc = ttest_ind(jack_herer_thc, other_flowers_thc, equal_var=False)
t_stat_mold, p_val_mold = ttest_ind(jack_herer_mold, other_flowers_mold, equal_var=False)

print("\nT-test Results")
print(f"THC: t-statistic = {t_stat_thc}, p-value = {p_val_thc}")
print(f"Mold: t-statistic = {t_stat_mold}, p-value = {p_val_mold}")


import statsmodels.api as sm

# Creating a dataframe for regression
reg_df = flower[flower[key].notna()][['StrainName', key]]
reg_df['is_jack_herer'] = reg_df['StrainName'].apply(lambda x: 1 if x == cultivar else 0)
X = reg_df['is_jack_herer']
X = sm.add_constant(X)
y = reg_df[key]
model = sm.OLS(y, X).fit()
print(model.summary())


#-----------------------------------------------------------------------
# Visualize stability.
#-----------------------------------------------------------------------

# Find strains that have been tested at all three time points.
stability_strains_1 = list(stability_t1['StrainName'].unique())
stability_strains_2 = list(stability_t2['StrainName'].unique())
stability_strains_3 = list(stability_t3['StrainName'].unique())
common_strains = set(stability_strains_1) & set(stability_strains_2) & set(stability_strains_3)
common_strains = list(common_strains)
print(common_strains)

# Look at the stability of a particular strain.
strain = 'Dog Patch'
print(stability_t1.loc[stability_t1['StrainName'] == strain][['TestingFacilityId', 'TestYear', 'THCA']])
print(stability_t2.loc[stability_t2['StrainName'] == strain][['TestingFacilityId', 'TestYear', 'THCA']])
print(stability_t3.loc[stability_t3['StrainName'] == strain][['TestingFacilityId', 'TestYear', 'THCA']])

# Plot a sample of strains.
sampled_strains = random.sample(common_strains, 5)
for strain in sampled_strains:
    thca_t1 = stability_t1.loc[stability_t1['StrainName'] == strain, 'THCA']
    thca_t2 = stability_t2.loc[stability_t2['StrainName'] == strain, 'THCA']
    thca_t3 = stability_t3.loc[stability_t3['StrainName'] == strain, 'THCA']

    # Assuming there's a clear order for the time points, like T1 < T2 < T3.
    plt.plot(['T1', 'T2', 'T3'], [thca_t1.mean(), thca_t2.mean(), thca_t3.mean()], label=strain)

plt.title('Average THCA Levels Over Time for Common Strains')
plt.xlabel('Time Point')
plt.ylabel('Percent (%)')
plt.legend()
plt.show()


#-----------------------------------------------------------------------
# Perform stability analysis.
#-----------------------------------------------------------------------

# Calculate the average change.
average_changes = []

# For each strain in common strains
for strain in common_strains:
    thca_t1 = stability_t1.loc[stability_t1['StrainName'] == strain, 'THCA'].mean()
    thca_t3 = stability_t3.loc[stability_t3['StrainName'] == strain, 'THCA'].mean()

    # Calculate the average change from T1 to T3 for each strain
    change = thca_t3 - thca_t1
    average_changes.append(change)

# Calculate the overall average change for all strains
overall_average_change = sum(average_changes) / len(average_changes)
print(f"Overall average change in THCA from T1 to T3: {overall_average_change:.2f}%")

change = pd.Series(average_changes)
change.hist()
plt.title('Average Change in THCA from Stability Test T1 to T3')
plt.xlabel('Percent Difference (%)')
plt.ylabel('Count')
plt.show()
change.describe()
