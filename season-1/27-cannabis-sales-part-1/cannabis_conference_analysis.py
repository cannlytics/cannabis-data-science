"""
Cannabis Types | Cannabis Data Science Meetup

Author: Keegan Skeate <keegan@cannlytics.com>
Created: 8/26/2021
Updated: 8/26/2021
License: MIT License

Description:

    Analyze some of the topics discussed at the Cannabis Conference (2021).

Data sources:

    WA Traceability Data 2018- 2020: https://lcb.app.box.com/s/fnku9nr22dhx04f6o646xv6ad6fswfy9?page=1

Resources:    

    WA Traceability Data Guide: https://lcb.wa.gov/sites/default/files/publications/Marijuana/traceability/WALeafDataSystems_UserManual_v1.37.5_AddendumC_LicenseeUser.pdf
"""
import numpy as np
import pandas as pd

# Internal import
from cannabis_type_analysis import cannabinoid_histogram

#-----------------------------------------------------------------------------
# Import the data and match lab results to licensee.
#-----------------------------------------------------------------------------

# Read in the lab result data.
lab_data = None
if lab_data is None:
    lab_data = pd.read_csv(
        f'../.datasets/LabResults_0.csv',
        sep='\t',
        encoding='utf-16',
        # nrows=10000, # FIXME: Read in all the data!
    )
    data = lab_data.sort_index()
    print('Number of observations: %i' % len(data))

#--------------------------------------------------------------------------
# Look at the data.
#--------------------------------------------------------------------------

# Find all of the sample types.
sample_types = list(data.intermediate_type.unique())

# Identify all flower samples.
flower_data = data.loc[(data.intermediate_type == 'flower') |
                       (data.intermediate_type == 'flower_lots')]
print('Number of flower samples: %i' % len(flower_data))

# Identify all flower samples with CBG.
cbg_present = flower_data.loc[flower_data.cannabinoid_cbg_percent > 0]
print('Number of flower samples with CBG: %i' % len(cbg_present))

# Calculate the percent of samples that test for CBG.
percent_of_flower_with_cbg = len(cbg_present) / len(flower_data) * 100
print('Percent of flower samples with CBG: %.2f' % percent_of_flower_with_cbg)

# Exclude outliers?
upper_bound = cbg_present.cannabinoid_cbg_percent.quantile(0.95)
cbg_data = cbg_present.loc[cbg_present.cannabinoid_cbg_percent < upper_bound]

# Plot a histogram of CBG in flower.
cannabinoid_histogram(
    cbg_data['cannabinoid_cbg_percent'],
    title='Distribution of CBG in CBG-Containing Flower in Washington State',
    y_label='Density',
    x_label='CBG Concentration',
    filename='distribution_of_cbg_in_wa_flower.png'
)

#-----------------------------------------------------------------------------
# Identify 4 types of cannabis
# 1. High THC:CBD ratio (Low CBD:THC ratio) (perhaps less than 0.9 CBD:THC)
# 2. Near unitary THC:CBD ratio (perhaps between 0.9 and 1.1).
# 3. High CBD:THC ratio (perhaps greater than 1.1)
# 4. CBG is present as above-average concentrations.
# 5. No major cannabinoids.
#-----------------------------------------------------------------------------

# Calculate THC to CBD ratio.
flower_data['cbd_to_thc_ratio'] = flower_data.cannabinoid_cbd_percent / \
                                  flower_data.cannabinoid_d9_thc_percent

# Drop N/A.
flower_data['cbd_to_thc_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
flower_data = flower_data.loc[flower_data['cbd_to_thc_ratio'].notna()]

# Drop outliers.
# upper_bound = thc_to_cbd_ratio_data.thc_to_cbd_ratio.quantile(0.95)
# thc_to_cbd_ratio_data = thc_to_cbd_ratio_data.loc[thc_to_cbd_ratio_data.thc_to_cbd_ratio < upper_bound]

high_thc_limit = 0.1
high_cbd_limit = 5

# Restrict to the bottom 95% quartile.
upper_bound = flower_data.cannabinoid_cbd_percent.quantile(0.95)
restricted_flower_data = flower_data.loc[flower_data.cannabinoid_cbd_percent < upper_bound]
upper_bound = restricted_flower_data.cannabinoid_d9_thc_percent.quantile(0.95)
restricted_flower_data = restricted_flower_data.loc[restricted_flower_data.cannabinoid_d9_thc_percent < upper_bound]

# Identify type 1.
type_1 = flower_data.loc[flower_data['cbd_to_thc_ratio'] <= high_thc_limit]
upper_bound = type_1.cannabinoid_cbd_percent.quantile(0.95)
type_1_data = type_1.loc[type_1.cannabinoid_cbd_percent < upper_bound]
upper_bound = type_1_data.cannabinoid_d9_thc_percent.quantile(0.95)
type_1_data = type_1_data.loc[type_1_data.cannabinoid_d9_thc_percent < upper_bound]


# Identify type 2.
type_2 = flower_data.loc[
    (flower_data['cbd_to_thc_ratio'] > high_thc_limit) &\
    (flower_data['cbd_to_thc_ratio'] < high_cbd_limit)
]
upper_bound = type_1.cannabinoid_cbd_percent.quantile(0.95)
type_2_data = type_2.loc[type_2.cannabinoid_cbd_percent < upper_bound]
upper_bound = type_2_data.cannabinoid_d9_thc_percent.quantile(0.95)
type_2_data = type_2_data.loc[type_2_data.cannabinoid_d9_thc_percent < upper_bound]

# Identify type 3.
type_3 = flower_data.loc[flower_data['cbd_to_thc_ratio'] >= high_cbd_limit]
upper_bound = type_1.cannabinoid_cbd_percent.quantile(0.95)
type_3_data = type_3.loc[type_3.cannabinoid_cbd_percent < upper_bound]
upper_bound = type_3_data.cannabinoid_d9_thc_percent.quantile(0.95)
type_3_data = type_3_data.loc[type_3_data.cannabinoid_d9_thc_percent < upper_bound]

# Identify type 4.
type_4_criterion = cbg_present['cannabinoid_cbg_percent'] >= cbg_present['cannabinoid_cbg_percent'].mean()
type_4 = cbg_present.loc[type_4_criterion]
upper_bound = type_4.cannabinoid_cbd_percent.quantile(0.95)
type_4_data = type_4.loc[type_4.cannabinoid_cbd_percent < upper_bound]
upper_bound = type_4_data.cannabinoid_d9_thc_percent.quantile(0.95)
type_4_data = type_4_data.loc[type_4_data.cannabinoid_d9_thc_percent < upper_bound]
# upper_bound = type_4.cannabinoid_cbg_percent.quantile(0.95)
# type_4_data = type_4.loc[type_4.cannabinoid_cbg_percent < upper_bound]
# type_4_data.cannabinoid_cbg_percent.hist()

#--------------------------------------------------------------------------
# Visualize at the data.
#--------------------------------------------------------------------------

# TODO: Plot the 4 types on the same scatterplot
import matplotlib.pyplot as plt

# fig, ax = plt.subplots(figsize=(6, 4))
# ax.scatter(restricted_flower_data.cannabinoid_d9_thc_percent, restricted_flower_data.cannabinoid_cbd_percent)
# ax.set_ylabel('CBD', fontsize=12)
# ax.set_xlabel('Delta-9 THC', fontsize=12)
# ax.set_title(f'CBD to THC in Flower in Washington State', fontsize=14)
# plt.show()

# Plot the 4 types
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(type_1_data.cannabinoid_d9_thc_percent, type_1_data.cannabinoid_cbd_percent, label='Type 1')
ax.scatter(type_2_data.cannabinoid_d9_thc_percent, type_2_data.cannabinoid_cbd_percent, label='Type 2')
ax.scatter(type_3_data.cannabinoid_d9_thc_percent, type_3_data.cannabinoid_cbd_percent, label='Type 3')
# ax.scatter(type_4_data.cannabinoid_d9_thc_percent, type_4_data.cannabinoid_cbd_percent, label='Type 4')

ax.grid()
ax.set_ylabel('CBD', fontsize=12)
ax.set_xlabel('Delta-9 THC', fontsize=12)
ax.set_title(f'Types of Cannabis Flower in Washington State', fontsize=14)
plt.legend(loc='upper right')
plt.show()
