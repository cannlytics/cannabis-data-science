"""
Cannabis Strain Analysis | Cannabis Data Science Meetup
Copyright (c) 2021 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 8/26/2021
Updated: 12/29/2021
License: MIT License <https://opensource.org/licenses/MIT>

Description:

    Analyze some of the topics discussed at the Cannabis Conference (2021).

Data sources:

    - Connecticut Medical Marijuana Brand Registry: https://data.ct.gov/Health-and-Human-Services/Medical-Marijuana-Brand-Registry/egd5-wb6r/data
    - Connecticut Socrata Open Data API: https://dev.socrata.com/foundry/data.ct.gov/egd5-wb6r
    - WA Traceability Data 2018 - Nov. 2021: https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd?page=1

Resources:    

    - WA Traceability Data Guide: https://lcb.wa.gov/sites/default/files/publications/Marijuana/traceability/WALeafDataSystems_UserManual_v1.37.5_AddendumC_LicenseeUser.pdf

"""
# External imports.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sodapy import Socrata

#-----------------------------------------------------------------------------
# Step 1. Get the data.
#-----------------------------------------------------------------------------

# Define lab datasets.
lab_datasets = ['LabResults_0', 'LabResults_1', 'LabResults_2']

# Read in the lab result data.
shards = []
for dataset in lab_datasets:
    lab_data = pd.read_csv(
        f'../.datasets/{dataset}.csv',
        sep='\t',
        encoding='utf-16',
        # nrows=1000,
    )
    shards.append(lab_data)

# Aggregate lab data.
data = pd.concat(shards)
data = data.sort_index()
data.index = data['global_id']

# Add a time column.
data['date'] = pd.to_datetime(data['created_at'])

# Remove values with nonsensical THCA.
data = data[data['cannabinoid_d9_thca_percent'] <= 100]

# Calculate total cannabinoids.
cannabinoids_wa = [
    'cannabinoid_d9_thca_percent',
    'cannabinoid_d9_thc_percent',
    'cannabinoid_d8_thc_percent',
    'cannabinoid_thcv_percent',
    'cannabinoid_cbd_percent',
    'cannabinoid_cbda_percent',
    'cannabinoid_cbdv_percent',
    'cannabinoid_cbg_percent',
    'cannabinoid_cbga_percent',
    'cannabinoid_cbc_percent',
    'cannabinoid_cbn_percent',
]
data['total_cannabinoids'] = data[cannabinoids_wa].sum(axis=1)

# Separate 2020 and 2021 data.
data_2020 = data.loc[(data['date'] >= pd.to_datetime('2020-01-01')) &
                     (data['date'] < pd.to_datetime('2021-01-01'))]

data_2021 = data.loc[(data['date'] >= pd.to_datetime('2021-01-01')) &
                     (data['date'] < pd.to_datetime('2022-01-01'))]

# Identify flower and concentrate data.
flower_data_2020 = data_2020.loc[data_2020['intermediate_type'] == 'flower_lots']
flower_data_2021 = data_2021.loc[data_2021['intermediate_type'] == 'flower_lots']

concentrates = [
    'hydrocarbon_concentrate',
    'concentrate_for_inhalation',
    'co2_concentrate',
    'non-solvent_based_concentrate',
    'food_grade_solvent_concentrate',
    'ethanol_concentrate',
]
conc_data_2020 = data_2020.loc[data_2020['intermediate_type'].isin(concentrates)]
conc_data_2021 = data_2021.loc[data_2021['intermediate_type'].isin(concentrates)]

#-----------------------------------------------------------------------------
# Step 2. Look at the data.
#-----------------------------------------------------------------------------

# Define format for all plots.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
})

# Compare the distribution of THCA from 2020 to 2021.
bins = 100
cannabinoid = 'cannabinoid_d9_thca_percent'
plt.hist(data_2020[cannabinoid], bins, alpha=0.5, label='2020', density=True)
plt.hist(data_2021[cannabinoid], bins, alpha=0.5, label='2021', density=True)
plt.legend(loc='upper right')
plt.title('Distribution of THCA in WA Cannabis')
plt.show()

# Compare the distribution of THCA in flower from 2020 to 2021.
bins = 100
cannabinoid = 'cannabinoid_d9_thca_percent'
plt.hist(flower_data_2020[cannabinoid], bins, alpha=0.5, label='2020', density=True)
plt.hist(flower_data_2021[cannabinoid], bins, alpha=0.5, label='2021', density=True)
plt.legend(loc='upper right')
plt.title('Distribution of THCA in WA Flower')
plt.show()

# Compare the distribution of THCA in concentrates from 2020 to 2021.
bins = 40
cannabinoid = 'cannabinoid_d9_thca_percent'
plt.hist(conc_data_2020[cannabinoid], bins, alpha=0.5, label='2020')
plt.hist(conc_data_2021[cannabinoid], bins, alpha=0.5, label='2021')
plt.legend(loc='upper right')
plt.title('Distribution of THCA in WA Concentrates')
plt.show()

# Plot the distribution of total cannabinoids in flower.
bins = 100
cannabinoid = 'total_cannabinoids'
plt.hist(flower_data_2020[cannabinoid].loc[flower_data_2020[cannabinoid] <= 100], bins, alpha=0.5, label='2020', density=True)
plt.hist(flower_data_2021[cannabinoid].loc[flower_data_2021[cannabinoid] <= 100], bins, alpha=0.5, label='2021', density=True)
plt.legend(loc='upper right')
plt.title('Distribution of Total Cannabinoids in WA Flower')
plt.show()


#-----------------------------------------------------------------------------
# Bonus: Identify the 4 types of cannabis
# 1. High THC:CBD ratio (Low CBD:THC ratio) (perhaps less than 0.9 CBD:THC)
# 2. Near unitary THC:CBD ratio (perhaps between 0.9 and 1.1).
# 3. High CBD:THC ratio (perhaps greater than 1.1)
# 4. CBG is present as above-average concentrations.
# 5. No major cannabinoids.
#-----------------------------------------------------------------------------

# # Define flower data.
# flower_data = flower_data_2021

# # Calculate THC to CBD ratio.
# flower_data['cbd_to_thc_ratio'] = flower_data.cannabinoid_cbd_percent / \
#                                   flower_data.cannabinoid_d9_thc_percent

# # Drop N/A.
# flower_data['cbd_to_thc_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
# flower_data = flower_data.loc[flower_data['cbd_to_thc_ratio'].notna()]

# # Drop outliers.
# # upper_bound = thc_to_cbd_ratio_data.thc_to_cbd_ratio.quantile(0.95)
# # thc_to_cbd_ratio_data = thc_to_cbd_ratio_data.loc[thc_to_cbd_ratio_data.thc_to_cbd_ratio < upper_bound]

# high_thc_limit = 0.9
# high_cbd_limit = 1.1

# # Restrict to the bottom 95% quartile.
# # upper_bound = flower_data.cannabinoid_cbd_percent.quantile(0.95)
# # restricted_flower_data = flower_data.loc[flower_data.cannabinoid_cbd_percent < upper_bound]
# # upper_bound = restricted_flower_data.cannabinoid_d9_thc_percent.quantile(0.95)
# # restricted_flower_data = restricted_flower_data.loc[restricted_flower_data.cannabinoid_d9_thc_percent < upper_bound]

# # Identify type 1: High THC:CBD ratio (Low CBD:THC ratio) (perhaps less than 0.9 CBD:THC)
# type_1 = flower_data.loc[flower_data['cbd_to_thc_ratio'] <= high_thc_limit]
# # upper_bound = type_1.cannabinoid_cbd_percent.quantile(0.95)
# # type_1_data = type_1.loc[type_1.cannabinoid_cbd_percent < upper_bound]
# # upper_bound = type_1_data.cannabinoid_d9_thc_percent.quantile(0.95)
# # type_1_data = type_1_data.loc[type_1_data.cannabinoid_d9_thc_percent < upper_bound]

# # Identify type 2: Near unitary THC:CBD ratio (perhaps between 0.9 and 1.1).
# type_2 = flower_data.loc[
#     (flower_data['cbd_to_thc_ratio'] > high_thc_limit) &\
#     (flower_data['cbd_to_thc_ratio'] < high_cbd_limit)
# ]
# # upper_bound = type_1.cannabinoid_cbd_percent.quantile(0.95)
# # type_2_data = type_2.loc[type_2.cannabinoid_cbd_percent < upper_bound]
# # upper_bound = type_2_data.cannabinoid_d9_thc_percent.quantile(0.95)
# # type_2_data = type_2_data.loc[type_2_data.cannabinoid_d9_thc_percent < upper_bound]

# # Identify type 3: High CBD:THC ratio (perhaps greater than 1.1)
# type_3 = flower_data.loc[flower_data['cbd_to_thc_ratio'] >= high_cbd_limit]
# # upper_bound = type_1.cannabinoid_cbd_percent.quantile(0.95)
# # type_3_data = type_3.loc[type_3.cannabinoid_cbd_percent < upper_bound]
# # upper_bound = type_3_data.cannabinoid_d9_thc_percent.quantile(0.95)
# # type_3_data = type_3_data.loc[type_3_data.cannabinoid_d9_thc_percent < upper_bound]

# # Identify type 4: CBG is present as above-average concentrations.
# cbg_present = flower_data.loc[flower_data.cannabinoid_cbg_percent > 0]
# print('Number of flower samples with CBG: %i' % len(cbg_present))

# # Calculate the percent of samples that test for CBG.
# percent_of_flower_with_cbg = len(cbg_present) / len(flower_data) * 100
# print('Percent of flower samples with CBG: %.2f' % percent_of_flower_with_cbg)

# # Exclude outliers?
# # upper_bound = cbg_present.cannabinoid_cbg_percent.quantile(0.95)
# # cbg_data = cbg_present.loc[cbg_present.cannabinoid_cbg_percent < upper_bound]
# type_4_criterion = cbg_present['cannabinoid_cbg_percent'] >= cbg_present['cannabinoid_cbg_percent'].mean()
# type_4 = cbg_present.loc[type_4_criterion]
# # upper_bound = type_4.cannabinoid_cbd_percent.quantile(0.95)
# # type_4_data = type_4.loc[type_4.cannabinoid_cbd_percent < upper_bound]
# # upper_bound = type_4_data.cannabinoid_d9_thc_percent.quantile(0.95)
# # type_4_data = type_4_data.loc[type_4_data.cannabinoid_d9_thc_percent < upper_bound]

# # upper_bound = type_4.cannabinoid_cbg_percent.quantile(0.95)
# # type_4_data = type_4.loc[type_4.cannabinoid_cbg_percent < upper_bound]
# # type_4_data.cannabinoid_cbg_percent.hist()

# # TODO: Plot the 4 types on the same scatterplot
# # fig, ax = plt.subplots(figsize=(6, 4))
# # ax.scatter(restricted_flower_data.cannabinoid_d9_thc_percent, restricted_flower_data.cannabinoid_cbd_percent)
# # ax.set_ylabel('CBD', fontsize=12)
# # ax.set_xlabel('Delta-9 THC', fontsize=12)
# # ax.set_title(f'CBD to THC in Flower in Washington State', fontsize=14)
# # plt.show()

# # Plot the 4 types
# fig, ax = plt.subplots(figsize=(6, 4))
# ax.scatter(type_1.cannabinoid_d9_thc_percent, type_1.cannabinoid_cbd_percent, label='Type 1')
# ax.scatter(type_2.cannabinoid_d9_thc_percent, type_2.cannabinoid_cbd_percent, label='Type 2')
# ax.scatter(type_3.cannabinoid_d9_thc_percent, type_3.cannabinoid_cbd_percent, label='Type 3')
# # ax.scatter(type_4_data.cannabinoid_d9_thc_percent, type_4_data.cannabinoid_cbd_percent, label='Type 4')
# ax.grid()
# ax.set_ylabel('CBD')
# ax.set_xlabel('Delta-9 THC')
# ax.set_title('Types of Cannabis Flower in Washington State')
# plt.legend(loc='upper right')
# plt.show()


#-----------------------------------------------------------------------------
# Step 3. Get supplementary data.
#-----------------------------------------------------------------------------

# Get Connecticut cannabis data.
cannabinoids_ct = [
    'tetrahydrocannabinol_thc',
    'tetrahydrocannabinol_acid_thca',
    'cannabidiols_cbd',
    'cannabidiol_acid_cbda',
    'cbg',
    'cbg_a',
    'cannabavarin_cbdv',
    'cannabichromene_cbc',
    'cannbinol_cbn',
    'tetrahydrocannabivarin_thcv',
]

# Get the cannabinoid data.
client = Socrata('data.ct.gov', None)
response = client.get('egd5-wb6r', limit=15000)
ct_data = pd.DataFrame.from_records(response)

# Convert values to floats, coding suspected non-detects as 0.
for analyte in cannabinoids_ct:
    ct_data[analyte] = ct_data[analyte].str.replace('<0.10', '0.0', regex=False)
    ct_data[analyte] = ct_data[analyte].str.replace('<0.1', '0.0', regex=False)
    ct_data[analyte] = ct_data[analyte].str.replace('<0.29', '0.0', regex=False)
    ct_data[analyte] = ct_data[analyte].str.replace('%', '', regex=False)
    ct_data[analyte] = ct_data[analyte].str.replace('-', '0.0', regex=False)
    ct_data[analyte] = pd.to_numeric(ct_data[analyte], errors='coerce').fillna(0.0)

# Calculate total cannabinoids.
ct_data['total_cannabinoids'] = ct_data[cannabinoids_ct].sum(axis=1)

# Add date column.
ct_data['date'] = pd.to_datetime(ct_data['approval_date'])

# Identify flower data in CT.
flower_data_ct = ct_data.loc[ct_data['dosage_form'].str.contains('flower') |
                             ct_data['dosage_form'].str.contains('Flower')]
flower_data_ct = flower_data_ct.loc[flower_data_ct['total_cannabinoids'] <= 40]

# Get Connecticut flower data in 2021.
flower_data_ct_2021 = flower_data_ct.loc[
    (flower_data_ct['date'] >= pd.to_datetime('2021-01-01')) &
    (flower_data_ct['date'] < pd.to_datetime('2022-01-01'))
]


#-----------------------------------------------------------------------------
# Step 4. Look at the supplementary data.
#-----------------------------------------------------------------------------

# Scatterplot of CBDA to THCA.
fig, ax = plt.subplots(figsize=(8, 5), dpi=80)
plt.scatter(
    x='tetrahydrocannabinol_acid_thca',
    y='cannabidiol_acid_cbda',
    s=20,
    color='#007acc',
    alpha=0.6,
    data=flower_data_ct.loc[(flower_data_ct.tetrahydrocannabinol_acid_thca < 100) &
                            (flower_data_ct.cannabidiols_cbd < 100)], 
    
)
plt.gca().set(
    xlim=0,
    ylim=0,
    xlabel='THCA',
    ylabel='CBDA'
)
plt.title('CBDA to THCA in Connecticut Cannabis')   
plt.show()


#-----------------------------------------------------------------------------
# Step 5. Analyze the data with the supplementary data.
#-----------------------------------------------------------------------------

# Define colors for each state.
palette = sns.color_palette('tab10')
ct_color = palette[0]
wa_color = palette[1]

# Plot the CBDA to THCA ratios in Washington and Connecticut.
fig, ax = plt.subplots(figsize=(8, 5), dpi=80)
plt.scatter(
    x='tetrahydrocannabinol_acid_thca',
    y='cannabidiol_acid_cbda',
    s=20,
    color=ct_color,
    alpha=0.6,
    data=flower_data_ct_2021.loc[
        (flower_data_ct_2021.tetrahydrocannabinol_acid_thca < 100) &
        (flower_data_ct_2021.cannabidiols_cbd < 100)
    ], 
    label='CT'
    
)
plt.scatter(
    x='cannabinoid_d9_thca_percent',
    y='cannabinoid_cbda_percent',
    s=20,
    color=wa_color,
    alpha=0.6,
    data=flower_data_2021.loc[
        (flower_data_2021.cannabinoid_d9_thca_percent < 100) &
        (flower_data_2021.cannabinoid_cbda_percent < 100)
    ], 
    label='WA'
    
)
plt.gca().set(
    xlim=0,
    ylim=0,
    xlabel='THCA',
    ylabel='CBDA'
)
plt.legend(loc='upper right')
plt.title('CBDA to THCA in Connecticut Cannabis')   
plt.show()

# Plot the distribution of THCA in CT and WA.
bins = 40
plt.hist(
    flower_data_2021['cannabinoid_d9_thca_percent'].loc[flower_data_2021['cannabinoid_d9_thca_percent'] <= 100],
    bins,
    alpha=0.5,
    label='WA 2021',
    color=wa_color,
    density=True,
)
plt.hist(
    flower_data_ct_2021['tetrahydrocannabinol_acid_thca'].loc[flower_data_ct_2021['tetrahydrocannabinol_acid_thca'] <= 100],
    bins,
    alpha=0.5,
    label='CT 2021',
    color=ct_color,
    density=True,
)
plt.legend(loc='upper right')
plt.title('Distribution of THCA')
plt.show()

# Plot the distribution of total cannabinoids in CT and WA.
bins = 40
cannabinoid = 'total_cannabinoids'
plt.hist(
    flower_data_2021[cannabinoid].loc[flower_data_2021[cannabinoid] <= 100],
    bins,
    alpha=0.5,
    label='WA 2021',
    color=wa_color,
    density=True,
)
plt.hist(
    flower_data_ct_2021[cannabinoid].loc[flower_data_ct_2021[cannabinoid] <= 100],
    bins,
    alpha=0.5,
    label='CT 2021',
    color=ct_color,
    density=True,
)
plt.legend(loc='upper right')
plt.title('Distribution of Total Cannabinoids')
plt.show()

# Calculate moisture-corrected total cannabinoids in WA and re-plot.
flower_data_2021['total_cannabinoids_mc'] = \
    flower_data_2021['total_cannabinoids'] \
    / (1 - flower_data_2021['moisture_content_percent'] * .01)

# Plot the distribution of total cannabinoids in CT and WA.
bins = 40
cannabinoid = 'total_cannabinoids'
plt.hist(
    flower_data_2021['total_cannabinoids_mc'].loc[flower_data_2021['total_cannabinoids_mc'] <= 100],
    bins,
    alpha=0.5,
    label='WA 2021',
    color=wa_color,
    density=True,
)
plt.hist(
    flower_data_ct_2021[cannabinoid].loc[flower_data_ct_2021[cannabinoid] <= 100],
    bins,
    alpha=0.5,
    label='CT 2021',
    color=ct_color,
    density=True,
)
plt.legend(loc='upper right')
plt.title('Distribution of Total Cannabinoids')
plt.show()
