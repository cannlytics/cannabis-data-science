"""
International Cannabis Varieties
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 5/15/2023
Updated: 5/16/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# Standard imports:
import ast

# External imports:
from cannlytics.utils import convert_to_numeric
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (6, 4),
    'font.family': 'Times New Roman',
    'font.size': 18,
})


#-----------------------------------------------------------------------
# Get the data.
#-----------------------------------------------------------------------

METADATA = {
    'ca': {
        'columns': {
            'sample_id': 'lab_result_id',
        },
    },
    'ct': {
        'columns': {
            'id': 'lab_result_id',
            'created_at': 'created_at',
            'brand_name': 'product_name',
            'dosage_form': 'product_type',
            'producer': 'producer',
            'tetrahydrocannabinol_thc': 'thc',
            'tetrahydrocannabinol_acid_thca': 'thca',
            'cannabidiols_cbd': 'cbd',
            'cannabidiol_acid_cbda': 'cbda',
            'limonene': 'd_limonene',
            'b_pinene': 'beta_pinene',
            'b_caryophyllene': 'beta_caryophyllene',
            'b_myrcene': 'beta_myrcene',
            'humulene_hum': 'alpha_humulene',
        }
    },
    'ma': {
        'columns': {
            'date_tested': 'created_at',
        },
    },
    'mi': {
        'columns': {
            'sample_id': 'lab_result_id',
            'date_tested': 'created_at',
        },
    },
    'wa': {
        'columns': {
            'strain_name': 'product_name',
            'inventory_type': 'product_type',
            'lab_result_id': 'lab_result_id',
            'created_date': 'created_at',
            'delta_9_thc': 'thc',
            'thca': 'thca',
            'total_thc': 'total_thc',
            'cbd': 'cbd',
            'cbda': 'cbda',
            'total_cbd': 'total_cbd',
        },
    },
}

# Read CT lab results.
datafile = 'D://data/connecticut/lab_results/ct-results-2023-05-16.xlsx'
ct_results = pd.read_excel(datafile)
ct_results.rename(columns=METADATA['ct']['columns'], inplace=True)
print('CT results:', ct_results.shape)

# Read MA lab results.
datafile = 'D://data/massachusetts/lab_results/mcr-lab-results-2023-05-17.xlsx'
ma_results = pd.read_excel(datafile)
print('MA results:', ma_results.shape)

# Read WA lab results.
datafile = 'D://data/washington/lab_results/wa-lab-results-2023-04-04.xlsx'
wa_results = pd.read_excel(datafile)
wa_results.rename(columns=METADATA['wa']['columns'], inplace=True)
print('WA results:', wa_results.shape)

# Optional: Add CA lab results.
datafile = 'D://data/california/lab_results/sc-labs-lab-results-2022-07-13.xlsx'
ca_results = pd.read_excel(datafile, index_col=0)
ca_results.rename(columns=METADATA['ca']['columns'], inplace=True)
print('CA results:', ca_results.shape)

# Optional: Add MI lab results.
datafile = 'D://data/michigan/lab_results/psi-lab-results-2022-07-12.xlsx'
mi_results = pd.read_excel(datafile, index_col=0)
mi_results.rename(columns=METADATA['mi']['columns'], inplace=True)
mi_results.set_index('lab_result_id', inplace=True)
print('MI results:', mi_results.shape)



#-----------------------------------------------------------------------
# Standardize the data.
#-----------------------------------------------------------------------

def get_analyte_value(results, analyte, key='key', value='value'):
    """Get the value for an analyte from a list of standardized results."""
    for obs in ast.literal_eval(results):
        if obs[key] == analyte:
            try:
                return convert_to_numeric(obs[value], strip=True)
            except:
                return obs[value]


# === Standardize MA results ===

# Get THC and CBD values for MA.
ma_results['thc'] = ma_results['results'].apply(lambda x: get_analyte_value(x, 'thc'))
ma_results['thca'] = ma_results['results'].apply(lambda x: get_analyte_value(x, 'thca'))
ma_results['cbd'] = ma_results['results'].apply(lambda x: get_analyte_value(x, 'cbd'))
ma_results['cbda'] = ma_results['results'].apply(lambda x: get_analyte_value(x, 'cbda'))

# Calculate total THC and CBD for MA.
ma_results['total_thc'] = ma_results['thc'] + ma_results['thca'] * 0.877
ma_results['total_cbd'] = ma_results['cbd'] + ma_results['cbda'] * 0.877

# Get terpene values for MA.
ma_results['beta_caryophyllene'] = ma_results['results'].apply(
    lambda x: get_analyte_value(x, 'beta_caryophyllene')
)
ma_results['alpha_humulene'] = ma_results['results'].apply(
    lambda x: get_analyte_value(x, 'alpha_humulene')
)
ma_results['beta_myrcene'] = ma_results['results'].apply(
    lambda x: get_analyte_value(x, 'beta_myrcene')
)
ma_results['caryophyllene_oxide'] = ma_results['results'].apply(
    lambda x: get_analyte_value(x, 'caryophyllene_oxide')
)
ma_results['beta_pinene'] = ma_results['results'].apply(
    lambda x: get_analyte_value(x, 'beta_pinene')
)
ma_results['d_limonene'] = ma_results['results'].apply(
    lambda x: get_analyte_value(x, 'd_limonene')
)
ma_results['terpinolene'] = ma_results['results'].apply(
    lambda x: get_analyte_value(x, 'terpinolene')
)


# === Standardize CT results ===

# Standardize CT results.
analytes = ['thc', 'thca', 'cbd', 'cbda']
for analyte in analytes:
    ct_results[analyte] = pd.to_numeric(ct_results[analyte], errors='coerce')

# Calculate total THC and CBD for MA.
ct_results['total_thc'] = ct_results['thc'] + ct_results['thca'] * 0.877
ct_results['total_cbd'] = ct_results['cbd'] + ct_results['cbda'] * 0.877

# Standardize terpene data for CT.
ct_results['d_limonene'] = ct_results['d_limonene'].apply(
    lambda x: convert_to_numeric(str(x), strip=True)
)
ct_results['beta_pinene'] = ct_results['beta_pinene'].apply(
    lambda x: convert_to_numeric(str(x), strip=True)
)
ct_results['beta_caryophyllene'] = ct_results['beta_caryophyllene'].apply(
    lambda x: convert_to_numeric(str(x), strip=True)
)
ct_results['beta_myrcene'] = ct_results['beta_myrcene'].apply(
    lambda x: convert_to_numeric(str(x), strip=True)
)
ct_results['alpha_humulene'] = ct_results['alpha_humulene'].apply(
    lambda x: convert_to_numeric(str(x), strip=True)
)

# === Standardize CA results ===

# Standardize total THC and CBD.
ca_results['total_thc'] = ca_results['total_thc'].apply(lambda x: convert_to_numeric(str(x), strip=True))
ca_results['total_cbd'] = ca_results['total_cbd'].apply(lambda x: convert_to_numeric(str(x), strip=True))

# Hot-Fix:
ca_results['total_thc'] = pd.to_numeric(ca_results['total_thc'], errors='coerce')
ca_results['total_cbd'] = pd.to_numeric(ca_results['total_cbd'], errors='coerce')

# FIXME: Get terpene values for CA.
# ca_results['beta_caryophyllene'] = ca_results['results'].apply(
#     lambda x: get_analyte_value(x, 'beta_caryophyllene', key='compound', value='result-percent')
# )
# ca_results['alpha_humulene'] = ca_results['results'].apply(
#     lambda x: get_analyte_value(x, 'alpha_humulene', key='compound', value='result-percent')
# )
# ca_results['beta_myrcene'] = ca_results['results'].apply(
#     lambda x: get_analyte_value(x, 'beta_myrcene', key='compound', value='result-percent')
# )
# ca_results['caryophyllene_oxide'] = ca_results['results'].apply(
#     lambda x: get_analyte_value(x, 'caryophyllene_oxide', key='compound', value='result-percent')
# )
# ca_results['beta_pinene'] = ca_results['results'].apply(
#     lambda x: get_analyte_value(x, 'beta_pinene', key='compound', value='result-percent')
# )
# ca_results['d_limonene'] = ca_results['results'].apply(
#     lambda x: get_analyte_value(x, 'd_limonene', key='compound', value='result-percent')
# )
# ca_results['terpinolene'] = ca_results['results'].apply(
#     lambda x: get_analyte_value(x, 'terpinolene', key='compound', value='result-percent')
# )

# === Standardize MI results ===

# Get THC and CBD values for MA.
mi_results['thc'] = mi_results['results'].apply(lambda x: get_analyte_value(x, 'Δ9-THC', key='name'))
mi_results['thca'] = mi_results['results'].apply(lambda x: get_analyte_value(x, 'THCa', key='name'))
mi_results['cbd'] = mi_results['results'].apply(lambda x: get_analyte_value(x, 'CBD', key='name'))
mi_results['cbda'] = mi_results['results'].apply(lambda x: get_analyte_value(x, 'CBDa', key='name'))

# Calculate total THC and CBD for MA.
mi_results['total_thc'] = mi_results['thc'] + mi_results['thca'] * 0.877
mi_results['total_cbd'] = mi_results['cbd'] + mi_results['cbda'] * 0.877

# FIXME: Get terpene values for MI.
# mi_results['beta_caryophyllene'] = mi_results['results'].apply(
#     lambda x: get_analyte_value(x, 'beta_caryophyllene')
# )
# mi_results['alpha_humulene'] = mi_results['results'].apply(
#     lambda x: get_analyte_value(x, 'alpha_humulene')
# )
# mi_results['beta_myrcene'] = mi_results['results'].apply(
#     lambda x: get_analyte_value(x, 'beta_myrcene')
# )
# mi_results['caryophyllene_oxide'] = mi_results['results'].apply(
#     lambda x: get_analyte_value(x, 'caryophyllene_oxide')
# )
# mi_results['beta_pinene'] = mi_results['results'].apply(
#     lambda x: get_analyte_value(x, 'beta_pinene')
# )
# mi_results['d_limonene'] = mi_results['results'].apply(
#     lambda x: get_analyte_value(x, 'd_limonene')
# )
# mi_results['terpinolene'] = mi_results['results'].apply(
#     lambda x: get_analyte_value(x, 'terpinolene')
# )


# === Aggregate the results ===

# Assign state IDs.
ca_results['state'] = 'ca'
ct_results['state'] = 'ct'
wa_results['state'] = 'wa'
ma_results['state'] = 'ma'
mi_results['state'] = 'mi'

# Re-index results.
ca_results.set_index('lab_result_id', inplace=True)
ct_results.set_index('lab_result_id', inplace=True)
# mi_results.set_index('lab_result_id', inplace=True)
wa_results.set_index('lab_result_id', inplace=True)

# Hot-fix: Drop duplicate columns.
wa_results = wa_results.loc[:,~wa_results.columns.duplicated()].copy()

# Aggregate results.
aggregate = [
    ca_results,
    ct_results,
    ma_results,
    mi_results,
    wa_results,
]
all_results = pd.concat(aggregate, ignore_index=True)
print('Aggregated %i lab results.' % len(all_results))


#-----------------------------------------------------------------------
# Sample the data.
#-----------------------------------------------------------------------

# Hot-fix: Ensure there are no NaN values.
all_results['product_name'] = all_results['product_name'].fillna('Unknown')
all_results['product_type'] = all_results['product_type'].fillna('Unknown')

# Get only results with a valid product type.
flower_results = all_results.loc[~all_results['product_type'].isna()]

# Get only flower results.
flower_results = flower_results.loc[
    (all_results['product_type'].str.contains('flower', case=False)) |
    (all_results['product_type'].str.contains('Usable Cannabis', case=False))
]

# Identify international varieties.
thai_results = flower_results.loc[
    flower_results['product_name'].str.contains('thai', case=False)
]
colombian_results = flower_results.loc[
    flower_results['product_name'].str.contains('colombian', case=False) |
    flower_results['product_name'].str.contains('medellin', case=False) |
    flower_results['product_name'].str.contains('santa marta', case=False)
]
durban_results = flower_results.loc[
    flower_results['product_name'].str.contains('durban', case=False)
]

# Remove outliers.
thai_results = thai_results.loc[(thai_results['total_thc'] > 0) & (thai_results['total_thc'] < 100) & (thai_results['total_cbd'] < 7)]
colombian_results = colombian_results.loc[(colombian_results['total_thc'] > 0) & (colombian_results['total_thc'] < 100) & (colombian_results['total_cbd'] < 7)]
durban_results = durban_results.loc[(durban_results['total_thc'] > 0) & (durban_results['total_thc'] < 100) & (durban_results['total_cbd'] < 7)]

# Future work: Assign dummy variables for "Thai", "Durban", "Colombian", "Other".


#-----------------------------------------------------------------------
# Visualize the data.
#-----------------------------------------------------------------------

# Look at the data.
thai_results[['state', 'product_name', 'total_thc']]
colombian_results[['state', 'product_name', 'total_thc']]
durban_results[['state', 'product_name', 'total_thc']]

# View frequency of each variety.
counts = [len(thai_results), len(colombian_results), len(durban_results)]
varieties = ['Thai', 'Colombian', 'Durban']
plt.bar(varieties, counts)
plt.xlabel('Variety')
plt.ylabel('Observations')
plt.title('Number of Lab Results of International Varieties')
plt.show()

# Visualize THC and CBD by variety.
sns.scatterplot(data=thai_results, x='total_cbd', y='total_thc')
sns.scatterplot(data=colombian_results, x='total_cbd', y='total_thc')
sns.scatterplot(data=durban_results, x='total_cbd', y='total_thc')
plt.legend(['Thai', 'Colombian', 'Durban'])
plt.show()


#-----------------------------------------------------------------------
# Analyze the data.
#-----------------------------------------------------------------------

# Calculate average THC by variety.
avg_thc_thai = thai_results['total_thc'].mean()
avg_thc_colombian = colombian_results['total_thc'].mean()
avg_thc_durban = durban_results['total_thc'].mean()
print('Average THC percent (Thai):', round(avg_thc_thai, 2))
print('Average THC percent (Colombian):', round(avg_thc_colombian, 2))
print('Average THC percent (Durban):', round(avg_thc_durban, 2))

# TODO: Use MANOVA to test if there is a significant difference in
# THC and CBD by variety.


# TODO: Use a probit model to see if THC and CBD are significant
# predictors of variety.



#-----------------------------------------------------------------------
# Draw insights from the data.
#-----------------------------------------------------------------------

# Calculate THC to CBD ratio for each variety.
thai_results['ratio'] = thai_results['total_thc'].astype(float).div(thai_results['total_cbd'].astype(float)).replace(np.inf, 0)
colombian_results['ratio'] = colombian_results['total_thc'].astype(float).div(colombian_results['total_cbd'].astype(float)).replace(np.inf, 0)
durban_results['ratio'] = durban_results['total_thc'].astype(float).div(durban_results['total_cbd'].astype(float)).replace(np.inf, 0)

# Look at top ratio for Thai.
thai_results.sort_values(by='ratio', ascending=False, inplace=True)
print(thai_results[['state', 'product_name', 'ratio']])
sns.barplot(data=thai_results[:10], x='product_name', y='ratio')
plt.xlabel('Product Name')
plt.ylabel('Ratio')
plt.title('THC to CBD Ratio of Durban Results')
plt.xticks(rotation=45)
plt.show()

# Look at top ratio for Colombian.
colombian_results.sort_values(by='ratio', ascending=False, inplace=True)
print(colombian_results[['state', 'product_name', 'ratio']])
sns.barplot(data=colombian_results[:10], x='product_name', y='ratio')
plt.xlabel('Product Name')
plt.ylabel('Ratio')
plt.title('THC to CBD Ratio of Colombian Results')
plt.xticks(rotation=45)
plt.show()

# Look at top ratio for Durban.
durban_results.sort_values(by='ratio', ascending=False, inplace=True)
print(durban_results[['state', 'product_name', 'ratio']])
sns.barplot(data=durban_results[:10], x='product_name', y='ratio')
plt.xlabel('Product Name')
plt.ylabel('Ratio')
plt.title('THC to CBD Ratio of Durban Results')
plt.xticks(rotation=45)
plt.show()


#-----------------------------------------------------------------------
# Expand analysis to terpenes.
#-----------------------------------------------------------------------


def sample_and_plot(results, x, y, label='product_name'):
    """Sample results with valid values and plot."""
    sample = results.loc[(results[x].notna()) & (results[y].notna())]
    sample = sample.replace('', np.nan).dropna(subset=[x, y])
    sample['ratio'] = sample[y].div(sample[x]).replace(np.inf, 0)
    sns.scatterplot(
        x=x,
        y=y,
        data=sample,
        hue='ratio',
        size='ratio',
        sizes=(150, 400),
        legend='full',
        palette='viridis',
    )
    for line in range(0, sample.shape[0]):
        plt.text(
            sample[x].iloc[line],
            sample[y].iloc[line],
            sample[label].iloc[line],
            horizontalalignment='left',
            size='medium',
        )
    plt.xlim(0)
    plt.ylim(0)
    plt.legend([],[], frameon=False)
    plt.show()


# === `beta_caryophyllene` to `alpha_humulene` ratio ===

# Durban.
sample_and_plot(durban_results, 'beta_caryophyllene', 'alpha_humulene')

# Colombian.
sample_and_plot(colombian_results, 'beta_caryophyllene', 'alpha_humulene')


# === `beta_myrcene` to `d_limonene` ratio ===

# Durban.
sample_and_plot(durban_results, 'd_limonene', 'beta_myrcene')


# Colombian.
sample_and_plot(colombian_results, 'd_limonene', 'beta_myrcene')


# === `beta_pinene` to `d_limonene` ratio ===

# Durban.
sample_and_plot(durban_results, 'd_limonene', 'beta_pinene')


# Colombian.
sample_and_plot(colombian_results, 'd_limonene', 'beta_pinene')


# Calculate ratios.
durban_r = durban_results['beta_pinene'].astype(float).div(durban_results['d_limonene'].astype(float)).replace(np.inf, 0)
durban_r.sort_values(ascending=False, inplace=True)
print('Average Durban beta-pinene to d-limonene ratio:', round(durban_r.mean(), 2))

colombian_r = colombian_results['beta_pinene'].astype(float).div(colombian_results['d_limonene'].astype(float)).replace(np.inf, 0)
colombian_r.sort_values(ascending=False, inplace=True)
print('Average Colombian beta-pinene to d-limonene ratio:', round(colombian_r.mean(), 2))


#-----------------------------------------------------------------------
# Future work: Expand analysis to landraces.
#-----------------------------------------------------------------------

# === Visualize the landrace terpene profile. ===

# Identify the terpene profile of Landrace Durban.
landrace = durban_results.loc[
    durban_results['product_name'].str.contains('landrace', case=False)
]
landrace_results = ast.literal_eval(landrace.iloc[0]['results'])
landrace_terpenes = [x for x in landrace_results if x['analysis'] == 'terpenes']

# Compile the terpene data into a DataFrame.
landrace_data = pd.DataFrame(landrace_terpenes)
landrace_data = landrace_data[landrace_data['value'].apply(lambda x: isinstance(x, (int, float)))]
landrace_data['value'] = pd.to_numeric(landrace_data['value'])

# Plot the terpene concentrations.
landrace_data.sort_values(by='key', ascending=False, inplace=True)
plt.bar(landrace_data['key'], landrace_data['value'])
plt.xlabel('Terpene')
plt.ylabel('Concentration')
plt.title('Landrace Durban Terpene Concentrations')
plt.xticks(rotation=90)
plt.show()

# Plot by relative concentration.
total = landrace_data['value'].sum()
landrace_data['relative_concentration'] = landrace_data['value'] / total
landrace_data.sort_values(by='key', ascending=False, inplace=True)
plt.bar(landrace_data['key'], landrace_data['relative_concentration'])
plt.xlabel('Terpene')
plt.ylabel('Relative Concentration')
plt.title('Landrace Durban Terpene Concentrations (Relative)')
plt.xticks(rotation=90)
plt.show()


# === Visualize the Colombian Prophet terpene profile. ===

# Identify the terpene profile of Colombian Prophet.
prophet = colombian_results.loc[
    colombian_results['product_name'].str.contains('Colombian Prophet', case=False)
]
prophet_results = ast.literal_eval(prophet.iloc[0]['results'])
prophet_terpenes = [x for x in prophet_results if x['analysis'] == 'terpenes']

# Compile the terpene data into a DataFrame.
prophet_data = pd.DataFrame(prophet_terpenes)

# Plot the terpene concentrations.
prophet_data = prophet_data[prophet_data['value'].apply(lambda x: isinstance(x, (int, float)))]
prophet_data['value'] = pd.to_numeric(prophet_data['value'])
prophet_data.sort_values(by='key', ascending=False, inplace=True)
plt.bar(prophet_data['key'], prophet_data['value'])
plt.xlabel('Terpene')
plt.ylabel('Concentration')
plt.title('Colombian Prophet Terpene Concentrations')
plt.xticks(rotation=90)
plt.show()

# Plot by relative concentration.
total = prophet_data['value'].sum()
prophet_data['relative_concentration'] = prophet_data['value'] / total
prophet_data.sort_values(by='key', ascending=False, inplace=True)
plt.bar(prophet_data['key'], prophet_data['relative_concentration'])
plt.xlabel('Terpene')
plt.ylabel('Relative Concentration')
plt.title('Colombian Prophet Terpene Concentrations (Relative)')
plt.xticks(rotation=90)
plt.show()
