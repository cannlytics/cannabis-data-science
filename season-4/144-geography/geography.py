"""
Geography of Hemp Producers
Copyright (c) 2024 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 1/17/2024
Updated: 1/26/2024
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Data Sources:

    - Aggregated Lab Result Data
    - CA: https://cannlytics.page.link/ca-results-2024-01-24 (use lab_state == 'CA')
    - CO: https://cannlytics.page.link/ca-results-2024-01-24 (use lab_state == 'CO')
    - CT: https://cannlytics.page.link/ct-results-2023-12-28
    - FL: https://cannlytics.page.link/fl-results-2024-01-26
    - MA: https://cannlytics.page.link/ma-results-2024-01-03
    - MI: https://cannlytics.page.link/mi-results-2022-07-12

    - Elevation Data API: https://www.opentopodata.org/

    - 2023 USDA Plant Hardiness Zone GIS Datasets
    Source: https://prism.oregonstate.edu/projects/plant_hardiness_zones.php
    Data: https://cannlytics.page.link/phzm-us-zipcode-2023

"""
# Standard imports:
import ast
import calendar
import json
import os
from time import sleep

# External imports:
from cannlytics.data.coas import get_result_value
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import zipcodes
import requests
from datetime import datetime
# Future work: Identify strain name from product name with NLP.
# import spacy
# from textacy.extract import ngrams


# === Setup ===

# Set figure theme.
sns.set_style('whitegrid')


# === Read the data ===

# CA
datafile = 'https://cannlytics.page.link/ca-results-2024-01-24'
ca_results = pd.read_excel(datafile)
ca_results = ca_results.loc[ca_results['lab_state'] == 'CA']
ca_results.drop_duplicates(subset='sample_id', inplace=True)
print('Number of CA results:', len(ca_results))

# CO
datafile = 'https://cannlytics.page.link/ca-results-2024-01-24'
co_results = pd.read_excel(datafile)
co_results = co_results.loc[co_results['lab_state'] == 'CO']
co_results.drop_duplicates(subset='sample_id', inplace=True)
print('Number of CO results:', len(co_results))

# FL
datafile = 'https://cannlytics.page.link/fl-results-2024-01-26'
fl_results = pd.read_csv(datafile)
fl_results.drop_duplicates(subset='sample_id', inplace=True)
fl_results.reset_index(drop=True, inplace=True)
fl_results['producer_zipcode'] = fl_results['producer_zipcode'].apply(
    lambda x: str(x).replace('.0', '')
)
print('Number of FL results:', len(fl_results))

# MA
datafile = 'https://cannlytics.page.link/ma-results-2024-01-03'
ma_results = pd.read_excel(datafile)
ma_results['producer_zipcode'] = None
ma_results.drop_duplicates(subset='sample_id', inplace=True)
print('Number of MA results:', len(ma_results))

# MI
datafile = 'https://cannlytics.page.link/mi-results-2022-07-12'
mi_results = pd.read_excel(datafile)
print('Number of MI results:', len(mi_results))

# CT
datafile = 'https://cannlytics.page.link/ct-results-2023-12-28'
ct_results = pd.read_excel(datafile)
print('Number of CT results:', len(ct_results))


# === Regression Analysis ===

# Future work: Identify strain name from product name.
ca_results['strain_name'] = ca_results['product_name']
co_results['strain_name'] = co_results['product_name']
ma_results['strain_name'] = ma_results['product_name']

# Create a panel sample.
ca_results.reset_index(drop=True, inplace=True)
co_results.reset_index(drop=True, inplace=True)
fl_results.reset_index(drop=True, inplace=True)
ma_results.reset_index(drop=True, inplace=True)
columns = [
    'lab',
    'lab_state',
    'date_tested',
    'producer_zipcode',
    'strain_name',
    'product_type',
    'results',
]
sample = pd.concat([
    ca_results[columns],
    co_results[columns],
    fl_results[columns],
    ma_results[columns],
], ignore_index=True)
print('Total number of lab results:', len(sample))


# === Flower Analysis ===

# DEV: Print out all types of products.
print('All product types:', list(sample.product_type.unique()))

# Restrict the sample to flower types.
flower_types = [
    "CANNABIS (MMTC's) Flower & Plants (Inhalation - Heated)",
    'CBD, Inhalable',
    'Flower',
    'Flower Inhalable',
    'Flower, Colorado Hemp/Flower',
    'Flower, Hemp',
    'Flower, Hemp Flower',
    'Flower, Inhalable',
    'Flower, Inhaled Product',
    'Flower, Medical Inhalable',
    'Flower, Product Inhalable',
    'Hybrid, Inhalable',
    'Indica, Inhalable',
    'Plant (Flower - Cured)',
    'Sativa, Inhalable',
    'flower'
]
sample = sample.loc[sample['product_type'].isin(flower_types)]
print('Number of flower samples:', len(sample))

# Assign year and month variables.
sample['date'] = pd.to_datetime(sample['date_tested'], errors='coerce')
sample['year'] = sample['date'].dt.year
sample['month'] = sample['date'].dt.month


# === Geographic data ===

def get_zipcode_data(zipcode) -> tuple:
    """Get the state for a given zipcode."""
    try:
        zipcode = str(zipcode).replace('.0', '')
        if len(zipcode) == 9:
            zipcode = zipcode[:5] + '-' + zipcode[5:]
        zipcode_info = zipcodes.matching(zipcode)
        if zipcode_info:
            match = zipcode_info[0]
            return (
                match['city'],
                match['county'],
                match['state'],
                match['lat'],
                match['long'],
            )
        else:
            return ()
    except Exception as e:
        print('Error on zipcode:', zipcode)
        return ()


# Add producer zipcode data.
gis_columns = ['producer_city', 'producer_county', 'producer_state', 'producer_latitude', 'producer_longitude']
sample['producer_zipcode'] = sample['producer_zipcode'].astype(str).str.replace('.0', '')
unique_zipcodes = sample['producer_zipcode'].unique()
zipcode_data_map = pd.DataFrame({'producer_zipcode': unique_zipcodes})
zipcode_data_map[gis_columns] = zipcode_data_map['producer_zipcode'].apply(
    lambda x: pd.Series(get_zipcode_data(x))
)
zipcode_data_map = zipcode_data_map.drop_duplicates(subset=['producer_zipcode'])
sample = sample.merge(
    zipcode_data_map,
    on='producer_zipcode',
    how='left',
    validate='m:1'
)


# === Elevation Data ===

def get_elevation(lat, long, dataset='ned10m'):
    """Get elevation data for a given latitude and longitude."""
    response = requests.get(f'https://api.opentopodata.org/v1/{dataset}?locations={lat},{long}')
    if response.status_code == 200:
        result = response.json()
        if result['status'] == 'OK' and result['results']:
            return result['results'][0]['elevation']
    return None


# Read in known elevation data and only lookup known values.
elevation_filename = 'D://data/environment/zipcode-elevations.json'
elevation_url = 'https://cannlytics.page.link/zipcode-elevations'
if os.path.exists(elevation_filename):
    with open(elevation_filename, 'r') as file:
        elevation_data = json.load(file)
else:
    response = requests.get(elevation_url)
    if response.status_code == 200:
        elevation_data = response.json()
    else:
        elevation_data = {}
        print("Failed to fetch elevation data:", response.status_code, response.text)

# Add elevation data.
unique_zipcode_locations = zipcode_data_map[['producer_zipcode', 'producer_latitude', 'producer_longitude']].drop_duplicates()
for _, row in unique_zipcode_locations.iterrows():
    zipcode = row['producer_zipcode']
    if zipcode not in elevation_data:
        lat, long = row['producer_latitude'], row['producer_longitude']
        elevation = get_elevation(lat, long) if pd.notnull(lat) and pd.notnull(long) else None
        elevation_data[row['producer_zipcode']] = elevation
        sleep(0.33)
unique_zipcode_locations['elevation'] = unique_zipcode_locations['producer_zipcode'].map(elevation_data)
sample = sample.merge(
    unique_zipcode_locations[['producer_zipcode', 'elevation']],
    on='producer_zipcode',
    how='left'
)

# DEV: Save elevation data.
# with open(elevation_filename, 'w') as file:
#     json.dump(elevation_data, file)


# === Chemical Data ===

# Dev: Identify all unique cannabinoids and terpenes.
cannabinoids = []
terpenes = []
for item in sample['results']:
    try:
        lab_results = ast.literal_eval(item)
    except:
        try:
            lab_results = json.loads(item)
        except:
            continue
    if lab_results == '[]':
        continue
    if isinstance(lab_results, str):
        lab_results = json.loads(lab_results)
    for result in lab_results:
        analysis = result.get('analysis', None)
        if analysis == 'cannabinoids':
            cannabinoids.append(result['key'])
        elif analysis == 'terpenes':
            terpenes.append(result['key'])
cannabinoids = list(set(cannabinoids))
terpenes = list(set(terpenes))
print('All unique cannabinoids:', cannabinoids)
print('All unique terpenes:', terpenes)

# Commonly observed cannabinoids.
cannabinoids = [
    'cbc',
    'cbca',
    'cbcv',
    'cbd',
    'cbda',
    'cbdv',
    'cbdva',
    'cbg',
    'cbga',
    'cbl',
    'cbla',
    'cbn',
    'cbna',
    'cbt',
    'delta_8_thc',
    'delta_9_thc',
    'thca',
    'thcv',
    'thcva',
]

# Commonly observed terpenes.
terpenes = [
    'alpha_bisabolol',
    'alpha_cedrene',
    'alpha_humulene',
    'alpha_ocimene',
    'alpha_phellandrene',
    'alpha_pinene',
    'alpha_terpinene',
    'alpha_terpineol',
    'beta_caryophyllene',
    'beta_myrcene',
    'beta_ocimene',
    'beta_pinene',
    'borneol',
    'camphene',
    'camphor',
    'caryophyllene_oxide',
    'cedrol',
    'citronellol',
    'd_limonene',
    'delta_3_carene',
    'eucalyptol',
    'farnesol_isomer',
    'fenchol',
    'fenchone',
    'gamma_terpinene',
    'geraniol',
    'geranyl_acetate',
    'guaiol',
    'hexahydrothymol',
    'isoborneol',
    'isopulegol',
    'linalool',
    'menthol',
    'nerol',
    'nerolidol',
    'p_cymene',
    'p_mentha_1_5_diene',
    'phytol',
    'pulegone',
    'sabinene',
    'terpineol',
    'terpinolene',
    'trans_beta_farnesene',
    'trans_nerolidol',
    'trans_beta_ocimene',
    'valencene',
]

# Major terpenes typically analyzed in plant patents.
major_terpenes = [
    'alpha_pinene',
    'camphene',
    'beta_pinene',
    'beta_myrcene',
    'd_limonene',
    'linalool',
    'beta_caryophyllene',
    'alpha_humulene',
    'nerolidol',
    # 'guaiol', # Not observed often in the results.
    'alpha_bisabolol',
]


def clean_terpene_data(df, terpene_columns):
    for col in terpene_columns:
        df[col].replace('', pd.NA, inplace=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# Get the results for each cannabinoid and terpene.
for analyte in cannabinoids + terpenes:
    print('Augmenting:', analyte)
    sample[analyte] = sample['results'].apply(
        lambda x: get_result_value(x, analyte, key='key')
    )

# Clean the terpene data.
sample = clean_terpene_data(sample, cannabinoids + terpenes)

# Calculate total cannabinoids.
sample['total_cannabinoids'] = sample[cannabinoids].sum(axis=1)

# Calculate total terpenes.
sample['total_terpenes'] = sample[terpenes].sum(axis=1)

# Calculate total THC.
sample['total_thc'] = sample['delta_9_thc'] + sample['thca'] * 0.877

# Calculate total CBD.
sample['total_cbd'] = sample['cbd'] + sample['cbda'] * 0.877


# DEV: Save the aggregate data at this stage.
# timestamp = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
# outfile = f"D://data/results/all-lab-results-{timestamp}.csv"
# sample.to_csv(outfile, index=False)


# === Strain name identification ===

# Rename 'Gorilla Glue' and 'GG #4' to 'GG4',
# “SLH” = “superlemon-haze,” and “GDP” = “granddaddy-purple,”
for df in [ca_results, co_results, fl_results, ma_results]:
    strain_column = 'strain_name' if 'strain_name' in df.columns else 'product_name'
    df[strain_column] = df[strain_column].replace(['Gorilla Glue', 'GG #4'], 'GG4')
    df[strain_column] = df[strain_column].replace(['SLH'], 'Super Lemon Haze')
    df[strain_column] = df[strain_column].replace(['GDP'], 'Granddaddy Purple')

# Future work: Identify most popular strains, common across states using n-grams.

# # Initialize the spacy model.
# nlp = spacy.load('en_core_web_lg')

# # Compile all of the product names into a single corpus.
# strain_names = list(sample['strain_name'].dropna().unique())
# corpus = '. '.join([str(x) for x in strain_names])
# corpus = corpus.replace('_', ' ')
# doc = nlp(corpus)

# # Identify unique unigrams, bi-grams, trigrams to use as strain names.
# unigrams = list(set([x.text for x in ngrams(doc, 1, min_freq=1)]))
# bigrams = list(set([x.text for x in ngrams(doc, 2, min_freq=1)]))
# trigrams = list(set([x.text for x in ngrams(doc, 3, min_freq=1)]))
# print('Unique unigrams:', len(unigrams))
# print('Unique bigrams:', len(bigrams))
# print('Unique trigrams:', len(trigrams))

# Find strains that are common across states.
unique_ca_strains = set(ca_results['strain_name'].dropna().unique())
unique_co_strains = set(co_results['strain_name'].dropna().unique())
unique_fl_strains = set(fl_results['strain_name'].dropna().unique())
unique_ma_strains = set(ma_results['strain_name'].dropna().unique())
common_strains = unique_ca_strains.intersection(unique_co_strains, unique_fl_strains, unique_ma_strains)
common_strains_list = list(common_strains)
print("Common strains across CA, CO, FL, and MA:")
print(common_strains_list)

# Define common strains.
common_strains = [
    'Blue Dream',
    'Skywalker OG',
    'OG Kush',
    'Strawberry Cough',
    'GMO',
    'Wedding Cake',
    'Super Lemon Haze'
    'Mimosa',
    'Runtz',
    'Dolato',
    'Zkittles',
    'White Truffle',
    'GG4',
    'Gelato',
]


def match_common_strain(strain_name, common_strains):
    """Match a product name containing a strain name to a common strain."""
    for common_strain in common_strains:
        if common_strain in str(strain_name):
            return common_strain
    return strain_name

# Filter the DataFrame for rows where 'strain_name' matches a common strain
sample['matched_strain'] = sample['strain_name'].apply(lambda x: match_common_strain(x, common_strains))
strains = sample[sample['matched_strain'].isin(common_strains)]
strains['strain_name'] = strains['matched_strain']


# === Visualize the chemical data ===

# Scatterplot of `beta_pinene` to `d_limonene` by strain.
filtered_strains = strains.sort_values('strain_name')
plt.figure(figsize=(15, 8))
sns.scatterplot(
    data=filtered_strains,
    x='d_limonene',
    y='beta_pinene',
    hue='strain_name',
    palette='tab20',
    legend='full',
    s=200,
)
plt.title('Scatterplot of Beta-Pinene vs D-Limonene', fontsize=28)
plt.ylabel('Beta-Pinene', fontsize=21)
plt.xlabel('D-Limonene', fontsize=21)
plt.tick_params(axis='x', labelsize=18)
plt.tick_params(axis='y', labelsize=18)
legend = plt.legend(
    title='Strain',
    bbox_to_anchor=(1.05, 1),
    loc=2,
    borderaxespad=0.0,
    fontsize=18,
)
legend.get_title().set_fontsize('24')
for handle in legend.legendHandles:
    handle.set_sizes([100])
plt.tight_layout()
plt.show()

# Scatterplot of `beta_caryophyllene` to `alpha_humulene` by strain.
x, y = 'beta_caryophyllene', 'alpha_humulene'
filtered_strains = strains.sort_values('strain_name')
plt.figure(figsize=(15, 8))
sns.scatterplot(
    data=filtered_strains,
    x=x,
    y=y,
    hue='strain_name',
    palette='tab20',
    legend='full',
    s=200,
)
y_label = y.replace('_', '-').title()
x_label = x.replace('_', '-').title()
plt.title(f'{y_label} to {x_label}', fontsize=28)
plt.ylabel(y_label, fontsize=21)
plt.xlabel(x_label, fontsize=21)
plt.tick_params(axis='x', labelsize=18)
plt.tick_params(axis='y', labelsize=18)
legend = plt.legend(
    title='Strain',
    bbox_to_anchor=(1.05, 1),
    loc=2,
    borderaxespad=0.0,
    fontsize=18,
)
legend.get_title().set_fontsize('24')
for handle in legend.legendHandles:
    handle.set_sizes([100])
plt.tight_layout()
plt.show()

# Create a pair plot for the major terpenes.
terpenes_data = filtered_strains[major_terpenes + ['strain_name']]
plt.figure(figsize=(24, 18))
sns.pairplot(
    data=terpenes_data,
    vars=major_terpenes,
    hue='strain_name',
    palette='tab20',
    markers='o'
)
plt.show()

# Visualize average terpene content by strain.
average_terpenes_strain = strains.groupby('strain_name')[major_terpenes].mean()
average_terpenes_strain_long = average_terpenes_strain.reset_index().melt(id_vars='strain_name', var_name='Terpene', value_name='Average Content')
plt.figure(figsize=(15, 8))
barplot = sns.barplot(
    data=average_terpenes_strain_long,
    x='strain_name',
    y='Average Content',
    hue='Terpene',
    palette='tab20',
)
plt.title('Average Terpene Content by Strain', fontsize=28)
plt.xlabel('Strain Name', fontsize=21)
plt.ylabel('Average Content', fontsize=21)
plt.xticks(rotation=90, fontsize=18)
plt.yticks(fontsize=18)
plt.legend(title='Terpene', bbox_to_anchor=(1.05, 1), loc=2, fontsize=18)
plt.legend().get_title().set_fontsize('24')
plt.tight_layout()
plt.show()

# Visualize average terpene content by state.
average_terpenes_state = strains.groupby('lab_state')[major_terpenes].mean()
average_terpenes_state_long = average_terpenes_state.reset_index().melt(id_vars='lab_state', var_name='Terpene', value_name='Average Content')
specific_states = average_terpenes_state.loc[['CA', 'MA']]
specific_states_long = specific_states.reset_index().melt(id_vars='lab_state', var_name='Terpene', value_name='Average Content')
plt.figure(figsize=(15, 8))
barplot = sns.barplot(
    data=specific_states_long,
    x='lab_state',
    y='Average Content',
    hue='Terpene',
    palette='tab20',
)
plt.title('Average Terpene Content by State', fontsize=28)
plt.xlabel('State', fontsize=21)
plt.ylabel('Average Content', fontsize=21)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(
    title='Terpene',
    bbox_to_anchor=(1.05, 1),
    loc=2,
    fontsize=18
)
plt.tight_layout()
plt.show()

# Create a timeseries of avg. terpene concentration by month.
filtered_strains = strains.sort_values('strain_name')
filtered_strains['date'] = pd.to_datetime(filtered_strains['date'])
filtered_strains = filtered_strains[filtered_strains['date'].dt.year >= 2021]
monthly_avg = filtered_strains.groupby([filtered_strains['date'].dt.to_period('M')])[major_terpenes].mean()
monthly_avg = monthly_avg.reset_index()
monthly_avg['date'] = monthly_avg['date'].dt.to_timestamp()
monthly_avg['month'] = monthly_avg['date'].dt.month
monthly_avg_melted = monthly_avg.melt(id_vars='date', var_name='Terpene', value_name='Average Concentration')
n_rows = 3
n_cols = int(np.ceil(len(major_terpenes) / float(n_rows)))
figsize = (21, 4 * n_rows)
fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
axes_flat = axes.flatten()
norm = plt.Normalize(1, 12)
cmap = cm.Spectral
for i, terpene in enumerate(major_terpenes):
    ax = axes_flat[i]
    terpene_data = monthly_avg[['date', terpene]].dropna()
    points = np.array([terpene_data['date'].values, terpene_data[terpene].values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(terpene_data['date'].dt.month)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_title(terpene.replace('_', '-').title(), fontsize=18)
    ax.set_ylabel('Average Concentration', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
for j in range(i + 1, n_rows * n_cols):
    axes_flat[j].axis('off')
fig.suptitle('Average Terpene Concentration by Month (2022+)', fontsize=24, y=1.01)
month_names = list(calendar.month_abbr)[1:]  # Get month names
month_colors = [cmap(norm(month)) for month in range(1, 13)]  # Get month colors from the colormap
legend_handles = [mpatches.Circle((0.5, 0.5), color=color, label=name) for name, color in zip(month_names, month_colors)]
fig.legend(
    handles=legend_handles,
    title='Month',
    bbox_to_anchor=(0.9, 0.175),
    loc='lower right',
    fontsize=14,
    title_fontsize=16,
    ncol=6
)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust the layout to make space for the legend
plt.show()

# TODO: Also look at caryophyllene-limonene

# TODO: Also look at myrcene-pinene


# === Chemical Diversity Analysis ===

def calculate_shannon_diversity(df, compounds):
    """Calculate Shannon Diversity Index."""
    diversities = []
    for _, row in df.iterrows():
        proportions = [pd.to_numeric(row[compound], errors='coerce') for compound in compounds if pd.to_numeric(row[compound], errors='coerce') > 0]
        proportions = np.array(proportions) / sum(proportions)
        shannon_index = -np.sum(proportions * np.log2(proportions))
        diversities.append(shannon_index)
    return diversities


# Calculate chemical diversity for each strain.
strains['chemical_diversity'] = calculate_shannon_diversity(strains, cannabinoids + terpenes)

# Visualize chemical diversity by strain.
plt.figure(figsize=(15, 8))
sns.barplot(
    data=strains.sort_values('strain_name'),
    x='strain_name',
    y='chemical_diversity',
    palette='tab20',
)
plt.title('Chemical Diversity by Strain', fontsize=28)
plt.xlabel('Strain Name', fontsize=21)
plt.ylabel('Chemical Diversity', fontsize=21)
plt.xticks(rotation=90, fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

# Calculate chemical diversity for the whole sample.
sample['chemical_diversity'] = calculate_shannon_diversity(sample, cannabinoids + terpenes)

# Visualize chemical diversity by strain.
plt.figure(figsize=(15, 8))
sns.distplot(
    sample['chemical_diversity'],
    kde=False,
    bins=20,
)
plt.title('Chemical Diversity of Cannabis Flower', fontsize=28)
# plt.xlabel('Strain Name', fontsize=21)
# plt.ylabel('Chemical Diversity', fontsize=21)
plt.xticks(rotation=90, fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

# Group by state and year-month, then calculate the mean chemical diversity
grouped = sample.groupby(['lab_state', sample['date'].dt.to_period('M')])['chemical_diversity'].mean().reset_index()
grouped.rename(columns={'date': 'year_month', 'chemical_diversity': 'avg_chemical_diversity'}, inplace=True)
grouped['year_month'] = grouped['year_month'].dt.to_timestamp()
grouped['avg_chemical_diversity'].fillna(0, inplace=True)
grouped = grouped.loc[grouped['year_month'].dt.year >= 2020]
grouped = grouped.loc[grouped['lab_state'] != 'FL']
pivot = grouped.pivot(index='year_month', columns='lab_state', values='avg_chemical_diversity')
plt.figure(figsize=(15, 8))
sns.lineplot(data=pivot)
plt.title('Average Monthly Chemical Diversity by State (2020-Present)', fontsize=20)
plt.xlabel('Year-Month', fontsize=15)
plt.ylabel('Average Chemical Diversity', fontsize=15)
plt.xticks(rotation=45)
plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc=2)
plt.tight_layout()
plt.show()

# Test if chemical diversity is increasing over time.
sample['year_month'] = sample['date'].dt.to_period('M')
avg_chem_div = sample.groupby(['lab_state', 'year_month'])['chemical_diversity'].mean().reset_index()
avg_chem_div = avg_chem_div.loc[
    (avg_chem_div['lab_state'] != 'FL') &
    (avg_chem_div['year_month'].dt.year >= 2020)
]
avg_chem_div['year_month_numeric'] = (avg_chem_div['year_month'].dt.year - avg_chem_div['year_month'].dt.year.min()) * 12 + avg_chem_div['year_month'].dt.month
model = ols(
    'chemical_diversity ~ year_month_numeric + C(lab_state)',
    data=avg_chem_div,
).fit()
print(model.summary())


# === Total Cannabinoids and Terpenes Analysis ===

# Scatterplot with categorical variables.
sns.catplot(
    data=strains,
    x="strain_name",
    y="total_terpenes",
    hue="strain_name",
    kind="swarm",
)

sns.catplot(
    data=strains,
    x="strain_name",
    y="total_cannabinoids",
    hue="strain_name",
    kind="swarm",
)

# Test for differences in total cannabinoid content across strains
fit_total_cannabinoids = ols('total_cannabinoids ~ C(strain_name)', data=strains).fit()
print(fit_total_cannabinoids.summary())

# Test for differences in total terpene content across strains
fit_total_terpenes = ols('total_terpenes ~ C(strain_name)', data=strains).fit()
print(fit_total_terpenes.summary())

# Test if any strain has a different total THC to total CBD ratio than other strains.
strains['thc_cbd_ratio'] = strains['total_thc'] / strains['total_cbd']
fit_thc_cbd_ratio = ols('thc_cbd_ratio ~ C(strain_name)', data=strains).fit()
print(fit_thc_cbd_ratio.summary())


# === Cultivar Identification ===

# Assign pinene_limonene_ratio to each sample.
strains['pinene_limonene_ratio'] = strains['beta_pinene'] / strains['d_limonene']

# Test if "Blue Dream" has a different ratio than other strains.
strains['is_blue_dream'] = (strains['strain_name'] == 'Blue Dream').astype(int)
strains_clean = strains.dropna(subset=['pinene_limonene_ratio', 'is_blue_dream'])
y = strains_clean['beta_pinene']  # Dependent variable
X = strains_clean[['is_blue_dream']]  # Independent variable
X = sm.add_constant(X)
model_1 = sm.OLS(y, X).fit()
model_1_summary_latex = model_1.summary().as_latex()
print(model_1_summary_latex)

# Test if "Blue Dream" has a different ratio than other strains.
y = strains_clean['d_limonene']  # Dependent variable
X = strains_clean[['is_blue_dream']]  # Independent variable
X = sm.add_constant(X)
model_1 = sm.OLS(y, X).fit()
model_1_summary_latex = model_1.summary().as_latex()
print(model_1_summary_latex)

# Test if all other strains have different ratios from "Blue Dream".
strains_clean = strains.dropna(subset=['pinene_limonene_ratio'])
min_obs = 1  # You can adjust this threshold
strain_counts = strains_clean['strain_name'].value_counts()
strains_to_keep = strain_counts[strain_counts >= min_obs].index
strains_clean_filtered = strains_clean[strains_clean['strain_name'].isin(strains_to_keep)]
fit_2 = ols('pinene_limonene_ratio ~ C(strain_name)', data=strains_clean_filtered).fit()
model_2_summary_latex = fit_2.summary().as_latex()
print(model_2_summary_latex)


# === Environment Analysis ===

# TODO: Test if elevation has an effect on total_terpenes, controlling for:
# - plant_hardiness_zone
# - 3_month_avg_temp
# - strain
# - lab
# - lab_state
# - year
# - month

# Add plant hardiness zone data.
datafile = 'https://cannlytics.page.link/phzm-us-zipcode-2023'
phz = pd.read_csv(datafile)
phz['zipcode'] = phz['zipcode'].astype(str).str.replace('.0', '')
strains = strains.merge(
    phz[['zipcode', 'zone', 'zonetitle']],
    left_on='producer_zipcode',
    right_on='zipcode',
    how='left',
    validate='m:1'
)

# Visualize total terpenes by plant hardiness zone.
zone_palette = {
    '3': '#DB83B4',
    '4': '#8568AD',
    '5': '#779DD0',
    '6': '#4EAB46',
    '6a': '#59b944',
    '6b': '#81c653',
    '7': '#ACD45C',
    '7a': '#afd566',  
    '7b': '#ceda6c',
    '8': '#ECE47C',
    '8a': '#ebda82',
    '8b': '#e8cb50',
    '9': '#E4BA34',
    '9a': '#d8b649',
    '9b': '#f0b674',
    '10': '#F68623',
    '10a': '#e59c2b',
    '10b': '#df7907',
}
zone_palette = {zone: zone_palette.get(str(zone).split(':')[0], '#000000') for zone in strains['zonetitle'].unique()}
strains_filtered = strains.dropna(subset=['zonetitle', 'total_terpenes'])
strains_filtered = strains_filtered[strains_filtered['total_terpenes'] > 0]
zone_order = sorted(strains_filtered['zonetitle'].unique(),
                    key=lambda x: (int(x.split(':')[0].replace('a', '').replace('b', '')), x[-1]))
plt.figure(figsize=(12, 8))
sns.swarmplot(
    data=strains_filtered,
    x="zonetitle",
    y="total_terpenes",
    order=zone_order,
    palette=zone_palette,
)
plt.title('Total Terpenes by Plant Hardiness Zone', fontsize=18)
plt.xlabel('Plant Hardiness Zone', fontsize=14)
plt.ylabel('Total Terpenes', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# === Spatial Analysis ===

# Count results by county.
county_counts = sample.groupby('producer_county').size().reset_index(name='lab_result_count')

# Test if Emerald Triangle producers have different terpene profiles than other producers.
emerald_triangle_counties = [
    'Humboldt County',
    'Trinity County',
    'Mendocino County',
]
sample['is_emerald_triangle'] = sample['producer_county'].isin(emerald_triangle_counties).astype(int)
fit_emerald_triangle = ols('total_terpenes ~ C(is_emerald_triangle)', data=sample).fit()
print(fit_emerald_triangle.summary())

# Future work: Assume elevation, temp, and other environmental data are merged into 'strains' DataFrame
# fit_environment = ols('total_terpenes ~ elevation + temp + C(strain_name) + C(lab_state) + year + month', data=strains).fit()
# print(fit_environment.summary())

# Future work: Create a Choropleth Map
# import geopandas as gpd
# counties = gpd.read_file('us_counties_shapefile.shp')  # Replace with your shapefile
# choropleth_data = counties.merge(county_counts, on='county')
# fig, ax = plt.subplots(1, 1, figsize=(15, 10))
# choropleth_data.plot(column='lab_result_count', ax=ax, legend=True, cmap='viridis')
# plt.title('Lab Results by County')
# plt.show()


# === Paper Replication: Phytochemical Diversity of Commercial Cannabis ===

# Aggregtae all results.
aggregate = pd.concat([
    ca_results[columns],
    co_results[columns],
    fl_results[columns],
    ma_results[columns],
], ignore_index=True)
terpene_results = sample[sample['total_terpenes'] > 0]

# Count the total number of samples.
sample_count_by_lab_state = aggregate.groupby(['lab_state']).size().reset_index(name='sample_count')
print(sample_count_by_lab_state)

# Count the total number of flower samples.
flower_tests_count_by_lab_state = sample.groupby(['lab_state']).size().reset_index(name='test_count')
print(flower_tests_count_by_lab_state)

# Count number of tests with terpene results by lab by state
tests_count_by_lab_state = terpene_results.groupby(['lab_state']).size().reset_index(name='test_count')
print(tests_count_by_lab_state)

# Count number of producers with terpene results by lab by state
producers_count_by_lab_state = terpene_results.groupby(['lab_state'])['producer_zipcode'].nunique().reset_index(name='producer_count')
print(producers_count_by_lab_state)

# Count number of strain names with terpene results by lab by state
strains_count_by_lab_state = terpene_results.groupby(['lab_state'])['strain_name'].nunique().reset_index(name='strain_count')
print(strains_count_by_lab_state)

# Create a table for LaTeX with columns.
# state | Total Tests | Flower Tests | Terpene Tests | Producers | Strains
merged_df = sample_count_by_lab_state
merged_df = merged_df.merge(flower_tests_count_by_lab_state, on='lab_state', how='outer')
merged_df = merged_df.merge(tests_count_by_lab_state, on='lab_state', how='outer')
merged_df = merged_df.merge(producers_count_by_lab_state, on='lab_state', how='outer')
merged_df = merged_df.merge(strains_count_by_lab_state, on='lab_state', how='outer')
merged_df.columns = ["State", "Total Tests", "Flower Tests", "Terpene Tests", "Producers", "Strains"]
merged_df = merged_df.fillna(0).astype({'Total Tests': 'int', 'Flower Tests': 'int', 'Terpene Tests': 'int', 'Producers': 'int', 'Strains': 'int'})
total_row = merged_df.sum(numeric_only=True)
total_row['State'] = 'Total'
merged_df = pd.concat([merged_df, total_row.to_frame().T], ignore_index=True)
format_with_commas = lambda x: "{:,}".format(x)
numeric_cols = ["Total Tests", "Flower Tests", "Terpene Tests", "Producers", "Strains"]
for col in numeric_cols:
    merged_df[col] = merged_df[col].apply(format_with_commas)
latex_table = merged_df.to_latex(index=False, column_format="lccccr", header=True)
print(latex_table)


# === Repeat analysis with the same strains as in the paper ===

top_strains = [
    "Original Glue",
    "Blue Dream",
    "Dutch Treat",
]
list_of_strains = [
    "66 Cookies",
    "9 Pound Hammer",
    "Ak 47",
    "Blue Dream",
    "Bruce Banner",
    "Candyland",
    "Chemdawg",
    "Chernobyl",
    "Cookies And Cream",
    "Do Si Dos",
    "Dogwalker Og",
    "Durban Poison",
    "Dutch Treat",
    "Fpog",
    "Gelato 33",
    "Ghost Og",
    "Grease Monkey",
    "Green Crack",
    "Gsc",
    "Headband",
    "Kimbo Kush",
    "Kosher Kush",
    "Lemon G",
    "Lsd",
    "Mazar X Blueberry Og",
    "Mimosa",
    "Og Kush",
    "Original Glue",
    "Pineapple Express",
    "Purple Punch",
    "Sour Diesel",
    "Sour Tangie",
    "Strawberry Cough",
    "Sunset Sherbert",
    "Super Lemon Haze",
    "Super Silver Haze",
    "Tangie",
    "Tropicana Cookies",
    "Wedding Cake",
    "White Fire Og",
    "White Tahoe Cookies"
]

# Filter the DataFrame for rows where 'strain_name' matches a common strain
subsample = sample.copy()
subsample['matched_strain'] = subsample['strain_name'].apply(
    lambda x: match_common_strain(x, list_of_strains)
)
subsample = subsample[subsample['matched_strain'].isin(list_of_strains)]
subsample['strain_name'] = subsample['matched_strain']

# Count occurrences of each strain in the sample.
strain_counts = strains.groupby(['strain_name']).size().reset_index(name='count')
strain_counts.sort_values('count', ascending=False, inplace=True)
strain_counts.columns = ['Strain Name', 'Observations']
print(strain_counts[['Strain Name', 'Observations']].head(20).to_latex(
    index=False,
    column_format="lr",
))

# Count occurrences of each strain in the sample.
strain_counts = subsample.groupby(['strain_name']).size().reset_index(name='count')
strain_counts.sort_values('count', ascending=False, inplace=True)
strain_counts.columns = ['Strain Name', 'Observations']
print(strain_counts[['Strain Name', 'Observations']].head(20).to_latex(
    index=False,
    column_format="lr",
))

# Look at the caryophyllene-limonene ratio.
y = 'beta_caryophyllene'
x = 'd_limonene'
top_20_strains = subsample['strain_name'].value_counts().nlargest(20).index.sort_values().tolist()
filtered_strains = subsample[subsample['strain_name'].isin(top_20_strains)]
plt.figure(figsize=(15, 8))
scatter_plot = sns.scatterplot(
    data=filtered_strains.sort_values('strain_name'),
    x=x,
    y=y,
    hue='strain_name',
    palette='tab20',
    legend='full',
    s=200
)
x_label = x.replace('_', '-').title()
y_label = y.replace('_', '-').title()
plt.title(f'{y_label} to {x_label} (Top 20 Strains)', fontsize=28)
plt.ylabel(y_label, fontsize=21)
plt.xlabel(x_label, fontsize=21)
plt.tick_params(axis='x', labelsize=18)
plt.tick_params(axis='y', labelsize=18)
scatter_plot.legend_.set_title('Strain', prop={'size': 18})
plt.setp(scatter_plot.get_legend().get_texts(), fontsize='16')  # Set legend text size
plt.setp(scatter_plot.get_legend().get_title(), fontsize='20')  # Set legend title size
plt.tight_layout()
plt.savefig('./presentation/images/top-20-caryophyllene-limonene.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Look at the myrcene-pinene ratio.
y = 'beta_myrcene'
x = 'beta_pinene'
top_20_strains = subsample['strain_name'].value_counts().nlargest(20).index.sort_values().tolist()
filtered_strains = subsample[subsample['strain_name'].isin(top_20_strains)]
plt.figure(figsize=(15, 8))
scatter_plot = sns.scatterplot(
    data=filtered_strains.sort_values('strain_name'),
    x=x,
    y=y,
    hue='strain_name',
    palette='tab20',
    legend='full',
    s=200
)
x_label = x.replace('_', '-').title()
y_label = y.replace('_', '-').title()
plt.title(f'{y_label} to {x_label} (Top 20 Strains)', fontsize=28)
plt.ylabel(y_label, fontsize=21)
plt.xlabel(x_label, fontsize=21)
plt.tick_params(axis='x', labelsize=18)
plt.tick_params(axis='y', labelsize=18)
scatter_plot.legend_.set_title('Strain', prop={'size': 18})
plt.setp(scatter_plot.get_legend().get_texts(), fontsize='16')  # Set legend text size
plt.setp(scatter_plot.get_legend().get_title(), fontsize='20')  # Set legend title size
plt.tight_layout()
plt.savefig('./presentation/images/top-20-myrcene-pinene.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Scatterplot of `beta_pinene` to `d_limonene` by strain.
y = 'beta_pinene'
x = 'd_limonene'
top_20_strains = subsample['strain_name'].value_counts().nlargest(20).index.sort_values().tolist()
filtered_strains = subsample[subsample['strain_name'].isin(top_20_strains)]
plt.figure(figsize=(15, 8))
scatter_plot = sns.scatterplot(
    data=filtered_strains.sort_values('strain_name'),
    x=x,
    y=y,
    hue='strain_name',
    palette='tab20',
    legend='full',
    s=200
)
x_label = x.replace('_', '-').title()
y_label = y.replace('_', '-').title()
plt.title(f'{y_label} to {x_label} (Top 20 Strains)', fontsize=28)
plt.ylabel(y_label, fontsize=21)
plt.xlabel(x_label, fontsize=21)
plt.tick_params(axis='x', labelsize=18)
plt.tick_params(axis='y', labelsize=18)
scatter_plot.legend_.set_title('Strain', prop={'size': 18})
plt.setp(scatter_plot.get_legend().get_texts(), fontsize='16')  # Set legend text size
plt.setp(scatter_plot.get_legend().get_title(), fontsize='20')  # Set legend title size
plt.tight_layout()
plt.savefig('./presentation/images/top-20-pinene-limonene.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# TODO: Calculate between-producer cosine similarity for each strain,
# using terpene data.

# Visualize chemical diversity by strain.
plt.figure(figsize=(12, 8))
chemical_diversity_data = subsample.loc[subsample['chemical_diversity'] > 0]['chemical_diversity']
sns.histplot(
    chemical_diversity_data,
    kde=False,
    bins=35
)
plt.title('Chemical Diversity (Top 20 Strains)', fontsize=28)
plt.ylabel('Observations', fontsize=21)
plt.xlabel('Chemical Diversity', fontsize=21)
plt.xticks(rotation=90, fontsize=18)
plt.yticks(fontsize=18)
mean_chemical_diversity = chemical_diversity_data.mean()
plt.axvline(mean_chemical_diversity, color='b', linestyle='--')
plt.text(
    mean_chemical_diversity,
    plt.ylim()[1] * 0.9,
    f'Mean: {mean_chemical_diversity:.2f}',
    color='b',
    ha='left',
    fontsize=18,
)
plt.tight_layout()
plt.savefig('./presentation/images/chemical-diversity.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Group by state and year-month, then calculate the mean chemical diversity
grouped = subsample.groupby(['lab_state', subsample['date'].dt.to_period('M')])['chemical_diversity'].mean().reset_index()
grouped.rename(columns={'date': 'year_month', 'chemical_diversity': 'avg_chemical_diversity'}, inplace=True)
grouped['year_month'] = grouped['year_month'].dt.to_timestamp()
grouped['avg_chemical_diversity'].fillna(0, inplace=True)
grouped = grouped.loc[grouped['year_month'].dt.year >= 2020]
grouped = grouped.loc[grouped['lab_state'] != 'FL']
pivot = grouped.pivot(index='year_month', columns='lab_state', values='avg_chemical_diversity')
plt.figure(figsize=(18, 10))
sns.set_style("whitegrid")
sns.set_context("talk")  # Larger font sizes
palette = sns.color_palette("husl", n_colors=pivot.shape[1])
sns.lineplot(data=pivot, palette=palette, linewidth=4, marker='o', markersize=10, linestyle='-')  # Solid lines
plt.title('Average Monthly Chemical Diversity by State (2020-Present)', fontsize=24)
plt.xlabel('Year-Month', fontsize=20)
plt.ylabel('Average Chemical Diversity', fontsize=20)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc=2, title_fontsize='18', fontsize='16')
plt.ylim(0, max(grouped['avg_chemical_diversity']) * 1.1)  # Dynamic upper limit based on data
plt.tight_layout()
plt.savefig('./presentation/images/chemical-diversity-by-month.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Test if chemical diversity is increasing over time.
sample['year_month'] = sample['date'].dt.to_period('M')
avg_chem_div = sample.groupby(['lab_state', 'year_month'])['chemical_diversity'].mean().reset_index()
avg_chem_div = avg_chem_div.loc[
    (avg_chem_div['lab_state'] != 'FL') &
    (avg_chem_div['year_month'].dt.year >= 2020)
]
avg_chem_div['year_month_numeric'] = (avg_chem_div['year_month'].dt.year - avg_chem_div['year_month'].dt.year.min()) * 12 + avg_chem_div['year_month'].dt.month
model = ols(
    'chemical_diversity ~ year_month_numeric + C(lab_state)',
    data=avg_chem_div,
).fit()
model_summary = model.summary().as_latex()
print(model_summary)

# Test if Blue Dream has higher beta-pinene than other strains.
subsample['is_blue_dream'] = (subsample['strain_name'] == 'Blue Dream').astype(int)
strains_clean = subsample.dropna(subset=['beta_pinene', 'is_blue_dream'])
y = strains_clean['beta_pinene']  # Dependent variable
X = strains_clean[['is_blue_dream']]  # Independent variable
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
model_summary = model.summary().as_latex()
print(model_summary)


# === Top 20 Totals ===

# Visualize average terpene content by strain.
average_terpenes_strain = subsample.groupby('strain_name')[major_terpenes].mean()
average_terpenes_strain_long = average_terpenes_strain[1:11].reset_index().melt(id_vars='strain_name', var_name='Terpene', value_name='Average Content')
plt.figure(figsize=(15, 9))
barplot = sns.barplot(
    data=average_terpenes_strain_long,
    x='strain_name',
    y='Average Content',
    hue='Terpene',
    palette='tab20',
)
plt.title('Average Terpene Content by Strain', fontsize=28)
plt.xlabel('Strain Name', fontsize=21)
plt.ylabel('Average Content', fontsize=21)
plt.xticks(rotation=90, fontsize=18)
plt.yticks(fontsize=18)
plt.legend(title='Terpene', bbox_to_anchor=(1.05, 1), loc=2, fontsize=18)
plt.legend().get_title().set_fontsize('24')
plt.tight_layout()
plt.savefig('./presentation/images/terpenes-by-strain.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Visualize average terpene content by state.
average_terpenes_state = strains.groupby('lab_state')[major_terpenes].mean()
average_terpenes_state_long = average_terpenes_state.reset_index().melt(id_vars='lab_state', var_name='Terpene', value_name='Average Content')
specific_states = average_terpenes_state.loc[['CA', 'MA']]
specific_states_long = specific_states.reset_index().melt(id_vars='lab_state', var_name='Terpene', value_name='Average Content')
plt.figure(figsize=(15, 8))
barplot = sns.barplot(
    data=specific_states_long,
    x='lab_state',
    y='Average Content',
    hue='Terpene',
    palette='tab20',
)
plt.title('Average Terpene Content by State', fontsize=28)
plt.xlabel('State', fontsize=21)
plt.ylabel('Average Content', fontsize=21)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(
    title='Terpene',
    bbox_to_anchor=(1.05, 1),
    loc=2,
    fontsize=18
)
plt.tight_layout()
plt.savefig('./presentation/images/terpenes-by-state.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Create a timeseries of avg. terpene concentration by month.
filtered_strains = subsample.sort_values('strain_name')
filtered_strains['date'] = pd.to_datetime(filtered_strains['date'])
filtered_strains = filtered_strains[filtered_strains['date'].dt.year >= 2021]
monthly_avg = filtered_strains.groupby([filtered_strains['date'].dt.to_period('M')])[major_terpenes].mean()
monthly_avg = monthly_avg.reset_index()
monthly_avg['date'] = monthly_avg['date'].dt.to_timestamp()
monthly_avg['month'] = monthly_avg['date'].dt.month
monthly_avg_melted = monthly_avg.melt(id_vars='date', var_name='Terpene', value_name='Average Concentration')
n_rows = 3
n_cols = int(np.ceil(len(major_terpenes) / float(n_rows)))
figsize = (21, 4 * n_rows)
fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
axes_flat = axes.flatten()
norm = plt.Normalize(1, 12)
cmap = cm.Spectral_r
for i, terpene in enumerate(major_terpenes):
    ax = axes_flat[i]
    terpene_data = monthly_avg[['date', terpene]].dropna()
    points = np.array([terpene_data['date'].values, terpene_data[terpene].values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(terpene_data['date'].dt.month)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_title(terpene.replace('_', '-').title(), fontsize=18)
    ax.set_ylabel('Average Concentration', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
for j in range(i + 1, n_rows * n_cols):
    axes_flat[j].axis('off')
fig.suptitle('Average Terpene Concentration by Month', fontsize=24, y=1.01)
month_names = list(calendar.month_abbr)[1:]  # Get month names
month_colors = [cmap(norm(month)) for month in range(1, 13)]  # Get month colors from the colormap
legend_handles = [mpatches.Circle((0.5, 0.5), color=color, label=name) for name, color in zip(month_names, month_colors)]
fig.legend(
    handles=legend_handles,
    title='Month',
    bbox_to_anchor=(0.9, 0.175),
    loc='lower right',
    fontsize=14,
    title_fontsize=16,
    ncol=6
)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust the layout to make space for the legend
plt.savefig('./presentation/images/terpenes-by-month.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


# === Environment Analysis for the top 20 strains ===

# Add plant hardiness zone data.
datafile = 'https://cannlytics.page.link/phzm-us-zipcode-2023'
phz = pd.read_csv(datafile)
phz['zipcode'] = phz['zipcode'].astype(str).str.replace('.0', '')
subsample = subsample.merge(
    phz[['zipcode', 'zone', 'zonetitle']],
    left_on='producer_zipcode',
    right_on='zipcode',
    how='left',
    validate='m:1'
)

# Visualize total terpenes by plant hardiness zone.
zone_palette = {
    '3': '#DB83B4',
    '4': '#8568AD',
    '5': '#779DD0',
    '6': '#4EAB46',
    '6a': '#59b944',
    '6b': '#81c653',
    '7': '#ACD45C',
    '7a': '#afd566',  
    '7b': '#ceda6c',
    '8': '#ECE47C',
    '8a': '#ebda82',
    '8b': '#e8cb50',
    '9': '#E4BA34',
    '9a': '#d8b649',
    '9b': '#f0b674',
    '10': '#F68623',
    '10a': '#e59c2b',
    '10b': '#df7907',
}
zone_palette = {zone: zone_palette.get(str(zone).split(':')[0], '#000000') for zone in subsample['zonetitle'].unique()}
strains_filtered = subsample.dropna(subset=['zonetitle', 'total_terpenes'])
strains_filtered = strains_filtered[strains_filtered['total_terpenes'] > 0]
zone_order = sorted(strains_filtered['zonetitle'].unique(),
                    key=lambda x: (int(x.split(':')[0].replace('a', '').replace('b', '')), x[-1]))

# Custom sorting function for plant hardiness zones.
def sort_zones(zone):
    parts = zone.split(' ')
    print(parts)
    main_part = int(parts[0].replace('a', '').replace('b', '').replace(':', ''))
    sub_part = parts[1] if len(parts) > 1 else ''
    return (main_part, sub_part)

# Sort the zones in human-readable order.
zone_order = sorted(strains_filtered['zonetitle'].unique(), key=sort_zones)

# Visualize total terpenes by plant hardiness zone.
plt.figure(figsize=(12, 8))
sns.swarmplot(
    data=strains_filtered,
    x="zonetitle",
    y="total_terpenes",
    order=zone_order,
    palette=zone_palette,
)
plt.title('Total Terpenes by Plant Hardiness Zone', fontsize=18)
plt.xlabel('Plant Hardiness Zone', fontsize=14)
plt.ylabel('Total Terpenes', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./presentation/images/total-terpenes-by-zone.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Visualize average terpene content by plant hardiness zone.
average_terpenes_zone = strains_filtered.groupby('zonetitle')[major_terpenes].mean()
average_terpenes_zone_long = average_terpenes_zone.reset_index().melt(id_vars='zonetitle', var_name='Terpene', value_name='Average Content')
plt.figure(figsize=(15, 8))
barplot = sns.barplot(
    data=average_terpenes_zone_long,
    x='zonetitle',
    y='Average Content',
    hue='Terpene',
    order=zone_order,  # Apply the same order here
    palette='tab20'
)
plt.title('Average Terpene Content by Plant Hardiness Zone', fontsize=28)
plt.xlabel('Plant Hardiness Zone', fontsize=21)
plt.ylabel('Average Content', fontsize=21)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.legend(
    title='Terpene',
    bbox_to_anchor=(1.05, 1),
    loc=2,
    fontsize=18
)
plt.tight_layout()
plt.savefig('./presentation/images/terpenes-by-zone.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Test if plant zone hardiness is correlated with total terpenes.
regression_data = subsample.dropna(subset=['total_terpenes', 'zonetitle'])
regression_data['total_terpenes'] = pd.to_numeric(regression_data['total_terpenes'], errors='coerce')
regression_data = regression_data.loc[regression_data['total_terpenes'] > 0]
zone_dummies = pd.get_dummies(regression_data['zonetitle'], drop_first=True)
X = sm.add_constant(zone_dummies)  # Adding a constant term to the predictors
y = regression_data['total_terpenes']
model = sm.OLS(y, X).fit()
model_summary = model.summary().as_latex()
print(model_summary)

# Test if plant zone hardiness is correlated with beta caryophyllene.
regression_data = subsample.dropna(subset=['beta_caryophyllene', 'zonetitle'])
regression_data['beta_caryophyllene'] = pd.to_numeric(regression_data['beta_caryophyllene'], errors='coerce')
regression_data = regression_data.loc[regression_data['beta_caryophyllene'] > 0]
zone_dummies = pd.get_dummies(regression_data['zonetitle'], drop_first=True)
X = sm.add_constant(zone_dummies)  # Adding a constant term to the predictors
y = regression_data['beta_caryophyllene']
model = sm.OLS(y, X).fit()
model_summary = model.summary().as_latex()
print(model_summary)


# === Elevation Analysis ===

# Create a relational plot between total terpenes and elevation (where elevation data is not missing).
elevation_terpenes_data = strains_filtered.dropna(subset=['elevation', 'total_terpenes'])
plt.figure(figsize=(12, 8))
scatter_plot = sns.scatterplot(
    data=elevation_terpenes_data,
    x='elevation',
    y='total_terpenes',
    color='olive',
    edgecolor='black',
    s=50  # Adjust marker size as needed
)
plt.title('Total Terpenes vs Elevation', fontsize=20)
plt.xlabel('Elevation (meters)', fontsize=16)
plt.ylabel('Total Terpenes', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('./presentation/images/terpenes-by-zone.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


# === Geography Analysis ===

# Count results by county.
county_counts = subsample.groupby('producer_county').size().reset_index(name='count')
county_counts.sort_values('count', ascending=False, inplace=True)
county_counts.columns = ['County', 'Observations']
print(county_counts[['County', 'Observations']].head(20).to_latex(
    index=False,
    column_format="lr",
))

# Test if Emerald Triangle producers have different terpene profiles than other producers.
emerald_triangle_counties = [
    'Humboldt County',
    'Trinity County',
    'Mendocino County',
]
subsample['is_emerald_triangle'] = subsample['producer_county'].isin(emerald_triangle_counties).astype(int)
fit_emerald_triangle = ols('total_terpenes ~ C(is_emerald_triangle)', data=sample).fit()
model_summary = fit_emerald_triangle.summary().as_latex()
print(model_summary)
