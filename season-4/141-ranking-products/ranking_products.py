"""
Analyze California Cannabis Lab Results
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 12/10/2023
Updated: 1/3/2024
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# Standard imports:
import ast
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# External imports:
import pandas as pd
import statsmodels.api as sm

from cannlytics.utils import convert_to_numeric


# TODO: Make these standard functions.


def get_result_value(
        results,
        analyte,
        key='key',
        value='value',
    ):
    """Get the value for an analyte from a list of standardized results."""
    # Ensure that the results are a list.
    try:
        result_list = json.loads(results)
    except:
        try:
            result_list = ast.literal_eval(results)
        except:
            result_list = []
    if not isinstance(result_list, list):
        return None

    # Get the value of interest from the list of results.
    result_data = pd.DataFrame(result_list)
    if result_data.empty:
        return None
    try:
        result = result_data.loc[result_data[key] == analyte, value].iloc[0]
    except:
        return 0
    try:
        return convert_to_numeric(result, strip=True)
    except:
        return result
    

# # Identify all unique cannabinoids and terpenes.
# cannabinoids = []
# terpenes = []
# for item in emerald['results']:
#     lab_results = ast.literal_eval(item)
#     for result in lab_results:
#         if result['analysis'] == 'cannabinoids':
#             cannabinoids.append(result['name'])
#         elif result['analysis'] == 'terpenes':
#             terpenes.append(result['name'])
# cannabinoids = list(set(cannabinoids))
# terpenes = list(set(terpenes))
# print('Cannabinoids:', cannabinoids)
# print('Terpenes:', terpenes)


# === Setup ===

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})


# === Read the data ===

# Define California datasets.
CA_LAB_RESULTS = {
    'Flower Company': {
        'datafiles': [
            # './data/ca-results-flower-company-2023-12-13.xlsx',
        ],
    },
    'Glass House Farms': {
        'datafiles': [],
    },
    'SC Labs':{
        'datafiles': [
            # 'D:/data/california/lab_results/datasets/sclabs/ca-lab-results-sclabs-2024-01-02-00-39-36.xlsx',
            # 'D:/data/california/lab_results/datasets/sclabs/ca-lab-results-sclabs-2024-01-03-06-24-11.xlsx',
            'data/ca-lab-results-sclabs-2023-12-31.xlsx',
        ],
    },
}
SC_LABS_BASE_URL = 'https://client.sclabs.com/verify/'


# Read all datafiles.
all_results = []
for source in CA_LAB_RESULTS:
    for datafile in CA_LAB_RESULTS[source]['datafiles']:
        data = pd.read_excel(datafile)
        all_results.append(data)

# Aggregate results.
results = pd.concat(all_results)
results.drop_duplicates(subset=['coa_id'], inplace=True)
print('Number of results:', len(results))
# results.to_excel('data/ca-lab-results-sclabs-2023-12-31.xlsx')

# === Assign chemical values ===
cannabinoids = [
    'thca',
    'cbga',
    'cbca',
    'delta_9_thc',
    'cbg',
    'thcva',
    'cbda',
    'delta_8_thc',
    'thcv',
    'cbd',
    'cbdv',
    'cbdva',
    'cbl',
    'cbn',
    'cbc',
]
terpenes = [
    'beta_caryophyllene',
    'd_limonene',
    'alpha_humulene',
    'beta_myrcene',
    'beta_pinene',
    'alpha_pinene',
    'beta_ocimene',
    'alpha_bisabolol',
    'terpineol',
    'fenchol',
    'linalool',
    'borneol',
    'camphene',
    'terpinolene',
    'fenchone',
    'nerolidol',
    'trans_beta_farnesene',
    'citronellol',
    'sabinene_hydrate',
    'nerol',
    'valencene',
    'sabinene',
    'alpha_phellandrene',
    'delta_3_carene',
    'alpha_terpinene',
    'p_cymene',
    'eucalyptol',
    'gamma_terpinene',
    'isopulegol',
    'camphor',
    'isoborneol',
    'menthol',
    'pulegone',
    'geraniol',
    'geranyl_acetate',
    'alpha_cedrene',
    'caryophyllene_oxide',
    'guaiol',
    'cedrol'
]

# Get the results for each cannabinoid and terpene.
for a in cannabinoids + terpenes:
    print('Augmenting:', a)
    results[a] = results['results'].apply(
        lambda x: get_result_value(x, a, key='key')
    )

# Save the augmented results.
timestamp = pd.Timestamp.now().strftime('%Y-%m-%d-%H-%M-%S')
results.to_excel(f'data/augmented-ca-lab-results-sclabs-{timestamp}.xlsx')


# === Look at the Emerald Cup results ===
    
# Read the Emerald Cup lab results.
emerald_2023 = results.loc[results['producer'] == 'Emerald Cup 2023']
emerald_2022 = results.loc[results['producer'] == 'Emerald Cup 2022']
emerald_2020 = results.loc[results['producer'] == 'Emerald Cup 2020']
emerald_2023['year'] = 2023
emerald_2022['year'] = 2022
emerald_2020['year'] = 2020
emerald = pd.concat([emerald_2023, emerald_2022, emerald_2020])

emerald_2023.set_index('coa_id', inplace=True)
emerald_2022.set_index('coa_id', inplace=True)

# Read Emerald Cup ranking data.
winners_2023 = pd.read_excel('data/emerald-cup-winners-2023.xlsx')
winners_2022 = pd.read_excel('data/emerald-cup-winners-2022.xlsx')

# Merge with Emerald Cup ranking data.
emerald_2023 = pd.merge(
    emerald_2023,
    winners_2023,
    on='coa_id',
    how='left'
)
emerald_2022 = pd.merge(
    emerald_2022,
    winners_2022,
    on='coa_id',
    how='left'
)


# Merge with Emerald Cup ranking data.
# Note: This is a fuzzy match and was finished with human hands.
# winners_2023['product_name'] = winners_2023['entry_name'].apply(
#     lambda x: x.split(' – ')[-1]
# )
# for index, row in winners_2023.iterrows():
#     name = row['product_name']
#     match = emerald_2023.loc[emerald_2023['product_name'].str.contains(name)]
#     if len(match):
#         coa_id = match['coa_id'].iloc[0]
#         lab_results_url = SC_LABS_BASE_URL + coa_id.split('-')[0]
#         winners_2023.loc[index, 'coa_id'] = coa_id
#         winners_2023.loc[index, 'lab_results_url'] = lab_results_url
#         print('Matched:', name)
#     else:
#         print('No match:', name)
# winners_2022['product_name'] = winners_2022['entry_name'].apply(
#     lambda x: x.split(' – ')[-1]
# )
# for index, row in winners_2022.iterrows():
#     name = row['product_name']
#     match = emerald_2022.loc[emerald_2022['product_name'].str.contains(name)]
#     if len(match):
#         coa_id = match['coa_id'].iloc[0]
#         lab_results_url = SC_LABS_BASE_URL + coa_id.split('-')[0].strip()
#         winners_2022.loc[index, 'coa_id'] = coa_id
#         winners_2022.loc[index, 'lab_results_url'] = lab_results_url
#         print('Matched:', name)
#     else:
#         print('No match:', name)
# winners_2023.to_excel('data/augmented-emerald-cup-winners-2023.xlsx')
# winners_2022.to_excel('data/augmented-emerald-cup-winners-2022.xlsx')


# === Contest statistics ===
        
# TODO: Calculate number of entrants.


# ===  Highest Terpene Content – Flower ===

# (2022) Highest Terpene Content – Flower	Woodwide Farms – Mendo Crumble
flower_2022 = emerald_2022.loc[emerald_2022['product_type'] == 'Flower, Inhalable']
highest_terpene_content_flower_2022 = flower_2022.loc[flower_2022['total_terpenes'] == flower_2022['total_terpenes'].max()]
highest_terpene_content_flower_2022[['product_name', 'total_terpenes']]

# (2023) Highest Terpene Content – Flower	Sanctuary Farms – Blunicorn – 4.87% Total Terpenes
flower_2023 = emerald_2023.loc[emerald_2023['product_type'] == 'Flower, Inhalable']
highest_terpene_content_flower_2023 = flower_2023.loc[flower_2023['total_terpenes'] == flower_2023['total_terpenes'].max()]
highest_terpene_content_flower_2023[['product_name', 'total_terpenes']]

# Calculate the percent change in total terpenes in the winner.
yoy_change = (highest_terpene_content_flower_2023['total_terpenes'].iloc[0] - highest_terpene_content_flower_2022['total_terpenes'].iloc[0]) / highest_terpene_content_flower_2022['total_terpenes'].iloc[0]
print(f'YoY change: {round(yoy_change * 100, 2)}%')


# === Highest Terpene Content – Solventless ===

# (2022) Highest Terpene Content – Solventless	Have Hash – Rainbow Belts


# (2023) Highest Terpene Content – Solventless	Papa’s Select – OGZ x Strawberry Meltshake 90u Ice Water Hash – 9.37% Total Terpenes


# === Highest Terpene Content – Solvent ===

# (2022) Highest Terpene Content – Solvent (Hydrocarbon)	Errl Hill – Gazberries


# (2023) Highest Terpene Content – Solvent	Jetty Extracts – Super Lemon Haze Live Badder – 11.44% Total Terpenes


# (2023) Highest Terpene Content – Personal Use	Paul Lopez – Project Power Jam Whip Donut – 10.38% Total Terpenes


# === TODO: Most Unique Terpene Profile – Flower ===

def calculate_shannon_diversity(df, compounds):
    """Calculate Shannon Diversity Index."""
    diversities = []
    for _, row in df.iterrows():
        # Convert the compound values to numeric and filter those greater than 0
        proportions = [pd.to_numeric(row[compound], errors='coerce') for compound in compounds if pd.to_numeric(row[compound], errors='coerce') > 0]
        proportions = np.array(proportions) / sum(proportions)
        shannon_index = -np.sum(proportions * np.log2(proportions))
        diversities.append(shannon_index)
    return diversities


# (2022) Most Unique Terpene Profile	Atrium Cultivation – Juice Z
flower_2022['terpene_diversity'] = calculate_shannon_diversity(flower_2022, terpenes)

# (2023) Most Unique Terpene Profile – Flower	Healing Herb Farms – Lilac Mintz
flower_2023['terpene_diversity'] = calculate_shannon_diversity(flower_2023, terpenes)


# Find the most unique terpene profiles
most_unique_2022 = flower_2022.loc[flower_2022['terpene_diversity'] == flower_2022['terpene_diversity'].max()]
most_unique_2023 = flower_2023.loc[flower_2023['terpene_diversity'] == flower_2023['terpene_diversity'].max()]

# Create a histogram for terpene diversity
plt.figure(figsize=(12, 6))
combined_data = pd.concat([flower_2022['terpene_diversity'], flower_2023['terpene_diversity']])
labels = ['2022', '2023']
plt.hist([flower_2022['terpene_diversity'], flower_2023['terpene_diversity']], bins=30, alpha=0.7, label=labels)
plt.axvline(most_unique_2022['terpene_diversity'].values[0], color='red', linestyle='dashed', linewidth=2)
plt.text(most_unique_2022['terpene_diversity'].values[0], 5, f"2022: {most_unique_2022['product_name'].values[0]}", rotation=90)
plt.axvline(most_unique_2023['terpene_diversity'].values[0], color='blue', linestyle='dashed', linewidth=2)
plt.text(most_unique_2023['terpene_diversity'].values[0], 10, f"2023: {most_unique_2023['product_name'].values[0]}", rotation=90)
plt.xlabel('Terpene Diversity')
plt.ylabel('Frequency')
plt.title('Histogram of Terpene Diversity for Cannabis Flowers (2022 vs 2023)')
plt.legend()
plt.show()

# TODO: Depict the terpene profile of the winner.

# TODO: Calculate the percent change in terpene diversity in the winner.


# === TODO: Most Unique Terpene Profile – Concentrate ===


# (2023) Most Unique Terpene Profile – Concentrate	Brett Byrd – Lilac Mintz 45-159 Live Rosin



# === Most Unique Cannabinoid Profile – Flower ===

# (2022) Most Unique Cannabinoid Profile	Emerald Spirit Botanicals – Pink Boost Goddess
flower_2022['cannabinoid_diversity'] = calculate_shannon_diversity(flower_2022, cannabinoids)

# (2023) Most Unique Cannabinoid Profile – Flower	Emerald Spirit Botanicals – Pink Boost Goddess – 1:1 THC:THCV
flower_2023['cannabinoid_diversity'] = calculate_shannon_diversity(flower_2023, cannabinoids)

# Find the most unique cannabinoid profiles
most_unique_2022 = flower_2022.loc[flower_2022['cannabinoid_diversity'] == flower_2022['cannabinoid_diversity'].max()]
most_unique_2023 = flower_2023.loc[flower_2023['cannabinoid_diversity'] == flower_2023['cannabinoid_diversity'].max()]

# Create a histogram for cannabinoid diversity
plt.figure(figsize=(12, 6))
combined_data = pd.concat([flower_2022['cannabinoid_diversity'], flower_2023['cannabinoid_diversity']])
labels = ['2022', '2023']
plt.hist([flower_2022['cannabinoid_diversity'], flower_2023['cannabinoid_diversity']], bins=30, alpha=0.7, label=labels)
plt.axvline(most_unique_2022['cannabinoid_diversity'].values[0], color='red', linestyle='dashed', linewidth=2)
plt.text(most_unique_2022['cannabinoid_diversity'].values[0], 5, f"2022: {most_unique_2022['product_name'].values[0]}", rotation=90)
plt.axvline(most_unique_2023['cannabinoid_diversity'].values[0], color='blue', linestyle='dashed', linewidth=2)
plt.text(most_unique_2023['cannabinoid_diversity'].values[0], 10, f"2023: {most_unique_2023['product_name'].values[0]}", rotation=90)
plt.xlabel('Terpene Diversity')
plt.ylabel('Frequency')
plt.title('Cannabinoid Diversity for Cannabis Flowers (2022 vs 2023)')
plt.legend()
plt.show()

# TODO: Depict the most unique strain's cannabinoids beside the average.
for cannabinoid in cannabinoids:
    flower_2022[cannabinoid] = pd.to_numeric(flower_2022[cannabinoid], errors='coerce')
    flower_2023[cannabinoid] = pd.to_numeric(flower_2023[cannabinoid], errors='coerce')
avg_cannabinoids_2022 = flower_2022[cannabinoids].mean()
avg_cannabinoids_2022.fillna(0, inplace=True)
avg_cannabinoids_2023 = flower_2023[cannabinoids].mean()
avg_cannabinoids_2023.fillna(0, inplace=True)
unique_cannabinoids_2022 = pd.to_numeric(most_unique_2022[cannabinoids].iloc[0], errors='coerce')
unique_cannabinoids_2023 = pd.to_numeric(most_unique_2023[cannabinoids].iloc[0], errors='coerce')
unique_cannabinoids_2022.fillna(0, inplace=True)
unique_cannabinoids_2023.fillna(0, inplace=True)
fig, ax = plt.subplots(2, 1, figsize=(15, 10))
bar_width = 0.35
bar_l = [i+1 for i in range(len(cannabinoids))]
tick_pos = [i+(bar_width/2) for i in bar_l]
ax[0].bar(bar_l, unique_cannabinoids_2022, width=bar_width, label='Most Unique 2022', color='orange')
ax[0].bar([p + bar_width for p in bar_l], avg_cannabinoids_2022, width=bar_width, label='Average 2022', color='blue')
ax[0].set_xticks(tick_pos)
ax[0].set_xticklabels(cannabinoids, rotation=45, ha='right')
ax[0].set_title('2022 Cannabinoid Comparison')
ax[0].legend()
ax[1].bar(bar_l, unique_cannabinoids_2023, width=bar_width, label='Most Unique 2023', color='red')
ax[1].bar([p + bar_width for p in bar_l], avg_cannabinoids_2023, width=bar_width, label='Average 2023', color='green')
ax[1].set_xticks(tick_pos)
ax[1].set_xticklabels(cannabinoids, rotation=45, ha='right')
ax[1].set_title('2023 Cannabinoid Comparison')
ax[1].legend()
plt.tight_layout()
plt.show()

# Calculate the percent change in cannabinoid diversity in the winner.
yoy_change = (most_unique_2023['cannabinoid_diversity'].iloc[0] - most_unique_2022['cannabinoid_diversity'].iloc[0]) / most_unique_2022['cannabinoid_diversity'].iloc[0]
print(f'YoY change in cannabinoid diversity of the winner: {round(yoy_change * 100, 2)}%')


# === Additional winnners ===


# (2023) Most Innovative Product – Consumable	Compound Genetics x Node Labs x The Original Resinator x Industry Processing Solutions – Perzimmon #2 Flower


# (2022) Most Innovative Product – Consumable	Holy Water x Honey Suckle Lotus – Jelly Ranchers. Unholy Rosin/Resin Split Jar




# === TODO: Build an ordered probit model to back-cast 2020 winners. ===


# Clean the rank.
rank_mapping = {
    '1st': 1, '2nd': 2, '3rd': 3, '4th': 4, '5th': 5,
    '6th': 6, '7th': 7, '8th': 8, '9th': 9, '10th': 10,
    # ... up to '20th'
}
df['rank'] = df['rank'].map(rank_mapping)


# === Image analysis ===

from rembg import remove
from PIL import Image


def remove_bg(input_path: str, output_path: str) -> None:
    """Convert a video file to another video with a transparent background.
    
    Args:
        input_path (str): The path to the input image file.
        output_path (str): The path to save the output image with a transparent background.
    """
    input = Image.open(input_path)
    output = remove(input)
    output.save(output_path)


# TODO: Is there any interesting analysis that can be done with the images?


# TODO: Download the images to an image dir.


# TODO: Crop the images.


# TODO: Get a measure of color for each image.

