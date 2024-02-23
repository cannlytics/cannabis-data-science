"""
Analyze California Cannabis Lab Results
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 12/10/2023
Updated: 1/4/2024
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# Standard imports:
import ast
import os
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import requests

# External imports:
import cv2
from PIL import Image
from cannlytics.utils import snake_case
import pandas as pd
from rembg import remove


# === Setup ===

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
})


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


def remove_bg(input_path: str, output_path: str) -> None:
    """Convert a video file to another video with a transparent background.
    
    Args:
        input_path (str): The path to the input image file.
        output_path (str): The path to save the output image with a transparent background.
    """
    input = Image.open(input_path)
    output = remove(input)
    output.save(output_path)


# === Define chemicals ===

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


# === Read the data ===

# Read the augmented results.
results = pd.read_excel('data/augmented-ca-lab-results-sclabs-2024-01-03-15-26-35.xlsx')


# === Look at the Emerald Cup results ===

# Read the Emerald Cup lab results.
emerald_2023 = results.loc[results['producer'] == 'Emerald Cup 2023']
emerald_2022 = results.loc[results['producer'] == 'Emerald Cup 2022']
emerald_2020 = results.loc[results['producer'] == 'Emerald Cup 2020']
emerald_2023['year'] = 2023
emerald_2022['year'] = 2022
emerald_2020['year'] = 2020
emerald_2023.set_index('coa_id', inplace=True)
emerald_2022.set_index('coa_id', inplace=True)

# Read Emerald Cup ranking data.
winners_2023 = pd.read_excel('data/augmented-emerald-cup-winners-2023.xlsx')
winners_2022 = pd.read_excel('data/augmented-emerald-cup-winners-2022.xlsx')

# Merge with Emerald Cup ranking data.
emerald_2023 = pd.merge(
    emerald_2023,
    winners_2023,
    on='coa_id',
    how='left',
    suffixes=('', '_winner')
)
emerald_2022 = pd.merge(
    emerald_2022,
    winners_2022,
    on='coa_id',
    how='left',
    suffixes=('', '_winner')
)

# Look at a sample.
def first_place_table(sample):
    sample = sample.loc[(sample['rank'] == '1st') & (sample['category'].str.contains('Flower'))]
    sample = sample.sort_values(by='rank', ascending=False)[['coa_id', 'product_name', 'category', 'total_cannabinoids', 'total_terpenes']]
    sample['coa_id'] = sample['coa_id'].apply(lambda x: x.split('-')[0].strip())
    # sample['product_name'] = sample['product_name'].apply(lambda x: x.split('-')[-1].strip())
    sample['category'] = sample['category'].str.replace(' Flower', '')
    sample.columns = ['COA ID', 'Product Name', 'Category', 'Cannabinoids', 'Terpenes']
    format_dict = {'Cannabinoids': '{:.2f}', 'Terpenes': '{:.2f}'}
    column_format = 'lL{4cm}L{3cm}rr'
    latex_table = sample.to_latex(
        index=False,
        formatters=format_dict,
        column_format=column_format,
        escape=False,
    )
    latex_table = latex_table.replace('\\\\\n', '\\\\[1.5em]\n')
    print(latex_table)


first_place_table(emerald_2023)
first_place_table(emerald_2022)


# === Contest statistics ===
  
# Calculate number of entrants.
# print('Number of entrants in 2023:', len(emerald_2023))
# print('Number of entrants in 2022:', len(emerald_2022))
# print('YoY change in entrants:', round((len(emerald_2023) - len(emerald_2022)) / len(emerald_2022) * 100, 2), '%')


# ===  Highest Terpene Content – Flower ===

# (2022) Highest Terpene Content – Flower
flower_2022 = emerald_2022.loc[emerald_2022['product_type'] == 'Flower, Inhalable']
highest_terpene_content_flower_2022 = flower_2022.loc[flower_2022['total_terpenes'] == flower_2022['total_terpenes'].max()]
highest_terpene_content_flower_2022[['product_name', 'total_terpenes']]
winner = highest_terpene_content_flower_2022.iloc[0]
print("\n === 2022 Highest Terpene Content ===")
print(f"{winner['product_name']}, {winner['total_terpenes']}% total terpenes")
print('ID:', winner['coa_id'])

# (2023) Highest Terpene Content – Flower
flower_2023 = emerald_2023.loc[emerald_2023['product_type'] == 'Flower, Inhalable']
highest_terpene_content_flower_2023 = flower_2023.loc[flower_2023['total_terpenes'] == flower_2023['total_terpenes'].max()]
highest_terpene_content_flower_2023[['product_name', 'total_terpenes']]
winner = highest_terpene_content_flower_2023.iloc[0]
print("\n === 2023 Highest Terpene Content ===")
print(f"{winner['product_name']}, {winner['total_terpenes']}% total terpenes")
print('ID:', winner['coa_id'])

# Calculate the percent change in total terpenes in the winner.
yoy_change = (highest_terpene_content_flower_2023['total_terpenes'].iloc[0] - highest_terpene_content_flower_2022['total_terpenes'].iloc[0]) / highest_terpene_content_flower_2022['total_terpenes'].iloc[0]
print(f'\n\nYoY change in total terpenes of the winner: {round(yoy_change * 100, 2)}%')

# Histogram of total terpenes for 2022 and 2023.
plt.figure(figsize=(15, 8))
plt.hist([flower_2022['total_terpenes'], flower_2023['total_terpenes']], bins=30, alpha=0.7, label=['2022', '2023'])
x_2022 = highest_terpene_content_flower_2022['total_terpenes'].values[0]
plt.axvline(x_2022, color='blue', linestyle='dashed', linewidth=2)
plt.text(
    x_2022, 5, f"2022: {highest_terpene_content_flower_2022['product_name'].values[0]}",
    fontdict={'size': 20}, ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7)
)
x_2023 = highest_terpene_content_flower_2023['total_terpenes'].values[0]
plt.axvline(x_2023, color='red', linestyle='dashed', linewidth=2)
plt.text(
    x_2023, 10, f"2023: {highest_terpene_content_flower_2023['product_name'].values[0]}",
    fontdict={'size': 20}, ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7)
)
plt.xlabel('Total Terpenes (%)', fontsize=24)
plt.ylabel('Observations', fontsize=24)
plt.title('Total Terpenes of Cannabis Flower (Emerald Cup 2022 vs 2023)', fontsize=26)
plt.legend(fontsize=20)
plt.savefig('./figures/emerald-cup-total-terpenes-histogram.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()


# === Most Unique Terpene Profile – Flower ===

# (2022) Most Unique Terpene Profile
flower_2022['terpene_diversity'] = calculate_shannon_diversity(flower_2022, terpenes)
most_unique_terpenes_2022 = flower_2022.loc[flower_2022['terpene_diversity'] == flower_2022['terpene_diversity'].max()]
print("\n === 2022 Most Unique Terpene Profile ===")
print(f"Predicted winner: {most_unique_terpenes_2022['product_name'].iloc[0]}")
print('Official winner: Atrium Cultivation – Juice Z')
winner_terpene_diversity_2022 = flower_2022.loc[flower_2022['entry_name'] == 'Atrium Cultivation – Juice Z']
print('ID:', winner_terpene_diversity_2022['coa_id'])

# (2023) Most Unique Terpene Profile – Flower	Healing Herb Farms – Lilac Mintz
flower_2023['terpene_diversity'] = calculate_shannon_diversity(flower_2023, terpenes)
most_unique_terpenes_2023 = flower_2023.loc[flower_2023['terpene_diversity'] == flower_2023['terpene_diversity'].max()]
print("\n === 2023 Most Unique Terpene Profile ===")
print(f"Predicted winner: {most_unique_terpenes_2023['product_name'].iloc[0]}")
print('Official winner: Healing Herb Farms – Lilac Mintz')
winner_terpene_diversity_2023 = flower_2023.loc[flower_2023['entry_name'] == 'Brett Byrd – Lilac Mintz 45-159 Live Rosin – Phytofacts Report']
print('ID:', winner_terpene_diversity_2023['coa_id'])

# Calculate the percent change in terpene diversity in the winner.
yoy_change = (most_unique_terpenes_2023['terpene_diversity'].iloc[0] - most_unique_terpenes_2022['terpene_diversity'].iloc[0]) / most_unique_terpenes_2022['terpene_diversity'].iloc[0]
print(f'\n\nYoY change in terpene diversity of the winner: {round(yoy_change * 100, 2)}%')

# Depict the terpene profile of the winner.
for terpene in terpenes:
    flower_2022[terpene] = pd.to_numeric(flower_2022[terpene], errors='coerce')
    flower_2023[terpene] = pd.to_numeric(flower_2023[terpene], errors='coerce')
avg_terpenes_2022 = flower_2022[terpenes].mean()
avg_terpenes_2022.fillna(0, inplace=True)  # Replace NaNs with 0 for plotting
avg_terpenes_2023 = flower_2023[terpenes].mean()
avg_terpenes_2023.fillna(0, inplace=True)
unique_terpenes_2022 = pd.to_numeric(most_unique_terpenes_2022[terpenes].iloc[0], errors='coerce')
unique_terpenes_2023 = pd.to_numeric(most_unique_terpenes_2023[terpenes].iloc[0], errors='coerce')
unique_terpenes_2022.fillna(0, inplace=True)  # Replace NaNs with 0 for plotting
unique_terpenes_2023.fillna(0, inplace=True)

fig, ax = plt.subplots(2, 1, figsize=(24, 10))
bar_width = 0.25  # Adjust this to control the width of the bars
bar_l = np.arange(len(terpenes))
tick_pos = bar_l + bar_width / 2
ax[0].bar(bar_l, unique_terpenes_2022, width=bar_width, label='2022 Highest Diversity', color='darkorange')
ax[0].bar(bar_l + bar_width, avg_terpenes_2022, width=bar_width, label='Average 2022', color='blue')
ax[0].bar(bar_l + 2*bar_width, winner_terpene_diversity_2022.iloc[0][terpenes].fillna(0), width=bar_width, label='2022 Winner', color='orange')
ax[0].set_xticks(tick_pos)
ax[0].set_xticklabels(terpenes, rotation=45, ha='right')
ax[0].set_title('2022 Terpene Comparison')
ax[0].legend()
ax[1].bar(bar_l, unique_terpenes_2023, width=bar_width, label='2023 Highest Diversity', color='violet')
ax[1].bar(bar_l + bar_width, avg_terpenes_2023, width=bar_width, label='Average 2023', color='green')
ax[1].bar(bar_l + 2*bar_width, winner_terpene_diversity_2023.iloc[0][terpenes].fillna(0), width=bar_width, label='2023 Winner', color='darkviolet')
ax[1].set_xticks(tick_pos)
ax[1].set_xticklabels(terpenes, rotation=45, ha='right')
ax[1].set_title('2023 Terpene Comparison')
ax[1].legend()
plt.tight_layout()
plt.savefig('./figures/emerald-cup-terpene-diversity-winner.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

# Create a histogram for terpene diversity
plt.figure(figsize=(15, 8))
labels = ['2022', '2023']
plt.hist([flower_2022['terpene_diversity'], flower_2023['terpene_diversity']], bins=30, alpha=0.7, label=labels)
def annotate_diversity(data, color, ypos, label):
    plt.axvline(data['terpene_diversity'].values[0], color=color, linestyle='dashed', linewidth=2)
    plt.text(
        data['terpene_diversity'].values[0], ypos, f"{label}: {data['product_name'].values[0]}",
        fontdict={'size': 20}, ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7)
    )
annotate_diversity(most_unique_terpenes_2022, 'blue', 5, '2022 Most Unique')
annotate_diversity(most_unique_terpenes_2023, 'red', 10, '2023 Most Unique')
annotate_diversity(winner_terpene_diversity_2022, 'green', 15, '2022 Winner')
annotate_diversity(winner_terpene_diversity_2023, 'orange', 20, '2023 Winner')
plt.xlabel('Terpene Diversity', fontsize=24)
plt.ylabel('Observations', fontsize=24)
plt.title('Terpene Diversity of Cannabis Flowers (Emerald Cup 2022 vs 2023)', fontsize=26)
plt.legend(fontsize=20)
plt.savefig('./figures/emerald-cup-terpene-diversity-histogram.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()


# === Most Unique Cannabinoid Profile – Flower ===

# (2022) Most Unique Cannabinoid Profile
flower_2022['cannabinoid_diversity'] = calculate_shannon_diversity(flower_2022, cannabinoids)
most_unique_cannabinoids_2022 = flower_2022.loc[flower_2022['cannabinoid_diversity'] == flower_2022['cannabinoid_diversity'].max()]
print("\n === 2022 Most Unique Cannabinoid Profile ===")
print(f"Predicted winner: {most_unique_cannabinoids_2022['product_name'].iloc[0]}")
print('Official winner: Emerald Spirit Botanicals – Pink Boost Goddess')
winner_cannabinoid_diversity_2022 = flower_2022.loc[flower_2022['entry_name'].astype(str).str.contains('Pink Boost Goddess')]

# (2023) Most Unique Cannabinoid Profile – Flower	Emerald Spirit Botanicals – Pink Boost Goddess – 1:1 THC:THCV
flower_2023['cannabinoid_diversity'] = calculate_shannon_diversity(flower_2023, cannabinoids)
most_unique_cannabinoids_2023 = flower_2023.loc[flower_2023['cannabinoid_diversity'] == flower_2023['cannabinoid_diversity'].max()]
print("\n === 2022 Most Unique Cannabinoid Profile ===")
print(f"Predicted winner: {most_unique_cannabinoids_2023['product_name'].iloc[0]}")
print('Official winner: Emerald Spirit Botanicals – Pink Boost Goddess')
winner_cannabinoid_diversity_2023 = flower_2023.loc[flower_2023['entry_name'].astype(str).str.contains('Pink Boost Goddess')]

# 2022 Runner-Up
sorted_cannabinoid_2022 = flower_2022.sort_values(by='cannabinoid_diversity', ascending=False)
runner_up_cannabinoids_2022 = sorted_cannabinoid_2022.iloc[2]  # Second item after sorting

# 2023 Runner-Up
sorted_cannabinoid_2023 = flower_2023.sort_values(by='cannabinoid_diversity', ascending=False)
runner_up_cannabinoids_2023 = sorted_cannabinoid_2023.iloc[2]  # Second item after sorting

print("2022 Runner-Up for Cannabinoid Diversity:", runner_up_cannabinoids_2022['product_name'])
print("2023 Runner-Up for Cannabinoid Diversity:", runner_up_cannabinoids_2023['product_name'])

# Create a histogram for cannabinoid diversity
plt.figure(figsize=(15, 8))
labels = ['2022', '2023']
plt.hist([flower_2022['cannabinoid_diversity'], flower_2023['cannabinoid_diversity']], bins=30, alpha=0.7, label=labels)
def annotate_cannabinoid(data, color, ypos, year):
    plt.axvline(data['cannabinoid_diversity'], color=color, linestyle='dashed', linewidth=2)
    plt.text(
        data['cannabinoid_diversity'], ypos, f"{year}: {data['product_name']}",
        fontdict={'size': 20}, ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7)
    )
annotate_cannabinoid(most_unique_cannabinoids_2022.iloc[0], 'blue', 5, '2022 Winner')
annotate_cannabinoid(most_unique_cannabinoids_2023.iloc[0], 'red', 10, '2023 Winner')
annotate_cannabinoid(runner_up_cannabinoids_2022, 'cornflowerblue', 15, '2022 Runner-Up')
annotate_cannabinoid(runner_up_cannabinoids_2023, 'salmon', 20, '2023 Runner-Up')
plt.xlabel('Cannabinoid Diversity Index', fontsize=24)
plt.ylabel('Observations', fontsize=24)
plt.title('Cannabinoid Diversity of Cannabis Flowers (Emerald Cup 2022 vs 2023)', fontsize=26)
plt.legend(fontsize=20)
plt.savefig('./figures/emerald-cup-cannabinoid-diversity-histogram.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

# Depict the most unique strain's cannabinoids beside the average.
minor_cannabinoids = [x for x in cannabinoids if x != 'thca']
for cannabinoid in minor_cannabinoids:
    flower_2022[cannabinoid] = pd.to_numeric(flower_2022[cannabinoid], errors='coerce')
    flower_2023[cannabinoid] = pd.to_numeric(flower_2023[cannabinoid], errors='coerce')
avg_cannabinoids_2022 = flower_2022[minor_cannabinoids].mean()
avg_cannabinoids_2022.fillna(0, inplace=True)
avg_cannabinoids_2023 = flower_2023[minor_cannabinoids].mean()
avg_cannabinoids_2023.fillna(0, inplace=True)
unique_cannabinoids_2022 = pd.to_numeric(most_unique_cannabinoids_2022[minor_cannabinoids].iloc[0], errors='coerce')
unique_cannabinoids_2023 = pd.to_numeric(most_unique_cannabinoids_2023[minor_cannabinoids].iloc[0], errors='coerce')
unique_cannabinoids_2022.fillna(0, inplace=True)
unique_cannabinoids_2023.fillna(0, inplace=True)
runner_up_cannabinoids_2022_data = pd.to_numeric(runner_up_cannabinoids_2022[minor_cannabinoids], errors='coerce').fillna(0)
runner_up_cannabinoids_2023_data = pd.to_numeric(runner_up_cannabinoids_2023[minor_cannabinoids], errors='coerce').fillna(0)

# Cannabinoid profile figure.
fig, ax = plt.subplots(2, 1, figsize=(15, 12))
bar_width = 0.25
bar_l = np.arange(len(minor_cannabinoids))
tick_pos = bar_l + bar_width / 2
ax[0].bar(bar_l, unique_cannabinoids_2022, width=bar_width, label='Pink Boost Goddess', color='orange')
ax[0].bar(bar_l + bar_width, runner_up_cannabinoids_2022_data, width=bar_width, label='Mendo Purps', color='green')
ax[0].bar(bar_l + bar_width * 2, avg_cannabinoids_2022, width=bar_width, label='2022 Average', color='blue')
ax[0].set_xticks(tick_pos)
ax[0].set_xticklabels(minor_cannabinoids, rotation=45, ha='right', fontsize=14)
ax[0].set_title('2022 Most Unique Cannabinoid Profile', fontsize=20)
ax[0].legend(fontsize=14)
ax[1].bar(bar_l, unique_cannabinoids_2023, width=bar_width, label='Pink Boost Goddess', color='violet')
ax[1].bar(bar_l + bar_width, runner_up_cannabinoids_2023_data, width=bar_width, label='Joyful & Present PR', color='mediumvioletred')
ax[1].bar(bar_l + bar_width * 2, avg_cannabinoids_2023, width=bar_width, label='2023 Average', color='darkviolet')
ax[1].set_xticks(tick_pos)
ax[1].set_xticklabels(minor_cannabinoids, rotation=45, ha='right', fontsize=14)
ax[1].set_title('2023 Most Unique Cannabinoid Profile', fontsize=20)
ax[1].legend(fontsize=14)
plt.tight_layout()
plt.savefig('./figures/emerald-cup-cannabinoid-diversity-winner.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

# Calculate the percent change in cannabinoid diversity in the winner.
yoy_change = (most_unique_cannabinoids_2023['cannabinoid_diversity'].iloc[0] - most_unique_cannabinoids_2022['cannabinoid_diversity'].iloc[0]) / most_unique_cannabinoids_2022['cannabinoid_diversity'].iloc[0]
print(f'YoY change in cannabinoid diversity of the winner: {round(yoy_change * 100, 2)}%')


# === Image management ===

# Download the images to an image dir.
image_files = []
image_dir = 'D://data/california/lab_results/images'
for index, row in pd.concat([emerald_2022, emerald_2023]).iterrows():
    images = ast.literal_eval(row['images'])
    if images:
        coa_id = row['coa_id'].split('-')[0].strip()
        product_name = row['product_name']
        slug = snake_case(product_name)
        filename = f'{image_dir}/{coa_id}-{slug}.jpg'
        if os.path.exists(filename):
            image_files.append(filename)
            continue
        image_url = images[0]['url']
        response = requests.get(image_url)
        if response.status_code == 200:
            print(f"Downloaded: {image_url}")
            with open(filename, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download: {image_url}")
        image_files.append(filename)
        sleep(1)

# Crop the images.
cropped_images = []
for image_file in image_files:
    cropped_file = image_file.replace('.jpg', '-cropped.png')
    if os.path.exists(cropped_file):
        cropped_images.append(cropped_file)
        continue
    remove_bg(image_file, cropped_file)
    cropped_images.append(cropped_file)
    print(f'Cropped: {cropped_file}')


# === Color Analysis ===

def calculate_purpleness(rgb):
    # Purple is dominant in red and blue channels, and low in green.
    # You might need to adjust the formula depending on your specific shade of purple
    return (rgb[0] + rgb[2]) - 2*rgb[1]


def calculate_colorfulness(image):
    # Split the image into its respective RGB components.
    (R, G, B) = cv2.split(image.astype("float"))

    # Compute rg = R - G
    rg = np.absolute(R - G)

    # Compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    # Compute the mean and standard deviation of both `rg` and `yb`.
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    # Combine the mean and standard deviation.
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    # Derive the "colorfulness" metric and return it.
    return stdRoot + (0.3 * meanRoot)


# Get a measure of color for each image.
color_scores = {}
for image_file in cropped_images:
    coa_id = image_file.split('/')[-1].split('-')[0]
    image = cv2.imread(image_file)
    cropped_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean_color = cropped_img_rgb.mean(axis=0).mean(axis=0)
    color_scores[coa_id] = mean_color
    print(f'Color score for {coa_id}: {mean_color}')

# Assign purple scores.
purple_scores = {}
for image_file in cropped_images:
    coa_id = image_file.split('/')[-1].split('-')[0]
    image = cv2.imread(image_file)
    cropped_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean_color = cropped_img_rgb.mean(axis=0).mean(axis=0)
    # color_scores[coa_id] = mean_color
    purple_scores[coa_id] = calculate_purpleness(mean_color)
    print(f'Color score for {coa_id}: {mean_color}, Purpleness: {purple_scores[coa_id]}')

# Assign colorfulness scores.
colorfulness_scores = {}
for image_file in cropped_images:
    coa_id = image_file.split('/')[-1].split('-')[0]
    image = cv2.imread(image_file)
    cropped_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    colorfulness = calculate_colorfulness(cropped_img_rgb)
    colorfulness_scores[coa_id] = colorfulness
    print(f'Color score for {coa_id}: {colorfulness_scores[coa_id]}')


# === Color Analysis Visualizations ===

# Merge color scores with the data.
flower_2022['id'] = flower_2022['coa_id'].apply(lambda x: x.split('-')[0].strip())
flower_2023['id'] = flower_2023['coa_id'].apply(lambda x: x.split('-')[0].strip())
flower_2022['purpleness'] = flower_2022['id'].map(purple_scores)
flower_2023['purpleness'] = flower_2023['id'].map(purple_scores)
flower_2022['color_score'] = flower_2022['id'].map(color_scores)
flower_2023['color_score'] = flower_2023['id'].map(color_scores)
flower_2022['colorfulness_score'] = flower_2022['id'].map(colorfulness_scores)
flower_2023['colorfulness_score'] = flower_2023['id'].map(colorfulness_scores)

# Re-identify the winners.
highest_terpene_content_flower_2022 = flower_2022.loc[flower_2022['total_terpenes'] == flower_2022['total_terpenes'].max()]
most_unique_terpenes_2022 = flower_2022.loc[flower_2022['terpene_diversity'] == flower_2022['terpene_diversity'].max()]
winner_terpene_diversity_2022 = flower_2022.loc[flower_2022['entry_name'] == 'Atrium Cultivation – Juice Z']
winner_cannabinoid_diversity_2022 = flower_2022.loc[flower_2022['entry_name'].astype(str).str.contains('Pink Boost Goddess')]
highest_terpene_content_flower_2023 = flower_2023.loc[flower_2023['total_terpenes'] == flower_2023['total_terpenes'].max()]
most_unique_terpenes_2023 = flower_2023.loc[flower_2023['terpene_diversity'] == flower_2023['terpene_diversity'].max()]
winner_terpene_diversity_2023 = flower_2023.loc[flower_2023['entry_name'] == 'Brett Byrd – Lilac Mintz 45-159 Live Rosin – Phytofacts Report']
winner_cannabinoid_diversity_2023 = flower_2023.loc[flower_2023['entry_name'].astype(str).str.contains('Pink Boost Goddess')]


# === Cannlytics Purple Award ===

# Top 5 most purple products in 2022.
purple = flower_2022.sort_values(by='purpleness', ascending=False)
purple = purple.loc[purple['purpleness'] < 5]
print('Top 10 Most Purple Flowers 2022')
print(purple.head(5)[['coa_id', 'product_name', 'purpleness']])
most_purple = purple.loc[purple['purpleness'] == purple['purpleness'].max()]
print(f'Most Purple Flower 2022: {most_purple.iloc[0]["product_name"]}, {most_purple.iloc[0]["purpleness"]}')

# Top 5 most purple products in 2023.
purple = flower_2023.sort_values(by='purpleness', ascending=False)
purple = purple.loc[purple['purpleness'] < 5]
print('Top 10 Most Purple Flowers 2023')
print(purple.head(5)[['coa_id', 'product_name', 'purpleness']])
most_purple = purple.loc[purple['purpleness'] == purple['purpleness'].max()]
print(f'Most Purple Flower 2023: {most_purple.iloc[0]["product_name"]}, {most_purple.iloc[0]["purpleness"]}')

# The Cannlytics Most Purple Flower Award
filtered_flower_2023 = flower_2023.loc[flower_2023['purpleness'] <= 5]
most_purple_2022 = flower_2022.loc[flower_2022['purpleness'] == flower_2022['purpleness'].max()]
most_purple_2023 = filtered_flower_2023.loc[filtered_flower_2023['purpleness'] == filtered_flower_2023['purpleness'].max()]
print(f'Most Purple Flower 2022: {most_purple_2022.iloc[0]["product_name"]}, {most_purple_2022.iloc[0]["purpleness"]}')
print(f'Most Purple Flower 2023: {most_purple_2023.iloc[0]["product_name"]}, {most_purple_2023.iloc[0]["purpleness"]}')

# Visualize the purpleness scores.
plt.figure(figsize=(15, 8))
plt.hist(
    flower_2022['purpleness'].loc[flower_2022['purpleness'] <= 5],
    bins=50,
    alpha=0.55,
    label='2022',
    color='violet',
)
plt.hist(
    flower_2023['purpleness'].loc[flower_2023['purpleness'] <= 5],
    bins=50,
    alpha=0.55,
    label='2023',
    color='darkviolet',
)
x_2022_purple = most_purple_2022['purpleness'].values[0]
plt.axvline(x_2022_purple, color='violet', linestyle='dashed', linewidth=2)
plt.text(
    x_2022_purple, 5, f"2022: {most_purple_2022['product_name'].values[0]}",
    fontdict={'size': 20}, ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7)
)
x_2023_purple = most_purple_2023['purpleness'].values[0]
plt.axvline(x_2023_purple, color='darkviolet', linestyle='dashed', linewidth=2)
plt.text(
    x_2023_purple, 10, f"2023: {most_purple_2023['product_name'].values[0]}",
    fontdict={'size': 20}, ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7)
)
plt.xlabel('Purpleness Score', fontsize=20)
plt.ylabel('Observations', fontsize=20)
plt.title('Flower Purpleness (Emerald Cup 2022 vs 2023)', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=18)
plt.savefig('./figures/emerald-cup-purple-scores.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

# Output tables for latex.
columns_to_output = ['Image', 'Rank', 'product_name', 'purpleness']
format_dict = {'Purpleness': '{:.2f}'}
top_purple_2022 = flower_2022.sort_values(by='purpleness', ascending=False).head(5)
top_purple_2022['Rank'] = range(1, 6)
top_purple_2022['Image'] = top_purple_2022['coa_id'].apply(lambda x: f"\\includegraphics[width=0.15\\textwidth]{{images/{x.split('-')[0].strip()}.png}}")
top_purple_2022 = top_purple_2022[columns_to_output]
top_purple_2022.columns = ['Image', 'Rank', 'Product Name', 'Purpleness']
top_purple_2022 = top_purple_2022.to_latex(
    index=False, 
    escape=False,
    column_format='clcc',
    formatters=format_dict,
)
print(top_purple_2022)

top_purple_2023 = flower_2023.sort_values(by='purpleness', ascending=False).head(6)
top_purple_2023['Rank'] = range(0, 6)
top_purple_2023['Image'] = top_purple_2023['coa_id'].apply(lambda x: f"\\includegraphics[width=0.15\\textwidth]{{images/{x.split('-')[0].strip()}.png}}")
top_purple_2023 = top_purple_2023[columns_to_output]
top_purple_2023.columns = ['Image', 'Rank', 'Product Name', 'Purpleness']
latex_table_2023 = top_purple_2023.to_latex(
    index=False, 
    escape=False,
    column_format='clcc',
    formatters=format_dict,
)
print(latex_table_2023)



# === Cannlytics Green Green Award ===

# Sorting by purpleness in ascending order to find the most green products in 2022
green_2022 = flower_2022.sort_values(by='purpleness', ascending=True)
print('Top 10 Most Green Flowers 2022')
print(green_2022.head(10)[['coa_id', 'product_name', 'purpleness']])

# Identifying the most green flower in 2022
most_green_2022 = green_2022.iloc[0]
print(f'Most Green Flower 2022: {most_green_2022["product_name"]}, Purpleness: {most_green_2022["purpleness"]}')

# Sorting by purpleness in ascending order to find the most green products in 2023
green_2023 = flower_2023.sort_values(by='purpleness', ascending=True)
print('Top 10 Most Green Flowers 2023')
print(green_2023.head(10)[['coa_id', 'product_name', 'purpleness']])

# Identifying the most green flower in 2023
most_green_2023 = green_2023.iloc[0]
print(f'Most Green Flower 2023: {most_green_2023["product_name"]}, Purpleness: {most_green_2023["purpleness"]}')

# Visualize most green.
plt.figure(figsize=(15, 8))
plt.hist(
    flower_2022['purpleness'].loc[flower_2022['purpleness'] <= 5],
    bins=100,
    alpha=0.55,
    label='2022',
    color='mediumseagreen',
)
plt.hist(
    flower_2023['purpleness'].loc[flower_2023['purpleness'] <= 5],
    bins=100,
    alpha=0.55,
    label='2023',
    color='forestgreen',
)

# Annotate the most green flower for each year
plt.axvline(most_green_2022['purpleness'], color='mediumseagreen', linestyle='dashed', linewidth=2)
plt.text(
    most_green_2022['purpleness'], 5, f"2022: {most_green_2022['product_name']}",
    fontdict={'size': 20}, ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7)
)
plt.axvline(most_green_2023['purpleness'], color='forestgreen', linestyle='dashed', linewidth=2)
plt.text(
    most_green_2023['purpleness'], 10, f"2023: {most_green_2023['product_name']}",
    fontdict={'size': 20}, ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7)
)
plt.xlabel('Purpleness Score', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.title('Flower Greenness (Emerald Cup 2022 vs 2023)', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=18)
plt.savefig('./figures/emerald-cup-greenness-scores.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

# Top 5 greenest (least purple) products in 2022
top_green_2022 = green_2022.head(5)
top_green_2022['Rank'] = range(1, 6)
top_green_2022['Image'] = top_green_2022['coa_id'].apply(lambda x: f"\\includegraphics[width=0.15\\textwidth]{{images/{x.split('-')[0].strip()}.png}}")
top_green_2022 = top_green_2022[['Image', 'Rank', 'product_name', 'purpleness']]
top_green_2022.columns = ['Image', 'Rank', 'Product Name', 'Greenness']
format_dict_green = {'Greenness': '{:.2f}'}
latex_table_green_2022 = top_green_2022.to_latex(index=False, escape=False, column_format='lcL{4cm}r', formatters=format_dict_green)

# Top 5 greenest (least purple) products in 2023
top_green_2023 = green_2023.head(5)
top_green_2023['Rank'] = range(1, 6)
top_green_2023['Image'] = top_green_2023['coa_id'].apply(lambda x: f"\\includegraphics[width=0.15\\textwidth]{{images/{x.split('-')[0].strip()}.png}}")
top_green_2023 = top_green_2023[['Image', 'Rank', 'product_name', 'purpleness']]
top_green_2023.columns = ['Image', 'Rank', 'Product Name', 'Greenness']
latex_table_green_2023 = top_green_2023.to_latex(index=False, escape=False, column_format='lcL{4cm}r', formatters=format_dict_green)

# Print or save the LaTeX tables
print("=== Top 5 Greenest Flowers 2022 ===")
print(latex_table_green_2022)
print("\n=== Top 5 Greenest Flowers 2023 ===")
print(latex_table_green_2023)


# === Colorfulness ====

# The Cannlytics Most Colorful Flower Award
filtered_flower_2023 = flower_2023.loc[flower_2023['colorfulness_score'] <= 45]
most_colorful_2022 = flower_2022.loc[flower_2022['colorfulness_score'] == flower_2022['colorfulness_score'].max()]
most_colorful_2023 = filtered_flower_2023.loc[filtered_flower_2023['colorfulness_score'] == filtered_flower_2023['colorfulness_score'].max()]
print(f'Most Colorful Flower 2022: {most_colorful_2022.iloc[0]["product_name"]}, Colorfulness Score: {most_colorful_2022.iloc[0]["colorfulness_score"]}')
print(f'Most Colorful Flower 2023: {most_colorful_2023.iloc[0]["product_name"]}, Colorfulness Score: {most_colorful_2023.iloc[0]["colorfulness_score"]}')

# Visualize the colorfulness scores.
plt.figure(figsize=(15, 8))
plt.hist(
    flower_2022['colorfulness_score'].loc[flower_2022['colorfulness_score'] <= 60],
    bins=50,
    alpha=0.55,
    label='2022',
    color='darkviolet'
)
plt.hist(
    flower_2023['colorfulness_score'].loc[flower_2023['colorfulness_score'] <= 45],
    bins=50,
    alpha=0.55,
    label='2023',
    color='darkorange',
)
x_2022_colorful = most_colorful_2022['colorfulness_score'].max()
plt.axvline(x_2022_colorful, color='darkviolet', linestyle='dashed', linewidth=2)
plt.text(
    x_2022_colorful, 5, f"2022: {most_colorful_2022['product_name'].values[0]}",
    fontdict={'size': 20}, ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7)
)
x_2023_colorful = most_colorful_2023['colorfulness_score'].max()
plt.axvline(x_2023_colorful, color='darkorange', linestyle='dashed', linewidth=2)
plt.text(
    x_2023_colorful, 10, f"2023: {most_colorful_2023['product_name'].values[0]}",
    fontdict={'size': 20}, ha='right', va='center', bbox=dict(facecolor='white', alpha=0.7)
)
plt.xlabel('Colorfulness Score', fontsize=20)
plt.ylabel('Observations', fontsize=20)
plt.title('Flower Colorfulness (Emerald Cup 2022 vs 2023)', fontsize=22)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=18)
plt.savefig('./figures/emerald-cup-colorfulness-scores.png', dpi=300, bbox_inches='tight', transparent=False)
plt.show()

# Top 5 most colorful products in 2022
top_colorful_2022 = flower_2022.sort_values(by='colorfulness_score', ascending=False).head(5)
top_colorful_2022['Rank'] = range(1, 6)
top_colorful_2022['Image'] = top_colorful_2022['coa_id'].apply(lambda x: f"\\includegraphics[width=0.15\\textwidth]{{images/{x.split('-')[0].strip()}.png}}")
top_colorful_2022 = top_colorful_2022[['Image', 'Rank', 'product_name', 'colorfulness_score']]
top_colorful_2022.columns = ['Image', 'Rank', 'Product Name', 'Colorfulness Score']
format_dict_colorful = {'Colorfulness Score': '{:.2f}'}
latex_table_colorful_2022 = top_colorful_2022.to_latex(index=False, escape=False, column_format='lcL{4cm}r', formatters=format_dict_colorful)

# Top 5 most colorful products in 2023
top_colorful_2023 = flower_2023.sort_values(by='colorfulness_score', ascending=False).head(7)[2:]
top_colorful_2023['Rank'] = range(1, 6)
top_colorful_2023['Image'] = top_colorful_2023['coa_id'].apply(lambda x: f"\\includegraphics[width=0.15\\textwidth]{{images/{x.split('-')[0].strip()}.png}}")
top_colorful_2023 = top_colorful_2023[['Image', 'Rank', 'product_name', 'colorfulness_score']]
top_colorful_2023.columns = ['Image', 'Rank', 'Product Name', 'Colorfulness Score']
latex_table_colorful_2023 = top_colorful_2023.to_latex(index=False, escape=False, column_format='lcL{4cm}r', formatters=format_dict_colorful)

# Print or save the LaTeX tables
print("=== Top 5 Most Colorful Flowers 2022 ===")
print(latex_table_colorful_2022)
print("\n=== Top 5 Most Colorful Flowers 2023 ===")
print(latex_table_colorful_2023)
