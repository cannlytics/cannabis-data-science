"""
Product Analysis | Cannabis Data Science
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 12/7/2023
Updated: 12/7/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Data:

    - [MA Metrc Product Catalog 2023-12-06](https://cannlytics.page.link/ma-products-2023-12-06T01-56-39)
    - [Augmented Dataset](https://cannlytics.page.link/augmented-ma-products-2023-12-07)
    - [License DBA Mapping](https://cannlytics.page.link/reviewed-dba-mappings)
    - [MA Licenses](https://huggingface.co/datasets/cannlytics/cannabis_licenses/tree/main/data/ma)

"""
# Standard imports:
from datetime import datetime
import json
import os

# External imports:
from dotenv import dotenv_values
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from openai import OpenAI
import pandas as pd


# === Read the product data ===

# Explore the data.
df = pd.read_csv('./data/ma-products-2023-12-06T01-56-39.csv')

# Merge with license data.
ma_licenses = pd.read_csv('./data/ma/licenses-ma-2023-08-13T13-35-31.csv')
merged = pd.merge(
    left=df,
    right=ma_licenses,
    how='left',
    on='license_number',
)

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})


# === Visualize products by tier ===

# Visualize products by tier.
group = merged.groupby(['cultivation_tier', 'producer']).size().unstack()
average_skus = group.mean(axis=1)
average_skus.plot(kind='barh')
plt.xlabel('Average Number of SKUs')
plt.title('Average Number of SKUs by Cultivation Tier in MA', pad=20)
plt.ylabel('')
plt.show()


# === Augment Producer DBA Names ===

# Initialize OpenAI client.
env_file = '../../.env'
os.environ['OPENAI_API_KEY'] = dotenv_values(env_file)['OPENAI_API_KEY']
client = OpenAI()

# Use OpenAI to create a column of DBA names.
names = list(df['producer'].unique())
prompt = 'Return JSON. Given a list of business names, return a list of DBA names, e.g. {"names": []}. Try to use a standard DBA when there are name variations, e.g. "Curaleaf Massachusetts, Inc". and "Curaleaf North Shore, Inc." have the same DBA name "Curaleaf". Please make sure the DBA names are in the same order as the business names and there are just as many DBA names as business names. Remove terms like "LLC" and "Inc" from the business names.'
prompt += '\n\nBusiness Names:\n'
prompt += '\n'.join(names)
prompt += '\n\nDBA Names:\n'
messages = [{'role': 'user', 'content': prompt}]
completion = client.chat.completions.create(
    model='gpt-4-1106-preview',
    messages=messages,
    max_tokens=4_096,
    temperature=0.0,
    user='cannlytics',
)
usage = completion.model_dump()['usage']
cost = 0.01 / 1_000 * usage['prompt_tokens'] + 0.03 / 1_000 * usage['completion_tokens']
content = completion.choices[0].message.content
extracted_json = content.split('```json\n')[-1].split('\n```')[0]
extracted_data = json.loads(extracted_json)
print('Cost:', cost)
print('Extracted:', extracted_data)

# Save the license to DBA mapping.
dba = extracted_data['names']
llc_to_dba = {llc: dba for llc, dba in zip(names, dba)}
mapping = pd.DataFrame({
    'producer': names,
    'dba': dba,
})
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
mapping.to_excel(f'./data/dba-mappings-{timestamp}.xlsx', index=False)


# === Human-in-the-middle review of DBAs ===

# Merge DBAs with product data.
reviewed_mappings = pd.read_excel('./data/reviewed-dba-mappings.xlsx')
reviewed_llc_to_dba = {llc: dba for llc, dba in zip(reviewed_mappings['producer'], reviewed_mappings['dba'])}
df['producer_dba_name'] = df['producer'].map(reviewed_llc_to_dba)

# Save the augmented data.
timestamp = datetime.now().strftime('%Y-%m-%d')
df.to_csv(f'./data/augmented-ma-products-{timestamp}.csv', index=False)


# === Production by DBA ===

# Look at production by DBA.
top = 5
sku_count = df['producer_dba_name'].value_counts()
sku_share = sku_count.div(len(df)).mul(100).round(2).to_dict()
sorted_data = dict(sorted(sku_share.items(), key=lambda x: x[1], reverse=True))
top_10 = dict(list(sorted_data.items())[:top])
other = sum(list(sorted_data.values())[top:])
top_10['Other'] = other
labels = top_10.keys()
sizes = top_10.values()
plt.figure(figsize=(12.5, 12.5))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Market Share of Top MA Producers')
plt.show()

# Visualize market share as a bar chart.
sorted_data = dict(sorted(sku_count.to_dict().items(), key=lambda x: x[1], reverse=True))
labels = list(sorted_data.keys())
sizes = list(sorted_data.values())
plt.figure(figsize=(15, 10))
plt.bar(labels, sizes, color='skyblue')
plt.xlabel('Producers')
plt.ylabel('Number of SKUs')
plt.title('SKUs by Producer in MA')
plt.xticks([])
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.show()


# === Product type analysis ===

# Visualize product types proportions.
stats = df['product_type'].value_counts().to_dict()
plt.figure(figsize=(8, 6))
plt.pie(stats.values(), labels=stats.keys(), autopct='%1.1f%%', startangle=140)
plt.title('Product Types in MA')
plt.show()


# === Product chemical analysis ===

# Visualize total THC vs. total CBD for each product type.
product_types = [
    'Buds',
    'Raw Pre-Rolls',
    'Infused (edible)',
    'Vape Product',
    'Concentrate (Each)'
]
for product_type in product_types:
    flower = (product_type == 'Buds') | (product_type == 'Raw Pre-Rolls')
    units = 'g' if flower else 'mg'
    maximum = 100 if flower else 1_000
    divisor = 1 if flower else 10
    sample = df.loc[
        (~df['total_thc'].isna()) &
        (~df['total_cbd'].isna()) &
        (df['total_thc_units'] == units)
    ]
    sample = sample.loc[
        (df['product_type'] == product_type)
    ]
    sample = sample.loc[
        (sample['total_thc'] < maximum) &
        (sample['total_cbd'] < maximum)
    ]
    # sample['total_thc'] = sample['total_thc'].div(divisor)
    # sample['total_cbd'] = sample['total_cbd'].div(divisor)
    plt.figure(figsize=(10, 6))
    plt.scatter(
        sample['total_cbd'][sample['total_thc'] + sample['total_cbd'] <= 100], 
        sample['total_thc'][sample['total_thc'] + sample['total_cbd'] <= 100], 
        alpha=0.5,
        color='blue',
    )
    plt.scatter(
        sample['total_cbd'][sample['total_thc'] + sample['total_cbd'] > 100], 
        sample['total_thc'][sample['total_thc'] + sample['total_cbd'] > 100], 
        alpha=0.5,
        color='red',
    )
    plt.title(f'Total THC vs Total CBD in {product_type} in MA')
    plt.ylabel('Total THC (mg)')
    plt.xlabel('Total CBD (mg)')
    plt.grid(True)
    plt.show()


# === Organic edible analysis ===

def is_organic(row):
    """Find edibles with "organic" in the ingredients pr product name, case insensitively."""
    product_name = str(row['product_name']).lower()
    ingredients = str(row['ingredients']).lower()
    return 'Organic' if 'organic' in product_name or 'organic' in ingredients else 'Non-Organic'


# Find edibles with "organic" in the ingredients, case insensitive
edibles = df[df['product_type'] == 'Infused (edible)']
edibles['edible_type'] = edibles.apply(is_organic, axis=1)
edible_counts = edibles.groupby(['producer_dba_name', 'edible_type']).size().unstack(fill_value=0)
top_10_edible_counts = edible_counts.sum(axis=1).sort_values(ascending=False).head(10)
top_10_edible_counts = edible_counts.loc[top_10_edible_counts.index]
top_10_edible_counts.plot(
    kind='barh',
    figsize=(12, 10),
    color=['skyblue', 'lightgreen',]
)
plt.xlabel('Count')
plt.ylabel('')
plt.title('Products Labelled Organic or with Organic Ingredients by Top 10 Producers in MA')
plt.legend(['Non-organic', 'Organic',])
plt.show()


# === Vape product analysis ===

# Find vape products with "terpene" in the ingredients, case insensitive
vapes = df[df['product_type'] == 'Vape Product']
terpene_vapes = vapes[vapes['ingredients'].str.contains('terpene', case=False, na=False)]
regular_vapes = vapes[~vapes['ingredients'].str.contains('terpene', case=False, na=False)]

# Visualize products with added terpenes.
vapes = df[df['product_type'] == 'Vape Product']
vapes['vape_type'] = vapes['ingredients'].apply(lambda x: 'Flavored (Terpene)' if 'terpene' in str(x).lower() else 'Unflavored')
vape_counts = vapes.groupby(['producer_dba_name', 'vape_type']).size().unstack(fill_value=0)
top_10_vape_counts = vape_counts.sum(axis=1).sort_values(ascending=False).head(10)
top_10_vape_counts = vape_counts.loc[top_10_vape_counts.index]
top_10_vape_counts.plot(
    kind='barh',
    figsize=(12, 10),
    color=['lightgreen', 'skyblue']
)
plt.xlabel('Count')
plt.ylabel('')
plt.title('Flavored vs Unflavored Vape Products by Top 10 Producers in MA')
plt.legend(['Flavored', 'Unflavored'])
plt.show()
