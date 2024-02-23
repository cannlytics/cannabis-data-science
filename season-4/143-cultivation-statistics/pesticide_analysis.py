"""

Data Sources:

    - [February 2023 CCRS Traceability Report](https://lcb.box.com/s/l9rtua9132sqs63qnbtbw13n40by0yml)
    - [March 2023 CCRS Traceability Report](https://lcb.box.com/s/lg50ow8qx2xki2d4lr6raj0c2r22v711)
    - [April 2023 CCRS Traceability Report](https://lcb.box.com/s/bj3g5inm77n8mrf7gk0h07f1o13dkfc7)
    - [May 2023 CCRS Traceability Report](https://lcb.box.com/s/dzlcx9uzt3t1td8enzbtbgknw6oh9bzw)
    - [June 2023 CCRS Traceability Report](https://lcb.box.com/s/d0g3mhtdyohhi4ic3zucekpnz017fy9o)
    - [July 2023 CCRS Traceability Report](https://lcb.box.com/s/plb3dr2fvsuvgixb38g10tbwqos73biz)
    - [August 2023 CCRS Traceability Report](https://lcb.box.com/s/59jw6qdt7sbg36g0xa2vw0ysr8us8cpo)
    - [September 2023 CCRS Traceability Report](https://lcb.box.com/s/59jw6qdt7sbg36g0xa2vw0ysr8us8cpo)
    - [October 2023 monthly CCRS traceability data report](https://lcb.box.com/s/qt9xd2oqp2wqqz4xuppzuphhjvwfm67j)
    - [November 2023 CCRS Traceability Report](https://lcb.box.com/s/pr8razl8bs3lu74ayk1d8a7iq8padhy1)
    - [December 2023 CCRS Traceability Report](https://lcb.app.box.com/s/4vweufdqsmg41t2zadr56r4dcwqmvlit)

References:

    - [WSLCB Guidance Sheets](https://lcb.box.com/s/n5f1eyybvjxfs8w49y4ztlqyzgd4842d)

"""
import os
import pandas as pd
from cannlytics.data.ccrs import (
    CCRS_ANALYTES,
    CCRS_DATASETS,
    get_datafiles,
    merge_datasets,
)


# Define the prohibited pesticides.
pesticides = {
    "abamectin": {"name": "Abamectin (Sum of Isomers)", "limit": 0.50, "cas": "71751-41-2"},
    "avermectin_b1a": {"name": "Avermectin B1a", "limit": None, "cas": "65195-55-3"},
    "avermectin_b1b": {"name": "Avermectin B1b", "limit": None, "cas": "65195-56-4"},
    "acephate": {"name": "Acephate", "limit": 0.40, "cas": "30560-19-1"},
    "acequinocyl": {"name": "Acequinocyl", "limit": 2.0, "cas": "57960-19-7"},
    "acetamiprid": {"name": "Acetamiprid", "limit": 0.20, "cas": "135410-20-7"},
    "aldicarb": {"name": "Aldicarb", "limit": 0.40, "cas": "116-06-3"},
    "azoxystrobin": {"name": "Azoxystrobin", "limit": 0.20, "cas": "131860-33-8"},
    "bifenazate": {"name": "Bifenazate", "limit": 0.20, "cas": "149877-41-8"},
    "bifenthrin": {"name": "Bifenthrin", "limit": 0.20, "cas": "82657-04-3"},
    "boscalid": {"name": "Boscalid", "limit": 0.40, "cas": "188425-85-6"},
    "carbaryl": {"name": "Carbaryl", "limit": 0.20, "cas": "63-25-2"},
    "carbofuran": {"name": "Carbofuran", "limit": 0.20, "cas": "1563-66-2"},
    "chlorantraniliprole": {"name": "Chlorantraniliprole", "limit": 0.20, "cas": "500008-45-7"},
    "chlorfenapyr": {"name": "Chlorfenapyr", "limit": 1.0, "cas": "122453-73-0"},
    "chlorpyrifos": {"name": "Chlorpyrifos", "limit": 0.20, "cas": "2921-88-2"},
    "clofentezine": {"name": "Clofentezine", "limit": 0.20, "cas": "74115-24-5"},
    "cyfluthrin": {"name": "Cyfluthrin", "limit": 1.0, "cas": "68359-37-5"},
    "cypermethrin": {"name": "Cypermethrin", "limit": 1.0, "cas": "52315-07-8"},
    "daminozide": {"name": "Daminozide", "limit": 1.0, "cas": "1596-84-5"},
    "ddvp_dichlorvos": {"name": "DDVP (Dichlorvos)", "limit": 0.10, "cas": "62-73-7"},
    "diazinon": {"name": "Diazinon", "limit": 0.20, "cas": "333-41-5"},
    "dimethoate": {"name": "Dimethoate", "limit": "0.20 μg/g", "cas": "60-51-5"},
    "ethoprophos": {"name": "Ethoprophos", "limit": "0.20 μg/g", "cas": "13194-48-4"},
    "etofenprox": {"name": "Etofenprox", "limit": "0.40 μg/g", "cas": "80844-07-1"},
    "etoxazole": {"name": "Etoxazole", "limit": "0.20 μg/g", "cas": "153233-91-1"},
    "fenoxycarb": {"name": "Fenoxycarb", "limit": "0.20 μg/g", "cas": "72490-01-8"},
    "fenpyroximate": {"name": "Fenpyroximate", "limit": "0.40 μg/g", "cas": "134098-61-6"},
    "fipronil": {"name": "Fipronil", "limit": "0.40 μg/g", "cas": "120068-37-3"},
    "flonicamid": {"name": "Flonicamid", "limit": "1.0 μg/g", "cas": "158062-67-0"},
    "fludioxonil": {"name": "Fludioxonil", "limit": "0.40 μg/g", "cas": "131341-86-1"},
    "hexythiazox": {"name": "Hexythiazox", "limit": "1.0 μg/g", "cas": "78587-05-0"},
    "imazalil": {"name": "Imazalil", "limit": "0.20 μg/g", "cas": "35554-44-0"},
    "imidacloprid": {"name": "Imidacloprid", "limit": "0.40 μg/g", "cas": "138261-41-3"},
    "kresoxim_methyl": {"name": "Kresoxim-methyl", "limit": "0.40 μg/g", "cas": "143390-89-0"},
    "malathion": {"name": "Malathion", "limit": "0.20 μg/g", "cas": "121-75-5"},
    "metalaxyl": {"name": "Metalaxyl", "limit": "0.20 μg/g", "cas": "57837-19-1"},
    "methiocarb": {"name": "Methiocarb", "limit": "0.20 μg/g", "cas": "2032-65-7"},
    "methomyl": {"name": "Methomyl", "limit": 0.40, "cas": "16752-77-5"},
    "methyl_parathion": {"name": "Methyl Parathion", "limit": 0.20, "cas": "298-00-0"},
    "mgk_264": {"name": "MGK-264", "limit": 0.20, "cas": "113-48-4"},
    "myclobutanil": {"name": "Myclobutanil", "limit": 0.20, "cas": "88671-89-0"},
    "naled": {"name": "Naled", "limit": 0.50, "cas": "300-76-5"},
    "oxamyl": {"name": "Oxamyl", "limit": 1.0, "cas": "23135-22-0"},
    "paclobutrazol": {"name": "Paclobutrazol", "limit": 0.40, "cas": "76738-62-0"},
    "permethrins": {
        "name": "Permethrins (Sum of Isomers)", 
        "limit": 0.20, 
        "cas": "52645-53-1",
        "cis_permethrin": {"name": "cis-Permethrin", "cas": "54774-45-7"},
        "trans_permethrin": {"name": "trans-Permethrin", "cas": "51877-74-8"}
    },
    "phosmet": {"name": "Phosmet", "limit": 0.20, "cas": "732-11-6"},
    "piperonyl_butoxide": {"name": "Piperonyl Butoxide", "limit": 2.0, "cas": "51-03-6"},
    "prallethrin": {"name": "Prallethrin", "limit": 0.20, "cas": "23031-36-9"},
    "propiconazole": {"name": "Propiconazole", "limit": 0.40, "cas": "60207-90-1"},
    "propoxur": {"name": "Propoxur", "limit": 0.20, "cas": "114-26-1"},
    "pyrethrins": {
        "name": "Pyrethrins (Sum of Isomers)", 
        "limit": 1.0, 
        "cas": "8003-34-7",
        "pyrethrin_i": {"name": "Pyrethrin I", "cas": "121-21-1"},
        "pyrethrin_ii": {"name": "Pyrethrin II", "cas": "121-29-9"}
    },
    "pyridaben": {"name": "Pyridaben", "limit": 0.20, "cas": "96489-71-3"},
    "spinosad": {
        "name": "Spinosad (Sum of Isomers)",
        "limit": 0.20,
        "cas": "168316-95-8"
    },
    "spinosyn_a": {
        "name": "Spinosyn A",
        "limit": None,
        "cas": "131929-60-7"
    },
    "spinosyn_d": {
        "name": "Spinosyn D",
        "limit": None,
        "cas": "131929-63-0"
    },
    "spiromesifen": {
        "name": "Spiromesifen",
        "limit": 0.20,
        "cas": "283594-90-1"
    },
    "spirotetramat": {
        "name": "Spirotetramat",
        "limit": 0.20,
        "cas": "203313-25-1"
    },
    "spiroxamine": {
        "name": "Spiroxamine",
        "limit": 0.40,
        "cas": "118134-30-8"
    },
    "tebuconazole": {
        "name": "Tebuconazole",
        "limit": 0.40,
        "cas": "80443-41-0"
    },
    "thiacloprid": {
        "name": "Thiacloprid",
        "limit": 0.20,
        "cas": "111988-49-9"
    },
    "thiamethoxam": {
        "name": "Thiamethoxam",
        "limit": 0.20,
        "cas": "153719-23-4"
    },
    "trifloxystrobin": {
        "name": "Trifloxystrobin",
        "limit": 0.20,
        "cas": "141517-21-7"
    },
}
banned_pesticides = [
    'abamectin',
    'avermectin_b1a',
    'avermectin_b1b',
    'acephate',
    'acequinocyl',
    'acetamiprid',
    'aldicarb',
    'azoxystrobin',
    'bifenazate',
    'bifenthrin',
    'boscalid',
    'carbaryl',
    'carbofuran',
    'chlorantraniliprole',
    'chlorfenapyr',
    'chlorpyrifos',
    'clofentezine',
    'cyfluthrin',
    'cypermethrin',
    'daminozide',
    'ddvp_dichlorvos',
    'diazinon',
    'dimethoate',
    'ethoprophos',
    'etofenprox',
    'etoxazole',
    'fenoxycarb',
    'fenpyroximate',
    'fipronil',
    'flonicamid',
    'fludioxonil',
    'hexythiazox',
    'imazalil',
    'imidacloprid',
    'kresoxim_methyl',
    'malathion',
    'metalaxyl',
    'methiocarb',
    'methomyl',
    'methyl_parathion',
    'mgk_264',
    'myclobutanil',
    'naled',
    'oxamyl',
    'paclobutrazol',
    'permethrins',
    'phosmet',
    'prallethrin',
    'propiconazole',
    'propoxur',
    'pyridaben',
    'spinosad',
    'spinosyn_a',
    'spinosyn_d',
    'spiromesifen',
    'spirotetramat',
    'spiroxamine',
    'tebuconazole',
    'thiacloprid',
    'thiamethoxam',
    'trifloxystrobin'
    # Also tested for:
    'clofentizine',
    'ddvp',
    'dichlorvos',
    # Not banned:
    # 'piperonyl_butoxide',
    # 'pyrethrins',
]


def extract_pesticides(pesticide_list_str):
    if not pesticide_list_str or pesticide_list_str == '[]':
        return []
    try:
        pesticides = eval(pesticide_list_str)
        if isinstance(pesticides, list):
            return pesticides
        return []
    except:
        return []


# === Data Cleaning ===

# Read lab results.
datafile = 'D://data/washington/stats/lab_results/wa-lab-results-aggregate.xlsx'
results = pd.read_excel(datafile)

# Isolate the year.
results['year'] = results['created_date'].apply(lambda x: x.year)

# Drop duplicates.
results = results.drop_duplicates(subset=['inventory_id'])

# DEV: Check for duplicates.
# results['inventory_id'].value_counts().sort_values(ascending=False)

# Clean the pesticides.
results['pesticides'] = results['pesticides'].apply(extract_pesticides)

# Optional: Restrict to a specific year.
# results = results.loc[results['year'] == 2023]


# === Lab Test Analysis ===

# Identify all unique pesticides detected
unique_pesticides = set()
results['pesticides'].apply(lambda x: unique_pesticides.update(x))

# Count the number of banned pesticides detected.
banned_pesticide_counts = sum(pesticide in unique_pesticides for pesticide in banned_pesticides)

# Count the number of batches with a banned pesticide.
results['banned_pesticide'] = results['pesticides'].apply(lambda x: any(pesticide in banned_pesticides for pesticide in x))
batches_with_banned_count = results.groupby('year')['banned_pesticide'].sum()
# print(f"Unique Pesticides Detected: {len(unique_pesticides)}")
print(f"Banned Pesticides Detected: {banned_pesticide_counts}")
print(f"Batches with Banned Pesticide: {batches_with_banned_count}")

# Calculate the change in banned pesticide detections from 2022 to 2023.
banned_pesticide_counts_2022 = batches_with_banned_count.loc[2022]
banned_pesticide_counts_2023 = batches_with_banned_count.loc[2023]
change = banned_pesticide_counts_2023 - banned_pesticide_counts_2022
percent_change = round((change / banned_pesticide_counts_2022) * 100, 2)
print(f"Percent change in banned pesticide detections from 2022 to 2023: {percent_change}")

# Calculate the proportion of batches that contained a banned pesticide.
for year in results['year'].unique():
    year_results = results[results['year'] == year]
    banned_pesticide_count = year_results['banned_pesticide'].sum()
    total_count = len(year_results)
    percent = round((banned_pesticide_count / total_count) * 100, 2)
    print(f"Percent of batches with banned pesticides in {year}: {percent}")

# Count the number of licensees that produced a product with a banned pesticide.
licensee_bad_batches = results.groupby('licensee_id')['banned_pesticide'].sum()
licensees_implicated = len(licensee_bad_batches[licensee_bad_batches > 0])
print(f"Licensees with Banned Pesticide: {licensees_implicated}")
percent = round(licensees_implicated  * 100 / len(results['licensee_id'].unique()), 2)
print('Percent of licensees that produced a product containing a banned pesticide:', percent)

# Calculate the proportion of batches that contained a banned pesticide
# for each licensee that produced a product with a banned pesticide.
proportions = {}
for licensee_id in results['licensee_id'].unique():
    licensee_results = results[results['licensee_id'] == licensee_id]
    banned_pesticide_count = licensee_results['banned_pesticide'].sum()
    total_count = len(licensee_results)
    proportions[licensee_id] = {
        'total': total_count,
        'banned': banned_pesticide_count,
        'proportion': round(banned_pesticide_count / total_count, 2),
    }
proportions = pd.DataFrame(proportions).T
proportions.sort_values(by='proportion', ascending=False, inplace=True)

# Calculate the number of licensees where 100% of their batches contained a banned pesticide.
proportions['all_banned'] = proportions['proportion'].apply(lambda x: x == 1.0)
all_banned = proportions[proportions['all_banned'] == True]
print(f"Licensees with 100% Banned Pesticide: {len(all_banned)}")

# Calculate the most number of batches produced by a licensee with a banned pesticide.
most_banned = proportions.loc[proportions['banned'] == proportions['banned'].max()].iloc[0]
print(f"Most Batches with Banned Pesticides by a Licensee: {most_banned['banned']}")


# === Amount Estimation ===

# Estimate amount of flower produced with banned pesticides.
pounds_per_batch = 3
flower_batches = results.loc[(~results['water_activity'].isna()) & (results['banned_pesticide'] == True)]
pounds_flower_2022 = len(flower_batches.loc[flower_batches['year'] == 2022]) * pounds_per_batch
pounds_flower_2023 = len(flower_batches.loc[flower_batches['year'] == 2023]) * pounds_per_batch
print(f"Estimated Pounds of Flower Produced with Banned Pesticides in 2022: {pounds_flower_2022}")
print(f"Estimated Pounds of Flower Produced with Banned Pesticides in 2023: {pounds_flower_2023}")
percent_change = round(((pounds_flower_2023 - pounds_flower_2022) / pounds_flower_2022) * 100, 2)
print(f"Percent change in pounds of flower produced with banned pesticides from 2022 to 2023: {percent_change}")


# === Quality control ===

def contains_banned(pesticide_list):
    return any(pesticide in banned_pesticides for pesticide in pesticide_list)

# Add a new column to indicate if a banned pesticide was detected in each test
results['banned_pesticide_detected'] = results['pesticides'].apply(contains_banned)

# Exclude labs 2908 and 2907.
sample = results.loc[(results['lab_licensee_id'] != 2908) & (results['lab_licensee_id'] != 2907)]

# Group data by 'lab_id' and calculate the detection rate of banned pesticides
lab_analysis = sample.groupby('lab_licensee_id').agg(
    total_tests=pd.NamedAgg(column='lab_result_id', aggfunc='count'),
    banned_detected=pd.NamedAgg(column='banned_pesticide_detected', aggfunc='sum')
)

# Calculate the detection rate as a percentage of tests.
lab_analysis['detection_rate'] = (lab_analysis['banned_detected'] / lab_analysis['total_tests']) * 100

# Sort the labs by detection rate for better visibility.
lab_analysis.sort_values(by='detection_rate', ascending=False, inplace=True)

# Calculate average detection rate.
print(f"Average Detection Rate: {round(lab_analysis['detection_rate'].mean(), 2)}")

# Calculate detection rates by lab.
print(lab_analysis)


# === Sales analysis ===

# Define all sales fields.
fields = CCRS_DATASETS['sale_details']['fields']
date_fields = CCRS_DATASETS['sale_details']['date_fields']
item_cols = list(fields.keys()) + date_fields
item_types = {k: fields[k] for k in fields if k not in date_fields}
item_types['IsDeleted'] = 'string'

# Find all of the sales with inventory IDs of banned batches.
all_sales = []
banned_ids = results[results['banned_pesticide'] == True]['inventory_id'].unique()
banned_ids = [str(x) for x in banned_ids]
matched_ids = []
base = 'D://data/washington/'
stats_dir = 'D://data/washington/stats'
releases = [
    'CCRS PRR (8-4-23)',
    'CCRS PRR (9-5-23)',
    'CCRS PRR (11-2-23)',
    'CCRS PRR (12-2-23)',
]
for release in reversed(releases):
    data_dir = os.path.join(base, release, release)
    sales_items_files = get_datafiles(data_dir, 'SalesDetail_')
    for datafile in sales_items_files:

        # Read in the sales items.
        items = pd.read_csv(
            datafile,
            sep='\t',
            encoding='utf-16',
            parse_dates=date_fields,
            usecols=item_cols,
            dtype=item_types,
        )

        # Find all sales for banned batches.
        banned_sales = items[items['InventoryId'].isin(banned_ids)]
        if len(banned_sales) > 0:
            print(f"Found {len(banned_sales)} banned sales in {datafile}.")
        
            # Merge headers with the sales details to get licensee IDs.
            sale_headers_files = get_datafiles(data_dir, 'SaleHeader_', desc=False)
            banned_sales = merge_datasets(
                banned_sales,
                sale_headers_files,
                dataset='sale_headers',
                on='SaleHeaderId',
                target='LicenseeId',
                how='left',
                validate='m:1',
            )

            # Record the sales.
            all_sales.append(banned_sales)


# Aggregate banned sales.
sales = pd.concat(all_sales)
sales.drop_duplicates(subset='SaleDetailId', inplace=True)
try:
    sales['year'] = sales['SaleDate'].apply(lambda x: x.year)
except:
    sales['year'] = sales['CreatedDate'].apply(lambda x: x.year)

# Calculate the total sales of products with banned pesticides.
sales['price'] = sales['UnitPrice'] - sales['Discount']
total_sales_2023 = sales.loc[(sales['price'] > 0) & (sales['year'] == 2023)]['price'].sum()
print(f"Total Sales of Banned Pesticide Products 2023: ${round(total_sales_2023, 2)}")

# Calculate the tax revenue generated from products containing banned pesticides.
sales['tax'] = sales['SalesTax'] + sales['OtherTax']
total_tax_2023 = sales.loc[(sales['tax'] > 0) & (sales['year'] == 2023)]['tax'].sum()
print(f"Total Tax from Banned Pesticide Products 2023: ${round(total_tax_2023, 2)}")

# Count the retailers who sold a product with a banned pesticide.
total_retailers_2023 = len(sales.loc[sales['year'] == 2023]['LicenseeId'].unique())
print(f"Total Retailers that Sold Products Containing Banned Pesticides in 2023: {total_retailers_2023}")


# === Archive ===

# TODO: Save the number of banned pesticides produced by license.
outfile = './data/wa-license-pesticide-proportions.xlsx'
proportions.to_excel(outfile)


# === Visualize the data ===

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

def with_commas(x, pos):
    return "{:,}".format(int(x))


# Visualize Batches with Banned Pesticides
plt.figure(figsize=(10, 6))
sns.barplot(
    x=batches_with_banned_count.index[2:],
    y=batches_with_banned_count.values[2:],
)
plt.title("Batches with Banned Pesticides Over Years", fontsize=21)
plt.xlabel("Year", fontsize=16)
plt.ylabel("Number of Batches", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.gca().yaxis.set_major_formatter(FuncFormatter(with_commas))
plt.tight_layout()
plt.savefig('./presentation/images/survival-analysis-banned-batches-over-years.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Visualize number of batches with banned pesticides and
# number of clean batches, sorted by total number of batches in a
# stacked bar chart for each licensee.
sample = results.loc[results['year'] == 2023]
licensee_batch_counts = sample.groupby('licensee_id')['banned_pesticide'].value_counts().unstack(fill_value=0)
licensee_batch_counts.columns = [
    'Clean batches',
    'Batches with banned pesticides'
]
licensee_batch_counts['Total Batches'] = licensee_batch_counts.sum(axis=1)

# Sorted by total batches.
licensee_batch_counts.sort_values('Total Batches', ascending=False, inplace=True)
plt.figure(figsize=(15, 8))
licensee_batch_counts[['Clean batches', 'Batches with banned pesticides']].head(10).plot(kind='bar', stacked=True, color=['green', 'red'])
plt.title('Total tests for the top 10 producers in WA in 2023', fontsize=18)
plt.xlabel('Licensee ID', fontsize=16)
plt.ylabel('Number of Batches', fontsize=16)
plt.legend(title='Batch Type')
plt.gca().yaxis.set_major_formatter(FuncFormatter(with_commas))
plt.tight_layout()
plt.savefig('./presentation/images/survival-analysis-licensees-total-batches.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Sorted by banned batches.
licensee_batch_counts.sort_values('Batches with banned pesticides', ascending=False, inplace=True)
plt.figure(figsize=(15, 8))
licensee_batch_counts[['Clean batches', 'Batches with banned pesticides']].head(10).plot(kind='bar', stacked=True, color=['green', 'red'])
plt.title('Tests for licensees with the most banned pesticide detections in WA in 2023', fontsize=18)
plt.xlabel('Licensee ID', fontsize=16)
plt.ylabel('Number of Batches', fontsize=16)
plt.legend(title='Batch Type')
plt.gca().yaxis.set_major_formatter(FuncFormatter(with_commas))
plt.tight_layout()
plt.savefig('./presentation/images/survival-analysis-licensees-banned-batches.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Histogram of Proportions of Banned Pesticides in Products per Licensee
plt.figure(figsize=(10, 6))
plt.hist(proportions['proportion'], bins=100, edgecolor='black')
plt.title('Distribution of Banned Pesticide Proportions per Licensee', fontsize=18)
plt.xlabel('Proportion of Banned Pesticides', fontsize=16)
plt.ylabel('Number of Licensees', fontsize=16)
plt.savefig('./presentation/images/survival-analysis-proportions-per-licensee.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Visualize detection rates by lab.
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=lab_analysis,
    x='total_tests',
    y='detection_rate',
    size='total_tests',
    hue='detection_rate',
    sizes=(40, 400),  # Adjust the range of sizes as needed
    palette='coolwarm_r',  # You can choose a different color palette if desired
    legend=None,
)
plt.title('Detection Rate of Banned Pesticides by Lab', fontsize=18)
plt.xlabel('Total Tests', fontsize=16)
plt.ylabel('Detection Rate (%)', fontsize=16)
plt.savefig('./presentation/images/survival-analysis-detection-rates-by-lab.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Pie Chart of Banned Pesticides Detected
unique_pesticides_counts = results['pesticides'].explode().value_counts()
filtered_unique_pesticides_counts = unique_pesticides_counts.drop(['piperonyl_butoxide', 'pyrethrins'])
plt.figure(figsize=(10, 6))
filtered_unique_pesticides_counts.head(10).plot(kind='barh')
plt.title('Proportion of Each Banned Pesticide Detected', fontsize=18)
plt.xlabel('Count', fontsize=16)
plt.ylabel('Pesticide', fontsize=16)
plt.savefig('./presentation/images/survival-analysis-banned-pesticides-detected.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
