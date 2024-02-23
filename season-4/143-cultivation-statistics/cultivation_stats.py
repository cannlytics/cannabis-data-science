"""
Cultivation Statistics
Copyright (c) 2024 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 1/17/2024
Updated: 1/17/2024
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# Standard imports:
import calendar
import os

# External imports:
from adjustText import adjust_text
from cannlytics.data.ccrs import CCRS, merge_datasets
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns


# === Setup ===

# Initialize.
base = 'D://data/washington/'
stats_dir = 'D://data/washington/stats'
manager = CCRS()

# Curate the inventory for each release.
releases = [
    # 'CCRS PRR (4-4-23)',
    # 'CCRS PRR (5-7-23)',
    # 'CCRS PRR (6-6-23)',
    'CCRS PRR (8-4-23)',
    'CCRS PRR (9-5-23)',
    'CCRS PRR (11-2-23)',
    'CCRS PRR (12-2-23)',
    'CCRS PRR (1-2-24)',
]


def get_prr_datafiles(releases, subset='') -> list:
    """Get the datafiles for given releases."""
    if isinstance(releases, str):
        releases = [releases]
    filenames = []
    for release in releases:
        data_dir = os.path.join(base, release, release)
        for directory, _, files in os.walk(data_dir):
            for datafile in files:
                if subset:
                    if not datafile.startswith(f'{subset}_') or datafile.startswith('~$'):
                        continue
                    filename = os.path.join(directory, datafile)
                    filenames.append(filename)
    return filenames


# TODO: Merge strain data with plants.
def merge_strains(items, strain_files):
    """Merge strains with inventory items using `StrainId`."""
    items = merge_datasets(
        items,
        strain_files,
        dataset='strains',
        on='StrainId',
        target='strain_name',
        how='left',
        validate='m:1',
        rename={
            'Name': 'strain_name',
            'CreatedDate': 'strain_created_at',
        },
        drop=['CreatedBy', 'UpdatedBy', 'UpdatedDate'],
        dedupe=True,
    )
    missing = (items['strain_name'] == False) | (items['strain_name'] == 'False')
    items.loc[missing, 'strain_name'] = items.loc[missing, 'StrainType']
    items.loc[missing, 'StrainType'] = None
    return items


def with_commas(x, pos):
    return "{:,}".format(int(x))


# === Analyze how long plants are grown. ===

# Find all harvested plants.
harvested_plants = []
plant_files = get_prr_datafiles(releases, 'Plant')
for plant_file in plant_files:
    plants = pd.read_csv(
        plant_file,
        sep='\t',
        encoding='utf-16',
        engine='python',
    )
    plants = plants[plants['PlantState'] == 'Harvested']
    harvested_plants.append(plants)
harvested_plants = pd.concat(harvested_plants)

# Calculate the lifetime of each mother plant.
harvested_plants['CreatedDate'] = pd.to_datetime(harvested_plants['CreatedDate'])
harvested_plants['HarvestDate'] = pd.to_datetime(harvested_plants['HarvestDate'])
harvested_plants['lifetime'] = harvested_plants['HarvestDate'] - harvested_plants['CreatedDate']

# Visualize the lifetime of each mother plant.
plt.figure(figsize=(10, 6))
sample = harvested_plants['lifetime'].dropna()
sample_days = sample.dt.total_seconds() / (24 * 3600)
sample_days = sample_days[(sample_days > 0) & (sample_days < 365)]
sample_days.hist(bins=125, color='skyblue', edgecolor='black')

# Calculate and annotate the average lifetime
average_lifetime = sample_days[sample_days > 30].mean()
plt.axvline(average_lifetime, color='green', linestyle='dashed', linewidth=2)
plt.text(average_lifetime + 5, plt.ylim()[1]*0.9, f'Avg: {average_lifetime:.2f} days', color='green')

# Calculate and annotate the mode
mode_lifetime = sample_days[sample_days > 30].mode()[0]
plt.axvline(mode_lifetime, color='red', linestyle='dashed', linewidth=2)
plt.text(mode_lifetime + 5, plt.ylim()[1]*0.8, f'Mode: {mode_lifetime:.2f} days', color='red')

# Set title and labels
plt.title('Lifetime of Harvested Plants')
plt.xlabel('Lifetime (days)')
plt.ylabel('Observations')
plt.gca().yaxis.set_major_formatter(FuncFormatter(with_commas))
plt.tight_layout()
plt.savefig('./presentation/images/lifetime-harvested-plants.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


# === Area analysis ===

# TODO: Count the number of "Growing" and "Flowering" plants per `AreaId`.
# PlantState	GrowthStage
# Growing	Flowering


# TODO: Visualize the number of flowering plants per area.


# TODO: Look at the number of "Growing" "Vegetative" or "Immature" plants per `AreaId`.



# === Clone success rate analysis. ===

# Find all the mother plant IDs.
all_plants = []
clones = {}
mother_plant_ids = []
plant_files = get_prr_datafiles(releases, 'Plant')
for plant_file in plant_files:
    plants = pd.read_csv(
        plant_file,
        sep='\t',
        encoding='utf-16',
        engine='python',
        usecols=['PlantId', 'MotherPlantId'],
    )
    ids = list(plants['MotherPlantId'].astype(str).unique())
    mother_plant_ids.extend(ids)

    # Keep track of all clone `PlantId`s for each mother plant.
    for _, row in plants.iterrows():
        if not pd.isna(row['MotherPlantId']):
            mother_id = str(row['MotherPlantId']).replace('.0', '')
            plant_id = str(row['PlantId']).replace('.0', '')
            if mother_id not in clones:
                clones[mother_id] = []
            clones[mother_id].append(plant_id)

# Remove duplicates from mother_plant_ids if necessary
mother_plant_ids = list(set(mother_plant_ids))

# Clean IDs.
mother_plant_ids = [x.replace('.0', '') for x in mother_plant_ids]

# Find all the mother plants.
all_plants = []
for plant_file in plant_files:
    plants = pd.read_csv(
        plant_file,
        sep='\t',
        encoding='utf-16',
        engine='python',
    )
    all_plants.append(plants[plants['PlantId'].astype(str).isin(mother_plant_ids)])

# Find all of the unique mothers.
mother_plants = pd.concat(all_plants)
mother_plants.drop_duplicates(subset=['PlantId'], inplace=True)
print('Number of mother plants:', len(mother_plants))

# Merge the clones for each mother plant.
mother_plants['clones'] = mother_plants['PlantId'].apply(lambda x: clones[str(x)])
mother_plants['number_of_clones'] = mother_plants['clones'].apply(lambda x: len(x))

# Visualize the number of clones per mother plant.
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sample = mother_plants['number_of_clones'].loc[
    (mother_plants['number_of_clones'] > 1) &
    (mother_plants['number_of_clones'] < 250)
]
sample.hist(bins=125)
plt.title('Number of Clones per Mother Plant')
plt.xlabel('Number of Clones')
plt.ylabel('Observations')
plt.tight_layout()
plt.savefig('./presentation/images/clones-per-mother.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Calculate the lifetime of each mother plant.
mother_plants['CreatedDate'] = pd.to_datetime(mother_plants['CreatedDate'], format='mixed')
mother_plants['HarvestDate'] = pd.to_datetime(mother_plants['HarvestDate'], format='mixed')
mother_plants['lifetime'] = mother_plants['HarvestDate'] - mother_plants['CreatedDate']

# Visualize the lifetime of each mother plant.
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sample = mother_plants['lifetime'].dropna()
sample_days = sample.dt.total_seconds() / (24 * 3600)
sample_days = sample_days[
    (sample_days > 0) & (sample_days < 365)
]
sample_days.hist(bins=125)
plt.title('Lifetime of Mother Plants')
plt.xlabel('Lifetime (days)')
plt.ylabel('Observations')
plt.tight_layout()
plt.savefig('./presentation/images/lifetime-mother-plants.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# TODO: Test if the lifetime of a mother plant is greater than the
# lifetime of an average plant (or a clone).

# TODO: Look at the `PlantSource` for each mother.


# === Plant survival analysis ===

# Get plant destruction reasons.
destruction_files = get_prr_datafiles(releases, 'PlantDestructions')
destructions = []
for destruction_file in destruction_files:
    destructions.append(pd.read_csv(
        destruction_file,
        sep='\t',
        encoding='utf-16',
        engine='python',
        usecols=[
            'PlantId',
            'DestructionReason',
            'DestructionDate',
        ]
    ))
destructions = pd.concat(destructions)
destructions['PlantId'] = destructions['PlantId'].astype(str).replace('.0', '')
destructions.drop_duplicates(subset=['PlantId'], inplace=True)
print('Number of destroyed plants:', len(destructions))

# Get destroyed plants.
destroyed_plants = []
plant_files = get_prr_datafiles(releases, 'Plant')
for plant_file in plant_files:
    plants = pd.read_csv(
        plant_file,
        sep='\t',
        encoding='utf-16',
        engine='python',
    )
    destroyed_plants.append(plants[plants['PlantId'].astype(str).isin(destructions['PlantId'])])
destroyed_plants = pd.concat(destroyed_plants)
print('Number of destroyed plants:', len(destroyed_plants))

# Merge destruction reasons with destroyed plants.
destroyed_plants['PlantId'] = destroyed_plants['PlantId'].astype(str).replace('.0', '')
destroyed_plants = pd.merge(
    destroyed_plants,
    destructions,
    on='PlantId',
    how='left',
)
destroyed_plants.drop_duplicates(subset=['PlantId'], inplace=True)

# Calculate the lifetime of each destroyed plant.
destroyed_plants['CreatedDate'] = pd.to_datetime(destroyed_plants['CreatedDate'], format='mixed')
destroyed_plants['DestructionDate'] = pd.to_datetime(destroyed_plants['DestructionDate'], format='mixed')
destroyed_plants['lifetime'] = destroyed_plants['DestructionDate'] - destroyed_plants['CreatedDate']

# Visualize the lifetime of each destroyed plant.
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sample = destroyed_plants['lifetime'].dropna()
sample_days = sample.dt.total_seconds() / (24 * 3600)
sample_days = sample_days[
    (sample_days > 0) & (sample_days < 365)
]
sample_days.hist(bins=125)
plt.title('Lifetime of Destroyed Plants')
plt.xlabel('Lifetime (days)')
plt.ylabel('Observations')
plt.tight_layout()
plt.savefig('./presentation/images/lifetime-destroyed-plants.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Visualize the lifetime of each destroyed plant by reason.
destroyed_plants['lifetime_days'] = destroyed_plants['lifetime'].dt.total_seconds() / (24 * 3600)
filtered_plants = destroyed_plants[
    (destroyed_plants['lifetime_days'] > 0) & (destroyed_plants['lifetime_days'] < 365)
]
plt.figure(figsize=(15, 10))
sns.set(style="whitegrid")
g = sns.FacetGrid(filtered_plants, col='DestructionReason', col_wrap=4, height=4, aspect=1)
g.map(plt.hist, 'lifetime_days', bins=20, color='skyblue', edgecolor='black')
g.fig.suptitle('Lifetime of Destroyed Plants by Reason', fontsize=16)
g.set_axis_labels('Lifetime (days)', 'Number of Plants')
plt.subplots_adjust(top=0.9)
plt.savefig('./presentation/images/lifetime-destroyed-plants-by-reason.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Pie chart of destruction reasons.
plt.figure(figsize=(8, 8))
destruction_reasons = destroyed_plants['DestructionReason'].value_counts()
patches, texts, autotexts = plt.pie(
    destruction_reasons[:-1],
    labels=destruction_reasons[:-1].index,
    autopct='%1.1f%%',
    startangle=45,
    # colors=plt.cm.Paired.colors
)
for text in texts + autotexts:
    text.set_fontsize(12)
# adjust_text(autotexts)
# adjust_text(texts)
plt.title('Plant Destruction Reasons')
plt.ylabel('')
plt.tight_layout()
plt.savefig('./presentation/images/plant-destruction-reasons.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


# === Survival analysis ===

# Format survival dataset.
destroyed_plants['survived'] = False
harvested_plants['survived'] = True
mother_plants['survived'] = ~mother_plants['HarvestDate'].isna()
mother_plants['mother'] = True
destroyed_plants['mother'] = False
harvested_plants['mother'] = False
all_plants = pd.concat([
    # mother_plants,
    destroyed_plants,
    harvested_plants,
])
all_plants.drop_duplicates(subset=['PlantId'], inplace=True)

# Visualize plants harvested and plants destroyed by week.
harvested_plants['week'] = harvested_plants['HarvestDate'].dt.isocalendar().week
destroyed_plants['week'] = destroyed_plants['DestructionDate'].dt.isocalendar().week
harvested_count = harvested_plants.groupby('week')['PlantId'].count()
destroyed_count = destroyed_plants.groupby('week')['PlantId'].count()
time_series_data = pd.DataFrame({
    'Harvested': harvested_count,
    'Destroyed': destroyed_count
}).fillna(0)  # Fill missing values with 0
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")
plt.plot(time_series_data[:-1]['Harvested'], label='Harvested', color='green')
plt.plot(time_series_data[:-1]['Destroyed'], label='Destroyed', color='red')
plt.title('Number of Plants Harvested and Destroyed by Week')
plt.xlabel('Week of the Year')
plt.ylabel('Number of Plants')
plt.legend()
plt.savefig('./presentation/images/number-of-finished-plants.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Print out descriptive statistics.
years_of_interest = [2022, 2023]
mother_plants['year_planted'] = mother_plants['CreatedDate'].dt.year
destroyed_plants['year_planted'] = destroyed_plants['CreatedDate'].dt.year
harvested_plants['year_planted'] = harvested_plants['CreatedDate'].dt.year
statistics = {}
for year in years_of_interest:
    year_df = all_plants[all_plants['year_planted'] == year]
    year_mother_plants = mother_plants[mother_plants['year_planted'] == year]
    year_destroyed_plants = destroyed_plants[destroyed_plants['year_planted'] == year]
    year_harvested_plants = harvested_plants[harvested_plants['year_planted'] == year]
    statistics[year] = {
        'Cultivators': len(year_df['LicenseeId'].unique()),
        'Areas': len(year_df['AreaId'].unique()),
        'Strains': len(year_df['StrainId'].unique()),
        'Mother Plants': len(year_mother_plants),
        'Destroyed Plants': len(year_destroyed_plants),
        'Harvested Plants': len(year_harvested_plants)
    }
stats_df = pd.DataFrame(statistics).T
stats_df = stats_df.T
stats_df['percent_change'] = (stats_df[2023] - stats_df[2022]) / stats_df[2022] * 100
stats_df['percent_change'] = stats_df['percent_change'].apply(lambda x: f'{x:.0f}\%')
stats_df = stats_df.applymap(lambda x: "{:,}".format(x) if isinstance(x, (int, float)) else x)
latex_table = stats_df.to_latex(column_format='lrrr')
print(latex_table)

# TODO: Visualize the number of plants grown by seed vs. clone by week.

# Calculate the lifetime in days for each plant.
all_plants['lifetime_days'] = all_plants.apply(lambda row: (row['HarvestDate'] if row['survived'] else row['DestructionDate']) - row['CreatedDate'], axis=1).dt.days


def perform_survival_analysis(
        data,
        time_col,
        event_col,
        group_col=None,
        title="Survival Analysis",
    ):
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    if group_col:
        for name, grouped_df in data.groupby(group_col):
            kmf.fit(grouped_df[time_col], event_observed=grouped_df[event_col], label=name)
            kmf.plot()
    else:
        kmf.fit(data[time_col], event_observed=data[event_col])
        kmf.plot()
    plt.title(title)
    plt.xlabel('Days')
    plt.ylabel('Survival Rate')
    plt.xlim(0, 365)
    figure_name = group_col.lower().replace(' ', '-').replace('_', '-') if group_col else 'overall'
    plt.savefig(f'./presentation/images/survival-analysis-{figure_name}.pdf', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


def survival_analysis_by_factor(
        data,
        factor,
        time_col='lifetime_days',
        event_col='survived',
        title_prefix=""
    ):
    """Sample function for survival analysis based on a factor"""
    title = f"{title_prefix} Survival Rate by {factor}" if title_prefix else f"Survival Rate by {factor}"
    perform_survival_analysis(data, time_col, event_col, factor, title)


# Overall Survival Analysis
perform_survival_analysis(all_plants, 'lifetime_days', 'survived', title="Overall Plant Survival Rate")

# Optional: Visualize the survival rate of harvested vs. destroyed plants.
# Does this make sense?

# Survival Rate by Year Planted
time_col = 'lifetime_days'
event_col = 'survived'
group_col = 'year_planted'
all_plants['year_planted'] = all_plants['CreatedDate'].dt.year
kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
for name, grouped_df in all_plants.groupby(group_col):
    if name < 2022:
        continue
    kmf.fit(grouped_df[time_col], event_observed=grouped_df[event_col], label=name)
    kmf.plot()
plt.title('Survival Rate by Year Planted')
plt.xlabel('Days')
plt.ylabel('Survival Rate')
plt.xlim(0, 365)
figure_name = group_col.lower().replace(' ', '-').replace('_', '-') if group_col else 'overall'
plt.savefig(f'./presentation/images/survival-analysis-{figure_name}.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Survival Rate by Month Planted
all_plants['month_planted'] = all_plants['CreatedDate'].dt.month
time_col = 'lifetime_days'
event_col = 'survived'
group_col = 'month_planted'
palette = sns.cubehelix_palette(12, start=.5, rot=-.75)
kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
for i, (name, grouped_df) in enumerate(all_plants.groupby(group_col)):
    month_name = calendar.month_name[name]
    kmf.fit(grouped_df[time_col], event_observed=grouped_df[event_col], label=month_name)
    kmf.plot(ci_show=False, color=palette[i % len(palette)])
plt.title('Survival Rate by Month Planted')
plt.xlabel('Days')
plt.ylabel('Survival Rate')
plt.xlim(0, 365)
plt.legend(title='Month', loc='best')
plt.savefig(f'./presentation/images/survival-analysis-month-planted.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Effect of Plant Source
survival_analysis_by_factor(all_plants, 'PlantSource', title_prefix="Effect of Plant Source on")

# Mother Plant Survival
survival_analysis_by_factor(all_plants, 'IsMotherPlant', title_prefix="Mother Plant")


# Survival rate depending on if banned pesticide detected in licensee's tests.
all_plants['LicenseeId'] = all_plants['LicenseeId'].astype(str)
pesticides_df = pd.read_excel(
    './data/wa-license-pesticide-proportions.xlsx',
    usecols=['LicenseeId', 'proportion'],
)
pesticides_df['pesticide_detected'] = pesticides_df['proportion'] > 0
pesticides_df['LicenseeId'] = pesticides_df['LicenseeId'].astype(str)
try:
    all_plants = all_plants.merge(
        pesticides_df,
        on='LicenseeId',
        how='left',
        validate='m:1',
    )
except:
    pass
time_col = 'lifetime_days'
event_col = 'survived'
group_col = 'pesticide_detected'
legend_labels = {
    True: "Banned Pesticide(s) Detected in Licensee's Tests",
    False: 'No Banned Pesticides Detected by Licensee'
}
kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
for name, grouped_df in all_plants.groupby(group_col):
    label = legend_labels[name]  # Use the more informative label
    kmf.fit(grouped_df[time_col], event_observed=grouped_df[event_col], label=label)
    kmf.plot()
plt.title('Survival Rate by Detection of Banned Pesticides')
plt.xlabel('Days')
plt.ylabel('Survival Rate')
plt.xlim(0, 365)
figure_name = group_col.lower().replace(' ', '-').replace('_', '-') if group_col else 'overall'
plt.savefig(f'./presentation/images/survival-analysis-{figure_name}.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


# FIXME:


# def plot_selected_categories_survival(data, category_field, title_prefix):
#     # Calculate the count for each category
#     category_counts = data[category_field].value_counts()

#     # Determine the categories for specific percentiles
#     bottom_category = category_counts.idxmin()
#     percentile_25_category = category_counts.quantile(0.25, interpolation='nearest')
#     median_category = category_counts.median()
#     percentile_75_category = category_counts.quantile(0.75, interpolation='nearest')
#     top_category = category_counts.idxmax()

#     selected_categories = [bottom_category, percentile_25_category, median_category, percentile_75_category, top_category]

#     # Plot survival analysis for the selected categories
#     kmf = KaplanMeierFitter()
#     plt.figure(figsize=(10, 6))
#     sns.set(style="whitegrid")

#     for category in selected_categories:
#         subset = data[data[category_field] == category]
#         kmf.fit(subset['lifetime_days'], event_observed=subset['survived'], label=f"{category_field} {category}")
#         kmf.plot()

#     plt.title(f'{title_prefix} Survival Rate by {category_field}')
#     plt.xlabel('Days')
#     plt.ylabel('Survival Rate')
#     plt.legend()
#     plt.savefig(f'./presentation/images/survival-analysis-{category_field}.pdf', dpi=300, bbox_inches='tight', transparent=True)
#     plt.show()

# # Strain Influence on Survival
# plot_selected_categories_survival(all_plants, 'StrainId', "Strain Influence on")

# # Area or Location Impact
# plot_selected_categories_survival(all_plants, 'AreaId', "Area Impact on")

# # Licensee Performance
# plot_selected_categories_survival(all_plants, 'LicenseeId', "Licensee Performance on")


# def plot_percentile_survival(data, category, title_prefix):
#     # Determine the percentiles based on the count of plants in each category
#     category_counts = data[category].value_counts()
#     bottom = category_counts.quantile(0.0)
#     p25 = category_counts.quantile(0.25)
#     median = category_counts.quantile(0.5)
#     p75 = category_counts.quantile(0.75)
#     top = category_counts.quantile(1.0)

#     # Select categories in each percentile
#     selected_categories = category_counts[
#         (category_counts == bottom) |
#         (category_counts == p25) |
#         (category_counts == median) |
#         (category_counts == p75) |
#         (category_counts == top)
#     ].index.tolist()

#     # Filter data to only include selected categories
#     filtered_data = data[data[category].isin(selected_categories)]

#     # Perform survival analysis
#     survival_analysis_by_factor(filtered_data, category, title_prefix=title_prefix)


# # Strain Influence on Survival
# plot_percentile_survival(all_plants, 'StrainId', "Strain Influence on")

# # Area or Location Impact
# plot_percentile_survival(all_plants, 'AreaId', "Area Impact on")

# # Licensee Performance
# plot_percentile_survival(all_plants, 'LicenseeId', "Licensee Performance on")


# TODO: Survival regression with all variables:
# - StrainId
# - AreaId
# - LicenseeId
# - PlantSource
# - IsMotherPlant
# - pesticide_detected

from lifelines import CoxPHFitter

# Preprocess the data for CoxPH model
# Convert categorical variables to dummy variables
categorical_vars = [
    # 'StrainId',
    # 'AreaId',
    # 'LicenseeId',
    'PlantSource',
    'IsMotherPlant',
    'pesticide_detected',
]
all_plants_cox = pd.get_dummies(
    all_plants,
    columns=categorical_vars,
    drop_first=True
)

# Prepare the CoxPH model data where:
# - 'lifetime_days' is the duration;
# - 'survived' is the event occurred flag.
cox_data = all_plants_cox[['lifetime_days', 'survived'] + [col for col in all_plants_cox.columns if col.startswith(tuple(categorical_vars))]]

# Fit the Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(cox_data, duration_col='lifetime_days', event_col='survived')

# Check the summary of the model
print(cph.summary)

# Plot the coefficients.
# cph.plot()
# plt.savefig('./presentation/images/coxph-model-coefficients.pdf', dpi=300, bbox_inches='tight', transparent=True)
# plt.show()
