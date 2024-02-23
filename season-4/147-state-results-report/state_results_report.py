"""
Sate Results Report | California
Copyright (c) 2024 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 2/20/2024
Updated: 2/22/2024
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# Standard imports:
import os
import matplotlib.pyplot as plt
import seaborn as sns

# External imports:
from adjustText import adjust_text
# from cannlytics.stats import calculate_shannon_diversity
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress


# === Setup ===

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
})

# Define the report directory.
report_dir = './state-results-report/stats'
assets_dir = './state-results-report/assets/figures'


# === Get the data ===

# Read Flower Co. results.
datafile = 'https://cannlytics.page.link/ca-results-flower-company-2024-02-21'
flower_co_results = pd.read_excel(datafile)
print('Read Flower Co. results:', len(flower_co_results))

# Define cannabinoids and terpenes.
cannabinoids = [
    'cbc',
    'cbca',
    'cbd',
    'cbda',
    'cbdv',
    'cbg',
    'cbga',
    'cbn',
    'delta_8_thc',
    'delta_9_thc',
    'thca',
    'thcv',
    'thcva',
]
terpenes = [
    'd_limonene',
    'beta_caryophyllene',
    'linalool',
    'trans_beta_farnesene',
    'fenchol',
    'alpha_humulene',
    'terpineol',
    'guaiol',
    'beta_myrcene',
    'beta_pinene',
    'alpha_pinene',
    'alpha_ocimene',
    'delta_3_carene',
    'borneol',
    'camphene',
    'camphor',
    'caryophyllene_oxide',
    'cedrol',
    'eucalyptol',
    'fenchone',
    'geraniol',
    'geranyl_acetate',
    'hexahydrothymol',
    'isoborneol',
    'isopulegol',
    'nerol',
    'pulegone',
    'sabinene',
    'valencene',
    'alpha_bisabolol',
    'alpha_cedrene', 
    'alpha_phellandrene',
    'alpha_terpinene',
    # 'alpha_terpinolene', # Not measured in CA
    'nerolidol',
    'gamma_terpinene',
    'trans_nerolidol',
    'beta_ocimene', # Measured in CA
    'endo_fenchyl', # Measured in CA
]

# Standard theme.
concentrate_colors = [
    '#00BFFF',
    '#006400',
    '#32CD32',
    '#4682B4',
    'crimson',
    '#9400D3',
    '#A0522D',
    '#CDAF95',
    '#DAA520',
    '#FF4500',
    '#FF8C00'
]


# === Clean the data ===

# Specify the results to analyze.
results = flower_co_results.copy()

# Combine similarly named terpenes.
results['trans_beta_farnesene'] = results['trans_beta_farnesene'].combine_first(results['beta_farnesene'])
results['trans_beta_farnesene'] = results['trans_beta_farnesene'].combine_first(results['cis_beta_farnesene'])
results['caryophyllene_oxide'] = results['caryophyllene_oxide'].combine_first(results['beta_caryophyllene_oxide'])
results['beta_ocimene'] = results['beta_ocimene'].combine_first(results['cis_beta_ocimene'])
results['beta_ocimene'] = results['beta_ocimene'].combine_first(results['trans_beta_ocimene'])
results['endo_fenchyl'] = results['endo_fenchyl'].combine_first(results['endo_fenchyl_alcohol'])
results['endo_fenchyl'] = results['endo_fenchyl'].combine_first(results['1_r_endo_fenchyl_alcohol'])
results['nerolidol'] = results['nerolidol'].combine_first(results['cis_nerolidol'])

# Standardize labs.
standardization_mapping = {
    'CC Testing Labs': 'CC Testing Labs, Inc.',
    'Pride Analytics dba 2 River Labs': '2 River Labs, Inc'
}
results['lab'] = results['lab'].replace(standardization_mapping)

# Standardize producers.
producer_standardization_mapping = {
    '2CW PRODUCTION, INC': '2CW Productions, Inc.',
    '2CW Productions,Inc.': '2CW Productions, Inc.',
    'CI Wellness LLC.': 'CI Wellness, LLC',
    'CI Wellness': 'CI Wellness, LLC',
    'CI Farms': 'CI Wellness, LLC',
    'CI Infusion, LLC': 'CI Wellness, LLC',
    'CRFT Manufacturing, Inc.': 'CRFT Manufacturing, Inc.',
    'CRFTManufacturing, Inc.': 'CRFT Manufacturing, Inc.',
    'Central Coast Ag Products, LLC': 'Central Coast Ag Products, LLC',
    'Central Coast AgProducts, LLC': 'Central Coast Ag Products, LLC',
    'DSM': 'DSM Group LLC',
    'DSM Group LLC': 'DSM Group LLC',
    'FAMILY FLORALS, INC': 'Family Florals, Inc.',
    'Family Florals, Inc.': 'Family Florals, Inc.',
    'GP Operations, INC.': 'GP Operations, Inc.',
    'GP Operations, Inc.': 'GP Operations, Inc.',
    'Green Dragon Collective': 'Green Dragon Collective, Inc.',
    'Green Dragon Caregivers, Inc.': 'Green Dragon Collective, Inc.',
    'Green Spectrum Trading LLC': 'Green Spectrum Trading, Inc',
    'Green Spectrum Trading, Inc': 'Green Spectrum Trading, Inc',
    'Ironworks Collective Inc.': 'Ironworks Collective, Inc.',
    'Ironworks Collective, Inc': 'Ironworks Collective, Inc.',
    'Ironworks Collective, Inc.': 'Ironworks Collective, Inc.',
    'KUSHMEN & BAKEFIELDS ENTERPRISES, INC.': 'KUSHMEN & BAKEFIELDS ENTERPRISES, INC.',
    'KUSHMEN &BAKEFIELDS ENTERPRISES, INC.': 'KUSHMEN & BAKEFIELDS ENTERPRISES, INC.',
    'Love Gem Inc': 'Love Gem Inc (Space Gem)',
    'Love Gem Inc(Space Gem)': 'Love Gem Inc (Space Gem)',
    'MVN Enterprises, LLC': 'MVN Productions, LLC',
    'MVN Productions': 'MVN Productions, LLC',
    'MVN Productions, LLC': 'MVN Productions, LLC',
    'Mvn Productions, LLC': 'MVN Productions, LLC',
    'Med For America Inc.': 'Med for America Inc.',
    'Med for America Inc.': 'Med for America Inc.',
    'Mendocino Grasslands I': 'Mendocino Grasslands',
    'Mendocino Grasslands II': 'Mendocino Grasslands',
    'PRUSSIAN STICKS LLC': 'PRUSSIAN STICKS LLC',
    'PRUSSIAN STICKSLLC': 'PRUSSIAN STICKS LLC',
    'Royal Key, LLC, Suprize Suprize': 'Royal Key, LLC, Suprize Suprize',
    'Royal Key, LLC,Suprize Suprize': 'Royal Key, LLC, Suprize Suprize',
    'S&B WATER INC': 'S&B WATER, INC',
    'S&B WATER, INC': 'S&B WATER, INC',
    'Shield Management Group': 'Shield Management Group, LLC',
    'Shield Management Group, LLC': 'Shield Management Group, LLC',
    'URBAN THERAPIES DISTRIBUTION': 'Urban Therapies Manufacturing, LLC',
    'Urban Therapies Distribution': 'Urban Therapies Manufacturing, LLC',
    'Urban Therapies Manufacturing, LLC': 'Urban Therapies Manufacturing, LLC',
    'VBR DHS MANAGEMENT, LLC': 'VBR DHS Management LLC',
    'VBR DHS Management LLC': 'VBR DHS Management LLC',
    'WESTSIDE CAREGIVERS CLUB, INC': 'Westside Caregivers Club, Inc.',
    'Westside Caregivers Club, Inc.': 'Westside Caregivers Club, Inc.',
    'WestsideCaregivers': 'Westside Caregivers Club, Inc.',
    'Mission Health Associates, Inc. dba Glass House Farms': 'Glass House Farms',
    'Glass House Camarillo Cultivation LLC': 'Glass House Farms',
}
results['producer'] = results['producer'].replace(producer_standardization_mapping)
print(results['producer'].value_counts().head(20))

# Output producer value counts to LaTeX.
top_producers = results['producer'].value_counts().head(15)
top_producers_df = top_producers.reset_index()
top_producers_df.columns = ['Top 15 Producers Observed', 'Observations']
latex_code = top_producers_df.to_latex(index=False, header=True)
print(latex_code)
with open(f'{report_dir}/top-producers.tex', 'w') as latex_file:
    latex_file.write(latex_code)


# === Calculate summary statistics ===

# Calculate summary statistics
total_strains = len(results['strain_name'].unique())
total_parsed_coas = len(results)
total_producers = len(results['producer'].unique())
total_labs = len(results['lab'].unique())

# Assuming all cannabinoids and terpenes listed are analyzed in each sample
number_of_cannabinoids = len(cannabinoids)
number_of_terpenes = len(terpenes)

# Compile summary statistics into a DataFrame
summary = pd.DataFrame({
    'Total Parsed COAs': [total_parsed_coas],
    'Total Producers': [total_producers],
    'Total Labs': [total_labs],
    # 'Total Strains': [total_strains],
    'Number of Cannabinoids': [number_of_cannabinoids],
    'Number of Terpenes': [number_of_terpenes],
}).T
summary.columns = ['Observations']
latex_code = summary.to_latex(
    index=True,
    header=False,
    formatters=[lambda x: "{:,}".format(x) if pd.notnull(x) else ""],
)
latex_code = latex_code.replace('\n\\midrule', '')
print(latex_code)
with open(f'{report_dir}/summary.tex', 'w') as file:
    file.write(latex_code)


# === Product Sub-type Analysis ===

def assign_product_subtype(product_type, subtypes):
    """
    Assigns a product subtype based on the product type description.
    Parameters:
        - product_type: str, the type of the product.
    Returns:
        - str, the assigned product subtype.
    """
    for subtype in subtypes:
        if subtype.lower() in str(product_type).lower():
            return subtype
    return 'Other'


def categorize_product_type(product_type):
    """
    Normalize the product type string for consistent comparison.
    Categorizes a given product type into 'Flower', 'Preroll', or 'Infused Preroll'.
    
    Parameters:
    - product_type: str, the type of the product.
    
    Returns:
    - str, the category of the product.
    """
    product_type = product_type.lower().replace('-', '')
    concentrate_keywords = ['concentrate', 'extract', 'distillate', 'vape', 'live resin', 'rosin', 'badder', 'batter', 'shatter', 'sauce', 'crumble', 'hash', 'diamonds']
    if 'infused' in product_type and 'preroll' in product_type:
        return 'Infused Preroll'
    elif 'preroll' in product_type:
        return 'Preroll'
    elif any(keyword in product_type for keyword in concentrate_keywords) and 'edible' not in product_type:
        return 'Concentrate'
    elif 'flower' in product_type:
        return 'Flower'
    else:
        return 'Other'


# Assign product subtypes.
categories = [
    'sauce',
    'liquid diamond',
    'diamond',
    'resin',
    'rosin',
    'distillate',
    'badder',
    'sugar',
    'shatter',
    'crumble',
    'hash',

]
results['product_subcategory'] = results['product_subtype'].apply(assign_product_subtype, subtypes=categories)
print(results['product_subcategory'].value_counts())

# Identify flower, prerolls, and infused prerolls.
results['product_category'] = results['product_type'].apply(categorize_product_type)

# Pie chart of category types.
plt.figure(figsize=(12, 12))
colors = ["#705898","#78C850", "#F85888",  "#F8D030",   "#6890F0",]
n_bins = 100
sample = results.loc[results['product_category'] != 'Other']
product_type_counts = sample['product_category'].str.title().value_counts()
plt.pie(
    product_type_counts,
    labels=product_type_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
)
plt.ylabel('')
plt.title('Product Categories')
plt.tight_layout()
plt.savefig(f'{assets_dir}/product-categories-pie-chart.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Proportion of concentrate types.
plt.figure(figsize=(12, 12))
colors = ["#78C850", "#F85888", "#F8D030", "#6890F0", "#705898"]
n_bins = 100
sample = results.loc[results['product_category'] == 'Concentrate']
sample = sample.loc[sample['product_subcategory'] != 'Other']
product_type_counts = sample['product_subcategory'].str.title().value_counts()
product_type_counts.sort_index(inplace=True)
plt.pie(
    product_type_counts,
    labels=product_type_counts.index,
    autopct='%1.1f%%',
    startangle=0,
    colors=concentrate_colors,
)
plt.ylabel('')
plt.title('Concentrate Sub-Categories')
plt.tight_layout()
plt.savefig(f'{assets_dir}/concentrate-subcategories-pie-chart.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


# === Total cannabinoids and terpenes analysis ===

# Visualize flower types total cannabinoids and total terpenes.
flower_types = [
    'Indoor',
    'Mixed Light',
    'Full Sun',
    # 'Non Infused',
]
plt.figure(figsize=(15, 9))
sns.scatterplot(
    data=results.loc[results['product_subtype'].isin(flower_types)].sort_values('product_subcategory'),
    y='total_cannabinoids',
    x='total_terpenes',
    hue='product_subtype',
    palette='bright',
    alpha=0.6,
    s=222,
)
plt.title('Total Cannabinoids to Total Terpenes in Flower', pad=20)
plt.ylabel('Total Cannabinoids')
plt.xlabel('Total Terpenes')
leg = plt.legend(title='Product Subtype', bbox_to_anchor=(1.05, 1), loc='upper left')
for leg_entry in leg.legendHandles: 
    leg_entry.set_sizes([222])
plt.tight_layout()
plt.ylim(18, 45)
plt.savefig(f'{assets_dir}/total-cannabinoids-total-terpenes-flower.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Visualize concentrate types total cannabinoids and total terpenes
plt.figure(figsize=(15, 9))
sns.scatterplot(
    data=results.loc[results['product_subcategory'] != 'Other'].sort_values('product_subcategory'),
    y='total_cannabinoids',
    x='total_terpenes',
    hue='product_subcategory',
    palette=concentrate_colors,
    s=222,
)
plt.title('Total Cannabinoids to Total Terpenes in Concentrates', pad=20)
plt.ylabel('Total Cannabinoids')
plt.xlabel('Total Terpenes')
leg = plt.legend(title='Product Subtype', bbox_to_anchor=(1.05, 1), loc='upper left')
for leg_entry in leg.legendHandles: 
    leg_entry.set_sizes([222])
plt.tight_layout()
plt.ylim(25, 100)
plt.savefig(f'{assets_dir}/total-cannabinoids-total-terpenes-concentrates.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


# === Chemical Diversity Analysis ===

# Clean the data.
totals = ['total_cannabinoids', 'total_terpenes', 'total_thc', 'total_cbd']
for compound in cannabinoids + terpenes + totals:
    results[compound] = pd.to_numeric(results[compound], errors='coerce')

# Calculate diversity of terpenes.
# results['cannabinoid_diversity'] = calculate_shannon_diversity(results, cannabinoids)
# results['terpene_diversity'] = calculate_shannon_diversity(results, terpenes)

# Visualize the distribution of terpenes by product category.
product_categories = results['product_category'].unique()
product_categories = product_categories[product_categories != 'Other']
product_categories = product_categories[product_categories != 'Infused Preroll']
plt.figure(figsize=(12, 7.25))
colors = ['crimson', 'mediumseagreen', 'royalblue', 'gold', 'purple']
color_dict = dict(zip(product_categories, colors))
texts = []
for category in product_categories:
    category_data = results.loc[results['product_category'] == category]
    sns.histplot(
        data=category_data,
        x='total_terpenes',
        kde=True,
        bins=24,
        color=color_dict.get(category, 'gray'),
        label=category,
        alpha=0.6,
        stat='density',
    )
for category in product_categories:
    category_data = results.loc[results['product_category'] == category]
    mean_value = category_data['total_terpenes'].mean()
    plt.axvline(
        x=mean_value,
        color=color_dict.get(category, 'black'),
        linestyle='--',
        lw=2,
    )
    text = plt.text(
        mean_value,
        plt.gca().get_ylim()[1] * 0.6,
        f'{category} Avg: {mean_value:.2f}%',
        color='black',
        ha='center',
    )
    texts.append(text)
adjust_text(texts)
plt.legend(loc='upper right')
plt.title('Total Terpenes in Flower', pad=20)
plt.xlabel('Total Terpenes (%)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f'{assets_dir}/total-terpenes-histogram.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Visualize the proportion of dominant terpenes.
dominant_terpene = results[terpenes].idxmax(axis=1)
results['dominant_terpene'] = dominant_terpene
category_dominant_terpene_counts = results.groupby(['product_category', 'dominant_terpene']).size().reset_index(name='count')
total_by_category = category_dominant_terpene_counts.groupby('product_category')['count'].transform('sum')
category_dominant_terpene_counts['percentage'] = (category_dominant_terpene_counts['count'] / total_by_category) * 100
category_dominant_terpene_counts = category_dominant_terpene_counts[category_dominant_terpene_counts['product_category'] != 'Other']
category_dominant_terpene_pivot = category_dominant_terpene_counts.pivot(index='dominant_terpene', columns='product_category', values='percentage').fillna(0)
category_dominant_terpene_pivot = category_dominant_terpene_pivot[(category_dominant_terpene_pivot > 2).any(axis=1)]
category_colors = {
    'Concentrate': 'darkorchid',
    'Flower': 'mediumseagreen',
    'Infused Preroll': 'darkkhaki',
    'Preroll': 'darkorange',
}
colors = [category_colors.get(x) for x in category_dominant_terpene_pivot.columns]
plt.figure(figsize=(15, 10))
category_dominant_terpene_pivot.plot(
    kind='bar',
    color=colors,
    figsize=(13, 8),
)
plt.title('Dominant Terpenes by Product Category', pad=20)
plt.ylabel('Percentage (%)')
plt.xlabel('')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Product Category')
plt.tight_layout()
plt.savefig(f'{assets_dir}/dominant-terpenes-by-category.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Look at linalool-dominant strains.
results['dominant_terpene'] = dominant_terpene
linalool_dominant = results.loc[results['dominant_terpene'] == 'linalool']

# TODO: Look at terpene profile of linalool-dominant strains (by product_category).


# === Chemical Analysis Continued ===

def plot_grouped_bar_chart(
        data,
        title,
        ylabel,
        output_filename,
        bar_width=0.15,
        rotation=0,
        annotate=False,
    ):
    plt.figure(figsize=(14, 8.5))
    n_groups = data.shape[1]
    n_bars = len(data)
    index = np.arange(n_groups)
    colors = ['#705898', '#78C850', '#F8D030', '#F85888', '#6890F0']
    for i, (product_type, row) in enumerate(data.iterrows()):
        bars = plt.bar(
            index + i * bar_width,
            row,
            bar_width,
            label=product_type,
            color=colors[i % len(colors)]  # Use modulo to cycle through colors if less than products
        )
        if annotate:
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f'{height:.0f}%',  # Format the value; adjust as needed
                    ha='center',  # Center the text horizontally
                    va='bottom',  # Position the text at the top (bottom of the text at the top of the bar)
                )
    plt.title(title, pad=20)
    plt.xlabel('')
    plt.ylabel(ylabel)
    plt.xticks(index + bar_width * (n_bars - 1) / 2, data.columns, rotation=rotation)
    plt.legend(title='Category', loc='upper right')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


# Create horizontal bar chart of average concentrations:
# total_cannabinoids, delta_9_thc, thca
sample = results.copy()
sample = sample.loc[sample['product_category'] != 'Other']
sample = sample.loc[
    (sample['total_cannabinoids'] < 100) &
    (sample['delta_9_thc'] < 100) &
    (sample['thca'] < 100)
]
avg_concentrations = sample.groupby('product_category')[cannabinoids + terpenes + totals].mean()
plot_grouped_bar_chart(
    avg_concentrations[['total_cannabinoids', 'delta_9_thc', 'thca']],
    'Average Concentration of Major Cannabinoids by Category',
    'Concentration (%)',
    f'{assets_dir}/major-cannabinoids-by-product-category.pdf',
    annotate=True,
)

# Create horizontal bar chart of average concentrations:
# Minor cannabinoids: all cannabinoids minus delta_9_thc, thca
minor_cannabinoids = [compound for compound in cannabinoids if compound not in ['delta_9_thc', 'thca',]]
plot_grouped_bar_chart(
    avg_concentrations[minor_cannabinoids],
    'Average Concentration of Minor Cannabinoids by Category',
    'Concentration (%)',
    f'{assets_dir}/minor-cannabinoids-by-product-category.pdf'
)

# Create horizontal bar chart of average concentrations:
# top_terpenes
top_terpenes = [
    'd_limonene',
    'beta_caryophyllene',
    'beta_myrcene',
    'linalool',
    'alpha_humulene',
    'alpha_pinene',
    'beta_pinene',
    'fenchol',
    'alpha_bisabolol',
    'trans_beta_farnesene',
]
plot_grouped_bar_chart(
    avg_concentrations[top_terpenes],
    'Average Concentration of Top Terpenes by Category',
    'Concentration (%)',
    f'{assets_dir}/top-terpenes-by-product-category.pdf',
    rotation=90,
)


# === Average Concentrations Table ===

def calculate_monthly_averages_by_type(df, compounds, product_type, month, year):
    """Calculate averages for a specific month and year."""
    df['date_tested'] = pd.to_datetime(df['date_tested'], errors='coerce')
    filtered_df = df[(df['date_tested'].dt.month == month) & (df['date_tested'].dt.year == year) & (df['product_category'] == product_type)]
    averages = {compound: filtered_df[compound].mean() for compound in compounds}
    return averages


def generate_latex_table_for_month(year, month, compounds, product_types, file_path):
    """Generate LaTeX table for average concentrations by product type for a specific month."""
    # Start LaTeX table
    latex_table = f"\\begin{{table}}[H]\n\\centering\nAverage Concentrations for {pd.to_datetime(f'{year}-{month}-01').strftime('%B %Y')}\n\\begin{{tabular}}{{l{'c' * len(product_types)}}}\n\\hline\ & {' & '.join(product_types)} \\\\\n\\hline\n"
    for compound_type, compounds_list in compounds.items():
        latex_table += f"\\multicolumn{{{len(product_types) + 1}}}{{l}}{{{compound_type}}} \\\\\n\\hline\n"
        for compound in sorted(compounds_list):
            row_data = []
            for product_type in product_types:
                avg = calculate_monthly_averages_by_type(sample, [compound], product_type, month, year).get(compound, np.nan)
                row_data.append(f"{avg:.2f}" if not np.isnan(avg) else "-")
            latex_table += f"{compound} & {' & '.join(row_data)} \\\\\n"
        latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n\\end{table}\n"
    latex_table = latex_table.replace('_', '\\_')
    latex_table = latex_table.replace(' nan ', ' - ')
    with open(file_path, 'w') as file:
        file.write(latex_table)
    print(f"Table for {pd.to_datetime(f'{year}-{month}-01').strftime('%B %Y')} saved to {file_path}")


# Generate compound tables.
sample = results.copy()
compounds = {"Cannabinoids": cannabinoids} # , "Terpenes": terpenes
product_types = ["Flower", "Preroll", "Concentrate"]
generate_latex_table_for_month(2023, 12, compounds, product_types, f'{report_dir}/compounds-2023-12.tex')
generate_latex_table_for_month(2024, 1, compounds, product_types, f'{report_dir}/compounds-2024-01.tex')


# === Analyze Ratios ===

def plot_ratio(
        data,
        x_compound,
        y_compound,
        hue="product_type",
        palette="Set2",
        title=None,
        xlabel=None,
        ylabel=None,
        ymax=None,
    ):
    """
    Plot a scatterplot with regression lines showing the ratio of two compounds and test if slopes are significantly different.
    """
    data = data.dropna(subset=[x_compound, y_compound])
    width = 14
    golden_ratio = (1 + 5 ** 0.5) / 2
    height = width / golden_ratio
    aspect_ratio = width / height
    lm = sns.lmplot(
        data=data,
        x=x_compound,
        y=y_compound,
        hue=hue,
        palette=palette,
        height=height,
        aspect=aspect_ratio,
        scatter_kws={'s': 100, 'alpha': 0.6},
        robust=True,
        ci=None,
        legend=None,
    )
    slopes = {}
    for pt in data[hue].unique():
        subset = data[data[hue] == pt].dropna(subset=[x_compound, y_compound])
        if subset.empty or len(subset[x_compound].unique()) < 2:
            print(f"Insufficient data for {pt} slope calculation.")
            continue
        slope, intercept, r_value, p_value, std_err = linregress(subset[x_compound], subset[y_compound])
        slopes[pt] = (slope, std_err)
    product_types = list(slopes.keys())
    # if len(product_types) == 2:  # Only compare if exactly two product types
    #     diff = abs(slopes[product_types[0]][0] - slopes[product_types[1]][0])
    #     se_diff = np.sqrt(slopes[product_types[0]][1]**2 + slopes[product_types[1]][1]**2)
    #     significant_diff = diff > 2 * se_diff  # Using 2*SE as a heuristic threshold for significance
    for i, pt in enumerate(product_types):
        slope, std_err = slopes[pt]
        annotation = f"Slope ({pt}): {slope:.2f}"
        # if len(product_types) == 2 and significant_diff:
        #     annotation += " *"
        plt.text(
            0.15,
            0.75 - i * 0.05,
            annotation,
            transform=lm.fig.transFigure,
            fontsize=24
        )
    plt.title(title if title else f'{y_compound} to {x_compound}')
    plt.xlabel(xlabel if xlabel else x_compound.replace('_', ' ').title())
    plt.ylabel(ylabel if ylabel else y_compound.replace('_', ' ').title())
    if ymax:
        plt.ylim(0, ymax)
    plt.tight_layout()
    plt.legend(title='Product Type', loc='lower right')
    outfile = f'{assets_dir}/{y_compound.replace("_", "-")}-to-{x_compound.replace("_", "-")}.pdf'
    plt.savefig(outfile, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

# Restrict to flower and concentrate.
sample = results.copy()
sample = sample.loc[sample['product_category'].isin(['Flower', 'Concentrate'])]

# Terpene ratio: Alpha-Humulene to Beta-Caryophyllene
# Justification: common enzyme.
sample['alpha_humulene'] = pd.to_numeric(sample['alpha_humulene'], errors='coerce')
sample['beta_caryophyllene'] = pd.to_numeric(sample['beta_caryophyllene'], errors='coerce')
plot_ratio(
    sample.loc[(sample['alpha_humulene'] >= 0.01) & (sample['beta_caryophyllene'] >= 0.01)],
    'beta_caryophyllene',
    'alpha_humulene',
    title='Alpha-Humulene to Beta-Caryophyllene',
    hue='product_category',
)

# Camphene to D-Limonene
sample['camphene'] = pd.to_numeric(sample['camphene'], errors='coerce')
sample['d_limonene'] = pd.to_numeric(sample['d_limonene'], errors='coerce')
plot_ratio(
    sample.loc[(sample['d_limonene'] >= 0.01) & (sample['camphene'] >= 0.01)],
    'd_limonene',
    'camphene',
    title='Camphene to D-Limonene',
    hue='product_category',
)

# Cannabinoid ratio: THCA to Delta-9 THC
sample = results.copy()
sample['delta_9_thc'] = pd.to_numeric(sample['delta_9_thc'], errors='coerce')
sample['thca'] = pd.to_numeric(sample['thca'], errors='coerce')
plt.figure(figsize=(15, 9))
sns.scatterplot(
    data=sample.loc[
        (sample['product_category'] == 'Concentrate') &
        (sample['product_subcategory'] != 'Other') &
        (sample['thca'] >= 0.01) &
        (sample['delta_9_thc'] >= 0.01)
    ].sort_values('product_subcategory'),
    y='thca',
    x='delta_9_thc',
    hue='product_subcategory',
    palette=concentrate_colors,
    alpha=0.7,
    s=222,
)
plt.title('THCA to Delta-9 THC in Concentrates', pad=20)
plt.ylabel('THCA')
plt.xlabel('Delta-9 THC')
leg = plt.legend(title='Product Subtype', bbox_to_anchor=(1.05, 1), loc='upper left')
for leg_entry in leg.legendHandles: 
    leg_entry.set_sizes([222])
plt.tight_layout()
plt.savefig(f'{assets_dir}/thca-to-delta-9-thc-concentrates.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Cannabinoid ratio: THCA to delta-9 THC.
sample = results.copy()
sample['delta_9_thc'] = pd.to_numeric(sample['delta_9_thc'], errors='coerce')
sample['thca'] = pd.to_numeric(sample['thca'], errors='coerce')
sample = sample.loc[~sample['product_subtype'].isin(['Diamond Infused', 'Distillate Infused', 'Live Resin', 'Live Resin Infused', 'Non Infused'])]
plt.figure(figsize=(15, 9))
sns.scatterplot(
    data=sample.loc[
        (sample['product_category'] == 'Flower') &
        (sample['thca'] >= 0.01) &
        (sample['delta_9_thc'] >= 0.01)
    ],
    y='thca',
    x='delta_9_thc',
    hue='product_subtype',
    palette='bright',
    alpha=0.7,
    s=222,
)
plt.title('THCA to Delta-9 THC in Flower', pad=20)
plt.ylabel('THCA')
plt.xlabel('Delta-9 THC')
leg = plt.legend(title='Product Subtype', bbox_to_anchor=(1.05, 1), loc='upper left')
for leg_entry in leg.legendHandles: 
    leg_entry.set_sizes([222])
plt.tight_layout()
plt.savefig(f'{assets_dir}/thca-to-delta-9-thc-flower.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


# === Pre-roll Analysis ===

# TODO: Compare total cannabinoids and total terpenes between prerolls and flower.




# === Edible Analysis ===

# Find solid edibles.

# TODO: Analyze edibles.




# === Concentrate Analysis ===

# Compare chemical profiles of different types of concentrates.
subtypes = [
    'Badder',
    'Cartridge',
    'Crumble',
    'Diamonds',
    'Distillate',
    'Hash',
    'Live Resin',
    'Live Rosin',
    'Sauce',
    'Shatter',
    'Sugar',
    'Vape'
]
concentrates = results.loc[results['product_category'] == 'Concentrate']

# Assign product_subtype based on product_type.
concentrates['product_subtype'] = concentrates['product_type'].apply(assign_product_subtype, subtypes=subtypes)

# FIXME: Scatterplot of total_cannabinoids to total_terpenes with hue as product_subtype.
# plt.figure(figsize=(10, 8))
# sns.scatterplot(
#     data=concentrates.loc[concentrates['product_subtype'] != 'Other'],
#     y='total_cannabinoids',
#     x='total_terpenes',
#     hue='product_subtype',
#     style='product_subtype',
#     palette='bright',
# )
# plt.title('Chemical Profile Comparison of Concentrate Subtypes', pad=20)
# plt.ylabel('Total Cannabinoids')
# plt.xlabel('Total Terpenes')
# plt.legend(title='Product Subtype', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.ylim(0, 100)
# plt.show()

# TODO: Look at concentrations by product type.




# === Moisture Analysis ===

# TODO: Look at moisture content between months (December 2023 to January 2024).


# TODO: Look at water activity between months (December 2023 to January 2024).



# === Trend Analysis ===

# Define when standard method was effective.
# Source: https://cannabis.ca.gov/cannabis-laws/rulemaking/standard-cannabinoids-test-method-and-standardized-operating-procedures/
effective_date = pd.to_datetime('2023-10-01')
compliance_date = pd.to_datetime('2024-01-01')

# Ensure 'date_tested' is in datetime format
results['date_tested'] = pd.to_datetime(results['date_tested'], errors='coerce')
results['month'] = results['date_tested'].dt.month
results['year'] = results['date_tested'].dt.year

# Filter data for December 2023 to January 2024
sample = results.copy()
sample = sample[(sample['date_tested'] >= '2023-08-01') & (sample['date_tested'] <= '2024-01-31')]


def plot_trend_line(data, date_column, value_column, poly_degree=1, line_color='gray', linestyle="--", label_prefix=''):
    """Plot a trend line with reduced points and add to legend."""
    z = np.polyfit(mdates.date2num(data[date_column]), data[value_column], poly_degree)
    p = np.poly1d(z)
    dates = pd.date_range(data[date_column].min(), data[date_column].max(), periods=10)  # Reduced points
    dates_num = mdates.date2num(dates)
    direction = '+' if np.sign(z[0]) > 0 else ''
    trend = z[0] * 7
    plt.plot(
        dates,
        p(dates_num),
        color=line_color,
        linestyle=linestyle,
        lw=5,
        label=f'{label_prefix} Trend: {direction}{trend:.1f}% / week')


def plot_timeseries(
        df,
        title,
        effective_date,
        compliance_date,
        x='date_tested',
        y='total_thc',
        outfile=None,
    ):
    """
    Plot the timeseries data with dots for actual values and separate trend lines 
    for periods before and after the compliance date.
    """
    plt.figure(figsize=(15, 8))
    dot_color = 'royalblue'
    line_color = 'navy'
    trend_color_before = 'crimson'
    trend_color_after = 'mediumseagreen'
    
    # Plot actual THC values as dots
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        color=dot_color,
        s=75,
        alpha=0.6,
    )

    # Plot weekly moving average.
    weekly = df.resample('W', on=x)[y].mean().reset_index()
    sns.lineplot(
        data=weekly,
        x=x,
        y=y,
        color=line_color,
    )
    
    # Separate data into before and after compliance date
    before_compliance = df[df[x] < compliance_date]
    after_compliance = df[df[x] >= compliance_date]
    if not before_compliance.empty:
        plot_trend_line(before_compliance, x, y, poly_degree=1, line_color=trend_color_before, linestyle="--", label_prefix='Before Compliance')
    if not after_compliance.empty:
        plot_trend_line(after_compliance, x, y, poly_degree=1, line_color=trend_color_after, linestyle="--", label_prefix='After Compliance')

    # Add vertical lines for effective and compliance dates.
    texts = []
    plt.axvline(
        x=effective_date,
        color='black',
        linestyle='--',
        lw=2,
    )
    plt.axvline(
        x=compliance_date,
        color='black',
        linestyle='--',
        lw=2,
    )
    text = plt.text(
        effective_date,
        plt.gca().get_ylim()[1] * 0.75,
        'Effective Date',
        color='black',
        ha='right',
    )
    texts.append(text)
    text = plt.text(
        compliance_date,
        plt.gca().get_ylim()[1] * 0.75,
        'Compliance Date',
        color='black',
        ha='right',
    )
    texts.append(text)
    adjust_text(texts)
    plt.title(title, pad=20)
    plt.xlabel('Date')
    plt.ylabel('Total THC (%)')
    plt.legend(loc='lower left')
    plt.tight_layout()
    if outfile is None:
        outfile = f'{assets_dir}/{y.replace("_", "-")}-timeseries.pdf'
    plt.savefig(outfile, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

# Visualize total THC in flower.
sample = results.loc[results['date_tested'] > pd.to_datetime('2023-08-01')]
sample = sample.loc[sample['product_category'] == 'Flower']
sample = sample.loc[(sample['total_thc'] < 50) & sample['total_thc'] > 0]
plot_timeseries(
    sample,
    title='Weekly Average Total THC in Flower',
    effective_date=effective_date,
    compliance_date=compliance_date,
    x='date_tested',
    y='total_thc',
    outfile=f'{assets_dir}/total-thc-flower-timeseries.pdf',
)

# Visualize total cannabinoids in flower.
sample = results.loc[results['date_tested'] > pd.to_datetime('2023-08-01')]
sample = sample.loc[sample['product_category'] == 'Flower']
sample = sample.loc[(sample['total_cannabinoids'] < 50) & sample['total_cannabinoids'] > 0]
plot_timeseries(
    sample,
    title='Weekly Average Total Cannabinoids in Flower',
    effective_date=effective_date,
    compliance_date=compliance_date,
    x='date_tested',
    y='total_cannabinoids',
    outfile=f'{assets_dir}/total-cannabinoids-flower-timeseries.pdf',
)

# Visualize total THC in concentrates.
sample = results.loc[results['date_tested'] > pd.to_datetime('2023-08-01')]
sample = sample.loc[sample['product_category'] == 'Concentrate']
sample = sample.loc[(sample['total_thc'] < 100) & sample['total_thc'] > 0]
plot_timeseries(
    sample,
    title='Weekly Average Total THC in Concentrates',
    effective_date=effective_date,
    compliance_date=compliance_date,
    x='date_tested',
    y='total_thc',
    outfile=f'{assets_dir}/total-thc-concentrates-timeseries.pdf',
)

# Visualize total cannabinoids in concentrates.
sample = results.loc[results['date_tested'] > pd.to_datetime('2023-08-01')]
sample = sample.loc[sample['product_category'] == 'Concentrate']
sample = sample.loc[(sample['total_cannabinoids'] < 100) & sample['total_cannabinoids'] > 0]
plot_timeseries(
    sample,
    title='Weekly Average Total Cannabinoids in Concentrates',
    effective_date=effective_date,
    compliance_date=compliance_date,
    x='date_tested',
    y='total_cannabinoids',
    outfile=f'{assets_dir}/total-cannabinoids-concentrates-timeseries.pdf',
)

# Visualize total terpenes in flower.
sample = results.loc[results['date_tested'] > pd.to_datetime('2023-08-01')]
sample = sample.loc[(sample['total_terpenes'] < 50) & sample['total_terpenes'] > 0]
sample = sample.loc[sample['product_category'] == 'Flower']
plot_timeseries(
    sample,
    title='Weekly Average Total Terpenes in Flower',
    effective_date=effective_date,
    compliance_date=compliance_date,
    x='date_tested',
    y='total_terpenes',
    outfile=f'{assets_dir}/total-terpenes-flower-timeseries.pdf',
)

# Visualize total terpenes in concentrates.
sample = results.loc[results['date_tested'] > pd.to_datetime('2023-08-01')]
sample = sample.loc[sample['product_category'] == 'Concentrate']
sample = sample.loc[(sample['total_terpenes'] < 100) & sample['total_terpenes'] > 0]
plot_timeseries(
    sample,
    title='Weekly Average Total Terpenes in Concentrates',
    effective_date=effective_date,
    compliance_date=compliance_date,
    x='date_tested',
    y='total_terpenes',
    outfile=f'{assets_dir}/total-terpenes-concentrates-timeseries.pdf',
)

# Visualize the distribution of total THC in December 2023 to January 2024.
sample = results.loc[results['date_tested'] > pd.to_datetime('2023-08-01')]
sample = sample.loc[sample['product_category'] == 'Flower']
sample = sample.loc[(sample['total_thc'] < 50) & sample['total_thc'] > 0]
december_flower = sample.loc[
    (sample['date_tested'] >= pd.to_datetime('2023-12-01'))
    & (sample['date_tested'] <= pd.to_datetime('2023-12-31'))
]
january_flower = sample.loc[
    (sample['date_tested'] >= pd.to_datetime('2024-01-01'))
    & (sample['date_tested'] <= pd.to_datetime('2024-01-31'))
]
plt.figure(figsize=(12, 7.25))
sns.histplot(
    data=december_flower,
    x='total_thc',
    kde=True,
    bins=25,
    color='crimson',
    label='December 2023',
)
sns.histplot(
    data=january_flower,
    x='total_thc',
    kde=True,
    bins=25,
    color='mediumseagreen',
    label='January 2024',
)
plt.axvline(
    x=december_flower['total_thc'].mean(),
    color='black',
    linestyle='--',
    lw=2,
)
plt.text(
    december_flower['total_thc'].mean(),
    9,
    f'Dec 2023 Avg: {december_flower["total_thc"].mean():.1f}%',
    color='black',
    ha='right',
)
plt.axvline(
    x=january_flower['total_thc'].mean(),
    color='black',
    linestyle='--',
    lw=2,
)
plt.text(
    january_flower['total_thc'].mean(),
    7,
    f'Jan 2024 Avg: {january_flower["total_thc"].mean():.1f}%',
    color='black',
    ha='right',
)
plt.legend(loc='upper right')
plt.title('Total THC in Flower', pad=20)
plt.xlabel('Total THC (%)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(f'{assets_dir}/total-thc-histogram.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


# === Statistical Analysis ===

from scipy.stats import ttest_ind

# Define when standard method was effective.
# Source: https://cannabis.ca.gov/cannabis-laws/rulemaking/standard-cannabinoids-test-method-and-standardized-operating-procedures/
effective_date = pd.to_datetime('2023-10-01')
compliance_date = pd.to_datetime('2024-01-01')

# Test if the regulation was effective.
# Use a difference-in-differences analysis to test if the regulation was effective.
# Filter data before and after the compliance date for 'Flower' category as an example
sample = results.copy()
sample = sample.loc[sample['product_category'] == 'Flower']
before_compliance = sample[sample['date_tested'] < compliance_date]
after_compliance = sample[sample['date_tested'] >= compliance_date]

# Test if total THC changed.
t_stat, p_value = ttest_ind(
    before_compliance['total_thc'].dropna(),
    after_compliance['total_thc'].dropna(),
)
summary_stats = pd.DataFrame({
    "Statistic": ["T-statistic", "P-value"],
    "Value": [t_stat, p_value]
})
print(summary_stats)
latex_code = summary_stats.to_latex(index=False, header=True)
with open(f'{report_dir}/compliance-test-total-thc.tex', 'w') as file:
    file.write(latex_code)

# Test if total cannabinoids changed.
t_stat, p_value = ttest_ind(
    before_compliance['total_cannabinoids'].dropna(),
    after_compliance['total_cannabinoids'].dropna(),
)
summary_stats = pd.DataFrame({
    "Statistic": ["T-statistic", "P-value"],
    "Value": [t_stat, p_value]
})
print(summary_stats)
latex_code = summary_stats.to_latex(index=False, header=True)
with open(f'{report_dir}/compliance-test-total-cannabinoids.tex', 'w') as file:
    file.write(latex_code)

# Test if total terpenes changed.
t_stat, p_value = ttest_ind(
    before_compliance['total_terpenes'].dropna(),
    after_compliance['total_terpenes'].dropna(),
)
summary_stats = pd.DataFrame({
    "Statistic": ["T-statistic", "P-value"],
    "Value": [t_stat, p_value]
})
print(summary_stats)
latex_code = summary_stats.to_latex(index=False, header=True)
with open(f'{report_dir}/compliance-test-total-terpenes.tex', 'w') as file:
    file.write(latex_code)


# === Lab Variation Analysis ===

# Count the number of results by lab
print(results['lab'].value_counts())
results['lab_uid'] = results['lab'].factorize()[0]

# Output lab value counts to LaTeX.
lab_counts = results['lab'].value_counts().head(15)
lab_counts_df = lab_counts.reset_index()
lab_counts_df.columns = ['Top 15 Labs Observed', 'Observations']
latex_code = lab_counts_df.to_latex(index=False, header=True)
print(latex_code)
with open(f'{report_dir}/top-labs.tex', 'w') as latex_file:
    latex_file.write(latex_code)


def preprocess_and_aggregate_by_lab(df):
    """
    Preprocess the data and aggregate it to calculate monthly averages of total THC by lab.
    """
    df['date'] = pd.to_datetime(df['date_tested'], errors='coerce')
    flower_types = [
        'Flower', 'Flower, Inhalable', 'Flower, Product Inhalable', 
        'Flower, Medical Inhalable', 'Plant (Flower - Cured)', 'Plant (Bulk Flower)'
    ]
    df = df.loc[df['product_type'].isin(flower_types)]
    df = df[['date', 'lab', 'total_thc']].dropna()
    rows = df.loc[(df['total_thc'] > 0) & (df['total_thc'] < 100)]
    avg = rows.groupby(['lab']).resample('W', on='date').mean().reset_index()
    return avg

def plot_timeseries_by_lab(
        df,
        title,
        effective_date,
        compliance_date
    ):
    """
    Plot the timeseries data of monthly average total THC by lab with unique colors, IDs, and scatterplot points.
    """
    plt.figure(figsize=(21, 10.75))

    # Clean the data.
    df['date'] = pd.to_datetime(df['date_tested'], errors='coerce')
    df = df.loc[df['product_category'] == 'Flower']
    df = df.loc[(df['total_thc'] > 0) & (df['total_thc'] < 100)]
    df = df.loc[df['date'] >= pd.to_datetime('2023-08-01')]

    # Preprocess and aggregate data for visualization
    group = preprocess_and_aggregate_by_lab(df)

    labs = group['lab'].unique()
    colors = sns.color_palette('tab20', n_colors=len(labs))
    lab_uid_map = {lab: uid for uid, lab in enumerate(labs)}
    
    for lab in labs:
        lab_data = group[group['lab'] == lab]
        # Line plot for trend
        sns.lineplot(
            x='date',
            y='total_thc',
            data=lab_data,
            label=f'Lab {lab_uid_map[lab] + 1}',
            color=colors[lab_uid_map[lab]],
        )
        # Scatter plot for actual data points
        sns.scatterplot(
            x='date',
            y='total_thc',
            data=df.loc[df['lab'] == lab],
            color=colors[lab_uid_map[lab]],
            alpha=0.6,
            s=120,
            edgecolor='none'  # Remove edge color for cleaner look
        )

    # Highlight effective and compliance dates
    plt.axvline(x=effective_date, color='dimgray', linestyle='--', lw=2)
    plt.axvline(x=compliance_date, color='dimgray', linestyle='--', lw=2)
    plt.text(effective_date, 45, 'Effective Date', color='black', ha='right')
    plt.text(compliance_date, 45, 'Compliance Date', color='black', ha='right')
    plt.title(title, pad=20)
    plt.xlabel('Date')
    plt.ylabel('Average Total THC (%)')
    plt.legend(title='Lab', loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.ylim(15, 50)
    plt.xlim(pd.to_datetime('2023-09-01'), pd.to_datetime('2024-02-01'))
    plt.tight_layout()
    plt.savefig(f'{assets_dir}/total-thc-by-lab-timeseries.pdf', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

# Plotting the timeseries for flower_co by lab
plot_timeseries_by_lab(
    results,
    'Average Total THC in Flower by Lab',
    effective_date=pd.to_datetime('2023-10-01'),
    compliance_date=pd.to_datetime('2024-01-01'),
)
plt.show()


# === PCA Analysis ===

# PCA of major terpenes.
features = ['beta_caryophyllene', 'd_limonene', 'beta_myrcene', 'alpha_pinene', 'linalool'] # 'alpha_terpinolene'
X = results[features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_
pca = PCA(n_components=2)
pca_scores = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(pca_scores, columns=['PC1', 'PC2'])
pca_df['Cluster'] = labels
pca_df['dominant_terpene'] = dominant_terpene
pca_df['product_type'] = results['product_type']
pca_df['strain_name'] = results['strain_name']
pca_df['producer'] = results['producer']
pca_df['lab_uid'] = results['lab_uid']

# Visualize PCA by dominant terpenes.
subsample = pca_df.loc[pca_df['dominant_terpene'].isin(['beta_caryophyllene', 'd_limonene', 'beta_myrcene', 'alpha_pinene', 'linalool'])]
plt.figure(figsize=(15, 8))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='dominant_terpene',
    data=subsample.sort_values(by='dominant_terpene'),
    palette='bright',
    s=222,
    alpha=0.7,
    legend='full',
)
leg = plt.legend(title='Dominant Terpene', loc='upper right', bbox_to_anchor=(1.05, 1),)
for leg_entry in leg.legendHandles: 
    leg_entry.set_sizes([222])
plt.title('PCA of Major Terpenes by Dominant Terpene', pad=20)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.savefig(f'{assets_dir}/pca-dominant-terpenes.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Visualize PCA by lab.
plt.figure(figsize=(15, 8))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='lab_uid',
    data=pca_df.sort_values(by='lab_uid'),
    palette='tab10',
    s=200,
    alpha=0.7,
    legend=None,
)
plt.title('PCA of Major Terpenes by Lab', pad=20)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.savefig(f'{assets_dir}/pca-by-lab.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Visualize PCA by producer.
plt.figure(figsize=(15, 8))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='producer',
    data=pca_df.sort_values(by='producer'),
    palette='tab10',
    s=200,
    alpha=0.7,
    legend=None,
)
plt.title('PCA of Major Terpenes by Producer', pad=20)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.savefig(f'{assets_dir}/pca-by-producer.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


# === Producer Variation Analysis ===

# TODO: Look at metric, producer by producer.


print('Report generation complete.')
