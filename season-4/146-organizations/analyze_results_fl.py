"""
Cultivator Report: Jungle Boys

Copyright (c) 2024 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 2/15/2024
Updated: 2/16/2024
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Data Source:

    - [Jungle Boys Florida](https://jungleboysflorida.com/)

"""
# Standard imports:
from time import sleep

# External imports:
from adjustText import adjust_text
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# === Setup ===

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
})

# Define the report directory.
report_dir = './cultivator-report/stats'
assets_dir = './cultivator-report/assets/figures'


# === Get the data ===

# Read the parsed Jungle Boys Florida lab results.
datafile = 'https://cannlytics.page.link/fl-lab-results-jungleboys-2024-02-15'
coa_data = pd.read_excel(datafile)

# Define cannabinoids and terpenes.
cannabinoids = [
    'delta_9_thc',
    'thca',
    'cbd',
    'cbda',
    'delta_8_thc',
    'cbg',
    'cbga',
    'cbn',
    'thcv',
    'cbdv',
    'cbc',
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
    'alpha_terpinolene',
    'nerolidol',
    'gamma_terpinene',
    'trans_nerolidol'
]


# === Summary Statistics ===

# Calculate the total number of strains.
total_strains = len(coa_data['strain_name'].unique())

# Count the number of parsed COAs.
total_parsed_coas = len(coa_data)

# Count the number of analytes.

# Output summary statistics to LaTeX.
summary = pd.DataFrame({
    'Total Parsed COAs': [total_parsed_coas],
    'Total Strains': [total_strains],
    'Number of Cannabinoids': [len(cannabinoids)],
    'Number of Terpenes': [len(terpenes)],
})
output = summary.T
output.columns = ['Observations']
latex_code = output.to_latex(index=True, header=False)
latex_code = latex_code.replace('\n\\midrule', '')
print(latex_code)
with open(f'{report_dir}/summary.tex', 'w') as file:
    file.write(latex_code)

# Visualize strain types.
plt.figure(figsize=(8, 6))
colors = ["#78C850", "#F85888", "#F8D030", "#6890F0", "#705898"]
n_bins = 100
cmap_name = "strain_gradient"
strain_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
order = ['Flower', 'Derivative']
coa_data['product_type'] = coa_data['product_type'].str.title()
sns.countplot(
    x='product_type',
    data=coa_data,
    order=order,
    palette=np.array(strain_cmap(np.linspace(0, 1, len(order)))),
)
plt.xlabel('')
plt.ylabel('Count')
plt.title('Count of Product Types')
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(f'{assets_dir}/product-types.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Pie chart of product types.
plt.figure(figsize=(15, 8.5))
colors = ["#78C850", "#F85888", "#F8D030", "#6890F0", "#705898"]
n_bins = 100
cmap_name = "strain_gradient"
strain_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
order = ['Flower', 'Derivative']
coa_data['product_type'] = coa_data['product_type'].str.title()
product_type_counts = coa_data['product_type'].value_counts().reindex(order, fill_value=0)
plt.pie(product_type_counts, labels=product_type_counts.index, colors=np.array(strain_cmap(np.linspace(0, 1, len(order)))), autopct='%1.1f%%', startangle=140)
plt.title('Proportion of Product Types')
plt.tight_layout()
plt.savefig(f'{assets_dir}/product-types-pie-chart.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


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



# Calculate diversity of terpenes.
coa_data['cannabinoid_diversity'] = calculate_shannon_diversity(coa_data, cannabinoids)
coa_data['terpene_diversity'] = calculate_shannon_diversity(coa_data, terpenes)

# Scatter plot of cannabinoid vs. terpene diversity
plt.figure(figsize=(14, 8.5))
sns.scatterplot(
    data=coa_data,
    x='cannabinoid_diversity',
    y='terpene_diversity',
    hue='product_type',
    style='product_type', 
    palette='Set2',
    s=200,
)
plt.title('Terpene and Cannabinoid Diversity by Strain')
plt.xlabel('Cannabinoid Diversity')
plt.ylabel('Terpene Diversity')
leg = plt.legend(title='Strain Type', bbox_to_anchor=(1.05, 1), loc='upper left')
for leg_entry in leg.legendHandles: 
    leg_entry.set_sizes([200])
plt.grid(True)
plt.tight_layout()
# for i in range(coa_data.shape[0]):
#     plt.text(
#         coa_data.iloc[i]['cannabinoid_diversity'],
#         coa_data.iloc[i]['terpene_diversity'],
#         coa_data.iloc[i]['strain_name'],
#         ha='left',
#         size='14',
#         color='black',
#     )
plt.tight_layout()
plt.savefig(f'{assets_dir}/diversity-scatterplot.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Average by strain type.
flower = coa_data[coa_data['product_type'] == 'Flower']
strain_avg = flower.groupby('strain_name', as_index=False)[['cannabinoid_diversity', 'terpene_diversity']].mean()

# Sorting data for better visualization
sorted_cannabinoids = strain_avg.sort_values(by='cannabinoid_diversity', ascending=False)
sorted_terpenes = strain_avg.sort_values(by='terpene_diversity', ascending=False)
plt.figure(figsize=(18, 21))
plt.subplot(1, 2, 1)
sns.barplot(data=sorted_cannabinoids, x='cannabinoid_diversity', y='strain_name', palette='flare_r')
plt.title('Cannabinoid Diversity')
plt.xlabel('')
plt.ylabel('')
avg_cannabinoid_diversity = sorted_cannabinoids['cannabinoid_diversity'].mean()
plt.axvline(x=avg_cannabinoid_diversity, color='black', linestyle='--', label=f'Avg: {avg_cannabinoid_diversity:.2f}')
plt.legend(loc='lower right')
plt.subplot(1, 2, 2)
sns.barplot(data=sorted_terpenes, x='terpene_diversity', y='strain_name', palette='flare_r')
plt.title('Terpene Diversity')
plt.xlabel('')
plt.ylabel('')
avg_terpene_diversity = sorted_terpenes['terpene_diversity'].mean()
plt.axvline(x=avg_terpene_diversity, color='black', linestyle='--', label=f'Avg: {avg_terpene_diversity:.2f}')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{assets_dir}/diversity.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


# === Cannabinoid and Terpene Concentrations Analysis ===

# Analyze average concentrations.
stats = coa_data.copy()
stats.replace({'<LOQ': 0, 'ND': np.nan}, inplace=True)
compounds = cannabinoids + terpenes
totals = ['total_cannabinoids', 'total_terpenes', 'total_thc', 'total_cbd']
for compound in compounds + totals:
    stats[compound] = pd.to_numeric(stats[compound], errors='coerce')
avg_concentrations = stats.groupby('product_type')[compounds + totals].mean()

# Define colors.
colors = ["#78C850", "#F85888", "#F8D030", "#6890F0", "#705898"]
n_bins = 100
cmap_name = "compound_gradient"
compound_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


def plot_grouped_bar_chart(
        data,
        title,
        ylabel,
        output_filename,
        bar_width=0.15,
        rotation=0,
    ):
    plt.figure(figsize=(14, 8.5))
    n_groups = data.shape[1]
    n_bars = len(data)
    index = np.arange(n_groups) 
    colors = ['#705898', '#78C850']
    for i, (product_type, row) in enumerate(data.iterrows()):
        plt.bar(
            index + i*bar_width,
            row,
            bar_width,
            label=product_type,
            color=colors[i]
        )
    plt.title(title)
    plt.xlabel('Compound')
    plt.ylabel(ylabel)
    plt.xticks(index + bar_width*(n_bars-1)/2, data.columns, rotation=rotation)
    plt.legend(title='Product Type')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

# Create horizontal bar chart of average concentrations:
# total_cannabinoids, delta_9_thc, thca
plot_grouped_bar_chart(
    avg_concentrations[['total_cannabinoids', 'delta_9_thc', 'thca']],
    'Average Concentrations: Major Cannabinoids by Product Type',
    'Concentration (%)',
    'major-cannabinoids-by-product-type.pdf'
)

# Create horizontal bar chart of average concentrations:
# cbd, cbda, total_cbd
plot_grouped_bar_chart(
    avg_concentrations[['cbda', 'cbd', 'total_cbd']],
    'Average Concentrations: CBD and Its Forms by Product Type',
    'Concentration (%)',
    'cbd-concentrations-by-product-type.pdf'
)

# Create horizontal bar chart of average concentrations:
# Minor cannabinoids: all cannabinoids minus delta_9_thc, thca, cbd, cbda
minor_cannabinoids = [compound for compound in cannabinoids if compound not in ['delta_9_thc', 'thca', 'cbd', 'cbda']]
plot_grouped_bar_chart(
    avg_concentrations[minor_cannabinoids],
    'Average Concentrations: Minor Cannabinoids by Product Type',
    'Concentration (%)',
    'minor-cannabinoids-by-product-type.pdf'
)

# Create horizontal bar chart of average concentrations:
# total_terpenes
plot_grouped_bar_chart(
    avg_concentrations[['total_terpenes']],
    'Average Concentrations: Total Terpenes by Product Type',
    'Concentration (%)',
    'total-terpenes-by-product-type.pdf',
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
    'Average Concentrations: Top Terpenes by Product Type',
    'Concentration (%)',
    'top-terpenes-by-product-type.pdf',
    rotation=75,
)


# === Average Concentrations Table ===
# Ensure numeric data for concentration columns, converting '<LOQ' and 'ND' to 0 and NaN, respectively
stats.replace({'<LOQ': 0, 'ND': np.nan}, inplace=True)

# Calculate average concentrations for flowers and concentrates separately
def calculate_averages_by_type(df, compounds, product_type):
    averages = {}
    for compound in compounds:
        df[compound] = pd.to_numeric(df[compound], errors='coerce')
        averages[compound] = df[df['product_type'] == product_type][compound].mean()
    return averages

cannabinoids_averages_flower = calculate_averages_by_type(stats, cannabinoids, 'Flower')
cannabinoids_averages_concentrate = calculate_averages_by_type(stats, cannabinoids, 'Derivative')
terpenes_averages_flower = calculate_averages_by_type(stats, terpenes, 'Flower')
terpenes_averages_concentrate = calculate_averages_by_type(stats, terpenes, 'Derivative')

# Sort the lists alphabetically
cannabinoids_sorted = sorted(cannabinoids)
terpenes_sorted = sorted(terpenes)

# Generate LaTeX table
latex_table = """
\\begin{table}[H]
\\centering
\\begin{tabular}{lccc}
\\multicolumn{4}{l}{Average Concentration of Compounds (\\%)} \\\\
\\hline
& Flower & Derivative & Diff. \\\\
\\hline
\\multicolumn{4}{l}{Cannabinoids} \\\\
\\hline
"""
for compound in cannabinoids_sorted:
    flower_avg = cannabinoids_averages_flower.get(compound, 0)
    concentrate_avg = cannabinoids_averages_concentrate.get(compound, 0)
    diff = round((concentrate_avg - flower_avg) / max(flower_avg, 1) * 100)  # Avoid division by zero
    sign = "+" if diff >= 0 else ""
    latex_table += f"{compound} & {flower_avg:.2f} & {concentrate_avg:.2f} & {sign}{diff}\\% \\\\\n"

latex_table += "\\hline\n\\multicolumn{4}{l}{Terpenes} \\\\\n\\hline\n"
for compound in terpenes_sorted:
    flower_avg = terpenes_averages_flower.get(compound, 0)
    concentrate_avg = terpenes_averages_concentrate.get(compound, 0)
    diff = round((concentrate_avg - flower_avg) / max(flower_avg, 1) * 100)  # Again, avoid division by zero
    sign = "+" if diff >= 0 else ""
    latex_table += f"{compound} & {flower_avg:.2f} & {concentrate_avg:.2f} & {sign}{diff}\\% \\\\\n"

latex_table += "\\hline\n\\end{tabular}\n\\end{table}\n"
latex_table = latex_table.replace('_', '\\_')
print(latex_table)
with open(f'{report_dir}/compounds.tex', 'w') as file:
    file.write(latex_table)



# # Average THC/CBD levels across strains
# stats = extraction.copy()
# stats.replace({'<LOQ': 0, 'ND': np.nan}, inplace=True)
# average_thc_cbd = stats.groupby('strain_name')[['thca', 'total_cbd']].mean()
# average_thc_cbd.columns = ['THCA', 'Total CBD']
# average_thc_cbd.plot(kind='bar', figsize=(8, 5), color=['#17becf', '#b07aa1'])
# plt.title('THCA and Total CBD Concentrations', fontsize=18)
# plt.ylabel('Concentration (%)', fontsize=14)
# plt.xticks(ha='right', rotation=45, fontsize=12)
# plt.xlabel('')
# plt.tight_layout()
# plt.legend(fontsize=12, loc='upper right')
# plt.yticks(np.arange(0, 21, 5), fontsize=12)
# plt.ylim(0, 20)
# plt.savefig(f'{assets_dir}/thca-cbd.pdf', dpi=300, bbox_inches='tight', transparent=True)
# plt.show()


# === Chemical similarity ===

# Plotting function for cannabinoids
def plot_averages_by_type(data, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(14, 8.5))
    data = data.T  # Transpose to have compounds as rows and product types as columns
    bar_width = 0.35
    index = np.arange(data.shape[0])
    product_types = data.columns
    colors = ['#705898', '#78C850']
    for i, product_type in enumerate(product_types):
        ax.bar(
            index + i * bar_width,
            data[product_type],
            width=bar_width,
            label=product_type,
            color=colors[i],
        )
    ax.set_xlabel('Compound')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(index + bar_width / 2 * (len(product_types) - 1))
    ax.set_xticklabels(data.index, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

# Visualize the average concentrations of specific cannabinoids by product type
average_cannabinoid_concentrations_by_type = stats.groupby('product_type')[cannabinoids].mean().drop(columns=['thca', 'delta_9_thc'])
plot_averages_by_type(
    average_cannabinoid_concentrations_by_type,
    'Average Cannabinoid Concentrations by Product Type',
    'Concentration (%)',
    f'{assets_dir}/cannabinoids-by-type.pdf'
)

# Calculate the average concentrations of terpenes by product type
average_terpene_concentrations_by_type = stats.groupby('product_type')[terpenes].mean()
overall_avg_terpenes = average_terpene_concentrations_by_type.mean(axis=0).sort_values(ascending=False).head(10)
average_terpene_concentrations_by_type = average_terpene_concentrations_by_type[overall_avg_terpenes.index]
plot_averages_by_type(
    average_terpene_concentrations_by_type,
    'Top 10 Average Terpene Concentrations by Product Type',
    'Concentration (%)',
    f'{assets_dir}/top-10-terpenes-by-type.pdf'
)


# === Analyze Ratios ===

from scipy.stats import linregress


def plot_ratio_lm(
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
        scatter_kws={'s': 100},
        robust=True,
        ci=None,
        legend=None
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
    if len(product_types) == 2:  # Only compare if exactly two product types
        diff = abs(slopes[product_types[0]][0] - slopes[product_types[1]][0])
        se_diff = np.sqrt(slopes[product_types[0]][1]**2 + slopes[product_types[1]][1]**2)
        significant_diff = diff > 2 * se_diff  # Using 2*SE as a heuristic threshold for significance
    for i, pt in enumerate(product_types):
        slope, std_err = slopes[pt]
        annotation = f"Slope ({pt}): {slope:.2f}"
        if len(product_types) == 2 and significant_diff:
            annotation += " *"
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
    print('Saved:', outfile)
    plt.show()

# Beta-Myrcene to Alpha-Pinene Ratio
plot_ratio_lm(
    stats,
    'alpha_pinene',
    'beta_myrcene',
    title='Beta-Myrcene to Alpha-Pinene',
)

# Beta-Caryophyllene to D-Limonene Ratio
plot_ratio_lm(
    stats,
    'd_limonene',
    'beta_caryophyllene',
    title='Beta-Caryophyllene to D-Limonene'
)

# Visualize ratio of `alpha_pinene` to `d_limonene`
plot_ratio_lm(
    stats,
    'd_limonene',
    'alpha_pinene',
    title='Alpha-Pinene to D-Limonene'
)

# Visualize ratio of `terpinolene` to `beta_myrcene`
plot_ratio_lm(
    stats,
    'beta_myrcene',
    'alpha_terpinolene',
    title='Terpinolene to Beta-Myrcene',
    ymax=0.065
)

# Terpene ratio: Alpha-Humulene to Beta-Caryophyllene
# Justification: common enzyme.
stats['alpha_humulene'] = pd.to_numeric(stats['alpha_humulene'], errors='coerce')
stats['beta_caryophyllene'] = pd.to_numeric(stats['beta_caryophyllene'], errors='coerce')
plot_ratio_lm(
    stats,
    'beta_caryophyllene',
    'alpha_humulene',
    title='Alpha-Humulene to Beta-Caryophyllene'
)

# Terpene ratio: Beta-Myrcene to D-Limonene
plot_ratio_lm(
    stats,
    'd_limonene',
    'beta_myrcene',
    title='Beta-Myrcene to D-Limonene'
)

# Camphene to D-Limonene
stats['camphene'] = pd.to_numeric(stats['camphene'], errors='coerce')
stats['d_limonene'] = pd.to_numeric(stats['d_limonene'], errors='coerce')
plot_ratio_lm(
    stats,
    'd_limonene',
    'camphene',
    title='Camphene to D-Limonene'
)

# Visualize total cannabinoids and total terpenes.
plot_ratio_lm(
    stats,
    'total_terpenes',
    'total_cannabinoids',
    title='Total Terpenes to Total Cannabinoids',
    xlabel='Total Terpenes (%)',
    ylabel='Total Cannabinoids (%)',
)

# Cannabinoid ratio: THCA to Delta-9 THC
stats['delta_9_thc'] = pd.to_numeric(stats['delta_9_thc'], errors='coerce')
stats['thca'] = pd.to_numeric(stats['thca'], errors='coerce')
plot_ratio_lm(
    stats,
    'delta_9_thc',
    'thca',
    title='THCA to Delta-9 THC',
    xlabel='Delta-9 THC (%)',
    ylabel='THCA (%)',
)

# Look at total CBG.
stats['cbg'] = pd.to_numeric(stats['cbg'], errors='coerce')
stats['cbga'] = pd.to_numeric(stats['cbga'], errors='coerce')
stats['total_cbg'] = stats['cbg'] + stats['cbga'].mul(0.877)
plot_ratio_lm(
    stats,
    'cbga',
    'thca',
    title='THCA to CBGA'
)

# CBG to CBD.
stats['cbd'] = pd.to_numeric(stats['cbd'], errors='coerce')
stats['cbda'] = pd.to_numeric(stats['cbda'], errors='coerce')
stats['calculated_total_cbd'] = stats['cbd'] + stats['cbda'].mul(0.877)
plot_ratio_lm(
    stats,
    'calculated_total_cbd',
    'total_cbg',
    title='Total CBG to Total CBD',
    xlabel='Total CBD (%)',
    ylabel='Total CBG (%)',
)


# === Classification Analysis (PCA) ===

# Identify the dominant terpene for each strain.
# for compound in terpenes:
#     stats[compound] = pd.to_numeric(stats[compound].replace({'ND': np.nan, '<LOQ': 0}), errors='coerce')
dominant_terpene = stats[terpenes].idxmax(axis=1)
dominant_terpene_counts = dominant_terpene.value_counts()
dominant_terpene_percentages = (dominant_terpene_counts / dominant_terpene_counts.sum()) * 100
dominant_terpene_percentages.plot(
    kind='barh',
    figsize=(13, 8),
    color='mediumseagreen',
)
for index, value in enumerate(dominant_terpene_percentages):
    plt.text(value, index, f" {value:.0f}%", va='center')
plt.title('Dominant Terpenes Across Strains')
plt.xlabel('Percent of Strains with Dominant Terpene (%)')
plt.ylabel('')
plt.xticks(rotation=0, ha='right')
plt.tight_layout()
plt.savefig(f'{assets_dir}/dominant-terpenes.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Select relevant features for clustering
features = ['beta_caryophyllene', 'd_limonene', 'beta_myrcene', 'alpha_pinene', 'alpha_terpinolene', 'linalool']
X = stats[features].fillna(0)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_
pca = PCA(n_components=2)
pca_scores = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(pca_scores, columns=['PC1', 'PC2'])
pca_df['Cluster'] = labels

# Visualize PCA scores with dominant terpenes.
pca_df['dominant_terpene'] = dominant_terpene
pca_df['product_type'] = stats['product_type']
pca_df['strain_name'] = stats['strain_name']
plt.figure(figsize=(15, 8))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='dominant_terpene',
    data=pca_df.sort_values(by='dominant_terpene'),
    palette='tab10',
    s=200,
    alpha=0.7,
    legend='full',
)
plt.title('PCA of Strains by Dominant Terpenes')
plt.xlabel('PC1')
plt.ylabel('PC2')
leg = plt.legend(title='Dominant Terpene', loc='upper right') # bbox_to_anchor=(1.05, 1),
for leg_entry in leg.legendHandles: 
    leg_entry.set_sizes([200])
texts = []
for _, row in pca_df.iterrows():
    if row['dominant_terpene'] in ['alpha_terpinolene', 'beta_myrcene']:
        text = plt.text(
            row['PC1'],
            row['PC2'],
            row['strain_name'],
            ha='left',
            size='small',
        )
        texts.append(text)
adjust_text(
    texts,
    expand_points=(1.2, 1.2),
)
plt.tight_layout()
plt.savefig(f'{assets_dir}/pca-dominant-terpenes.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
