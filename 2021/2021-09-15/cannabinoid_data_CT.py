"""
Cannabinoid Data in Connecticut | Cannabis Data Science

Author: Keegan Skeate <keegan@cannlytics.com>
Created: 9/14/2021
Updated: 9/16/2021
License: MIT License <https://opensource.org/licenses/MIT>

Data Source:
    Connecticut Medical Marijuana Brand Registry: https://data.ct.gov/Health-and-Human-Services/Medical-Marijuana-Brand-Registry/egd5-wb6r/data
    Connecticut Socrata Open Data API: https://dev.socrata.com/foundry/data.ct.gov/egd5-wb6r
"""
# External imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sodapy import Socrata

# Internal imports
from cannlytics_charts import crispy_bar_chart

# Define plot style.
plt.style.use('fivethirtyeight')

# Define cannabinoids and terpenes
CANNABINOIDS = [
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
TERPENES = [
    'a_pinene',
    'b_myrcene',
    'b_caryophyllene',
    'b_pinene',
    'limonene',
    'ocimene',
    'linalool_lin',
    'humulene_hum',
    # 'b_eudesmol', # Ignored because ubiquitous.
    # 'fenchone', # Ignored because ubiquitous.
    # 'camphor', # Ignored because ubiquitous.
    'a_bisabolol',
    'a_phellandrene',
    'a_terpinene',
    'b_terpinene',
    'pulegol',
    'borneol',
    'isopulegol',
    'carene',
    'camphene',
    'caryophyllene_oxide',
    'cedrol',
    'eucalyptol',
    'geraniol',
    'guaiol',
    'geranyl_acetate',
    'isoborneol',
    'menthol',
    'l_fenchone',
    'nerol',
    'sabinene',
    'terpineol',
    'terpinolene',
    'trans_b_farnesene',
    'valencene',
    'a_cedrene',
    'a_farnesene',
    'b_farnesene',
    'cis_nerolidol',
    'fenchol',
    'trans_nerolidol'
]

#--------------------------------------------------------------------------
# 1. Get the data.
#--------------------------------------------------------------------------

# Get the cannabinoid data.
client = Socrata('data.ct.gov', None)
response = client.get('egd5-wb6r', limit=15000)
data = pd.DataFrame.from_records(response)

# Convert values to floats, coding suspected non-detects as 0.
for analyte in TERPENES + CANNABINOIDS:
    data[analyte] = data[analyte].str.replace('<0.10', '0.0', regex=False)
    data[analyte] = data[analyte].str.replace('<0.1', '0.0', regex=False)
    data[analyte] = data[analyte].str.replace('<0.29', '0.0', regex=False)
    data[analyte] = data[analyte].str.replace('%', '', regex=False)
    data[analyte] = data[analyte].str.replace('-', '0.0', regex=False)
    data[analyte] = pd.to_numeric(data[analyte], errors='coerce').fillna(0.0)

# Calculate total terpenes and total cannabinoids.
data['total_terpenes'] = data[TERPENES].sum(axis=1)
data['total_cannabinoids'] = data[CANNABINOIDS].sum(axis=1)

#--------------------------------------------------------------------------
# 2. Look at the data.
#--------------------------------------------------------------------------

# Calculate the prevalence (percent of samples that contains) of each terpene.
# Also, calculate the average for each terpene when the terpene is present.
averages = {}
prevalence = {}
analytes = pd.DataFrame(columns=['analyte', 'concentration', 'brand_name',
                                 'dosage_form', 'producer'])
for analyte in TERPENES:
    analyte_present_data = data.loc[data[analyte] > 0].copy(deep=True)
    prevalence[analyte] = len(analyte_present_data) / len(data)
    averages[analyte] = analyte_present_data[analyte].mean()
    print('--------------\n%s' % analyte)
    print('Prevalence:', round(prevalence[analyte], 4))
    print('Avg. Concentration:', round(averages[analyte], 4))
    analyte_present_data['analyte'] = analyte
    analyte_present_data['concentration'] = analyte_present_data[analyte]
    subset = analyte_present_data[['analyte', 'concentration', 'brand_name',
                                   'dosage_form', 'producer']]
    analytes = analytes.append(subset)

# Create a DataFrame with statistics for each analyte.
terpene_data = pd.DataFrame(
    prevalence.items(),
    columns=['analyte', 'prevalence'],
    index=prevalence.keys()
)

# Sort the data by the most prevelant terpene.
terpene_data = terpene_data.sort_values('prevalence', ascending=False)

#--------------------------------------------------------------------------
# 3. Visualize the data.
#--------------------------------------------------------------------------
from cannlytics_charts import crispy_bar_chart

# Isolate top 10 most prevelant terpenes.
terpene_data = terpene_data.sort_values('prevalence', ascending=True)
top_ten = terpene_data[-10:].copy(deep=True)
top_ten['prevalence'] = top_ten.prevalence * 100

# Clean Y labels.
y_ticks = [s \
    .replace('_', ' ') \
    .replace(' lin', '') \
    .replace(' hum', '') \
    .title() \
    .replace('B ', 'β-') \
    .replace('A ', 'α-') \
    .replace('Trans ', 'trans-') \
    .replace('Cis ', 'cis-') \
    for s in list(top_ten.index)
]
    
# Define a color palette.
palette = sns.color_palette('Set2', n_colors=10)

# Write notes.
notes = """Data: 10,824 brand analyses between 3/24/2015 and 9/15/2021.
Data Source: Connecticut Medical Marijuana Brand Registry.
Notes: The terpenes β-eudesmol, fenchone, and camphor were present in more than 95% of
samples, so they were excluded from the top ten becaues they appear to be ubiquitous."""

# Create a bar chart of terpene prevalence.
# crispy_bar_chart(
#     top_ten,
#     annotations=True,
#     key='prevalence',
#     title='Top Ten Terpenes Found in Connecticut Cannabis',
#     fig_size=(9, 7),
#     text_color='#1a1a1a',
#     notes=notes,
#     notes_offset=.125,
#     palette=palette,
#     percentage=True,
#     y_ticks=y_ticks,
#     x_label='Prevalence (percent of samples where detected)',
#     save='figures/top-ten-terpenes-in-connecticut-cannabis.png',
#     zero_bound=True,
# )

#--------------------------------------------------------------------------
# 4. Analyze the data.
#--------------------------------------------------------------------------

# # Correlogram of terpenes.
# plt.figure(figsize=(12,10), dpi= 80)
# sns.heatmap(
#     data[TERPENES].corr(),
#     xticklabels=data[TERPENES].corr().columns,
#     yticklabels=data[TERPENES].corr().columns,
#     cmap='RdYlGn',
#     center=0,
#     annot=False
# )
# plt.title('Correlogram of Terpenes', fontsize=22)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()

# # Scatterplot of CBDA to THCA.
# fig, ax = plt.subplots(figsize=(8, 5), dpi=80)
# plt.scatter(
#     x='tetrahydrocannabinol_acid_thca',
#     y='cannabidiol_acid_cbda',
#     s=20,
#     color='#007acc',
#     alpha=0.6,
#     data=data.loc[(data.b_caryophyllene < 100) &
#                   (data.cannabidiols_cbd < 100)], 
    
# )
# plt.gca().set(
#     xlim=(0.0, 100),
#     ylim=(0, 100),
#     xlabel='thca',
#     ylabel='cbda'
# )
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# plt.xticks(fontsize=12); plt.yticks(fontsize=12)
# plt.title('cbda to thca in connecticut cannabis', fontsize=21)   
# plt.show()

# # Scatterplot of total terpenes vs. total cannabinoids by producer.
# categories = list(data.producer.unique())
# categories = [x.title().replace('Llc', 'LLC') for x in categories]
# colors = sns.color_palette('Set2', n_colors=len(categories))
# plt.figure(figsize=(16, 10), dpi= 80)
# for i, category in enumerate(categories):
#     plt.scatter(
#         x='total_cannabinoids',
#         y='total_terpenes',
#         data=data.loc[data.producer==category.upper(), :], 
#         s=20,
#         c=colors[i],
#         label=str(category)
#     )

# plt.gca().set(
#     xlim=(0.0, 100),
#     ylim=(0, 10),
#     xlabel='Total Cannabinoids',
#     ylabel='Total Terpenes'
# )
# plt.xticks(fontsize=12); plt.yticks(fontsize=12)
# plt.title('Total Terpenes vs Total Cannabinoids by Producer in Connecticut Cannabis', fontsize=22)
# plt.legend(fontsize=12, loc='upper left')
# plt.savefig('figures/total-terpenes-vs-cannabinoids.png', dpi=300, bbox_inches='tight')
# plt.show()
