"""
Cannabis Plant Patents
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 5/21/2022
Updated: 5/25/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    Find and curate data for cannabis patents. In particular, this
    script collects detailed data for plant patents. Subsequent
    intellectual property (IP) analytics provide actionable insights
    for cannabis cultivar inventors and consumers. For example,
    cultivators can use the methodology to predict if a particular
    cultivar would make a good patent candidate given its lab results.
    Consumers can find the nearest patented strain to a set of lab results
    printed on a cultivar's label.

Data Source:

    - United States Patent and Trademark Office
    URL: <www.uspto.gov>

"""
# External imports.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Internal imports.
from cannpatent import (
    curate_lab_results,
    get_patent_details,
    search_patents,
)


# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})


#-----------------------------------------------------------------------
# Demonstration of cannabis plant patent API.
#-----------------------------------------------------------------------

# Search for a term in USPTO patents.
query = 'cannabis plant'
patents = search_patents(query, limit=50, term='TTL%2F')
print('Found %i patents.' % len(patents))

# Restrict to plant patents.
cultivars = patents.loc[
    (patents['patent_title'].str.contains('plant', case=False)) |
    (patents['patent_title'].str.contains('cultivar', case=False))
]
print('Found %i cultivar patents.' % len(cultivars))

# Look up a specific plant patent.
patent = get_patent_details(pd.Series({
    'patent_number': '11240978',
    'patent_title': 'Hemp variety NBS CBD-1',
    'patent_url': 'https://patft.uspto.gov/netacgi/nph-Parser?Sect1=PTO2&Sect2=HITOFF&p=1&u=%2Fnetahtml%2FPTO%2Fsearch-bool.html&r=3&f=G&l=50&co1=AND&d=PTXT&s1=%22hemp+cultivar%22&OS=%22hemp+cultivar%22&RS=%22hemp+cultivar%22',
}))


#-----------------------------------------------------------------------
# Aggregate the data.
#-----------------------------------------------------------------------

# Read in all lab results from subjective effects paper.
data_dir = '../.datasets/subjective-effects'
lab_results = curate_lab_results(data_dir)

# Read programmatically collected plant patent data.
datafile = '../.datasets/plant-patents/plant-patents.xlsx'
details = pd.read_excel(datafile, sheet_name='Patent Details')

# Read manually collected plant patent data.
datafile = '../.datasets/plant-patents/plant-patents.xlsx'
results = pd.read_excel(datafile, sheet_name='Patent Lab Results')


#-----------------------------------------------------------------------
# Visualize the data.
#-----------------------------------------------------------------------

# Count plant patents over time.
details['date'] = pd.to_datetime(details['patent_issue_date'])
yearly = details.groupby(pd.Grouper(key='date', freq='Y'))['patent_number'].count()

# Visualize patents by year through 2021.
yearly[:-1].plot()
plt.title('Cannabis Plant Patents Issued by Year')
plt.ylabel('Count')
plt.xlabel('')
plt.show()


#-----------------------------------------------------------------------
# Perform analysis of patent data by looking at pertinent ratios.
#-----------------------------------------------------------------------

# - THCV to THC ratio.
sample = results.loc[
    (~results['thcv'].isna()) &
    (~results['delta_9_thc'].isna())
]
sample['thcv_to_thc'] = sample['thcv'] / sample['delta_9_thc']
sample['thcv_to_thc'].hist(bins=10)
plt.show()
outlier = sample.loc[sample['thcv_to_thc'] == sample['thcv_to_thc'].max()].iloc[0]
print('THCV to THC ratio outlier:', outlier['strain_name'])

#-----------------------------------------------------------------------

# Calculate THC / CBD ratio.
sample = results.loc[
    (~results['delta_9_thc'].isna()) &
    (~results['cbd'].isna()) &
    (results['cbd'] != 0)
]
sample['thc_to_cbd'] = sample['delta_9_thc'] / sample['cbd']

# Visualize THC / CBD ratio
fig, ax = plt.subplots(figsize=(18, 11.5))
sns.regplot(
    x='cbd',
    y='delta_9_thc',
    data=sample,
)
for line in range(0, sample.shape[0]):
    plt.text(sample['cbd'].iloc[line]+0.2, sample['delta_9_thc'].iloc[line], sample['strain_name'].iloc[line], horizontalalignment='left', size='medium',)

#-----------------------------------------------------------------------

# Optional: Calculate CBD / CBC / THC ratio.


#-----------------------------------------------------------------------

# Calculate `beta_pinene` to `d_limonene` ratio.
sample = results.loc[
    (~results['d_limonene'].isna()) &
    (~results['beta_pinene'].isna())
]
fig = sns.scatterplot(
    x='d_limonene',
    y='beta_pinene',
    data=results,
    hue='strain_name',
)
for line in range(0, sample.shape[0]):
    plt.text(sample['d_limonene'].iloc[line], sample['beta_pinene'].iloc[line], sample['strain_name'].iloc[line], horizontalalignment='left', size='small')
plt.legend([],[], frameon=False)
plt.show()


#-----------------------------------------------------------------------

# Calculate `beta_myrcene` to `d_limonene` ratio.
sample = results.loc[
    (~results['d_limonene'].isna()) &
    (~results['beta_myrcene'].isna())
]
fig = sns.scatterplot(
    x='d_limonene',
    y='beta_myrcene',
    data=results,
    hue='strain_name',
)
for line in range(0, sample.shape[0]):
    plt.text(sample['d_limonene'].iloc[line], sample['beta_myrcene'].iloc[line], sample['strain_name'].iloc[line], horizontalalignment='left', size='small')
plt.legend([],[], frameon=False)
plt.show()


#-----------------------------------------------------------------------

# Calculate `beta_caryophyllene` to `alpha_humulene` ratio.
sample = results.loc[
    (~results['beta_caryophyllene'].isna()) &
    (~results['alpha_humulene'].isna())
]
fig = sns.scatterplot(
    x='beta_caryophyllene',
    y='alpha_humulene',
    data=results,
    hue='strain_name',
)
for line in range(0, sample.shape[0]):
    plt.text(sample['beta_caryophyllene'].iloc[line], sample['alpha_humulene'].iloc[line], sample['strain_name'].iloc[line], horizontalalignment='left', size='small')
plt.legend([],[], frameon=False)
plt.show()

#-----------------------------------------------------------------------

# Saturday Morning Statistics Teaser:
# Ridge plot of strain fingerprints.

