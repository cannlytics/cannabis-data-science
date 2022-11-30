"""
Analyze Cannabis Plant Patent Data
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 5/21/2022
Updated: 5/21/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description:

    Analyze plant patent data in relation to observed data
    to determine if plant patents are outliers and other
    potential outliers that would be good patent candidates.

Data Sources:

    - Data from: Over eight hundred cannabis strains characterized
    by the relationship between their subjective effects, perceptual
    profiles, and chemical compositions
    URL: <https://data.mendeley.com/datasets/6zwcgrttkp/1>
    License: CC BY 4.0. <https://creativecommons.org/licenses/by/4.0/>

Resources:

    - Over eight hundred cannabis strains characterized by the
    relationship between their psychoactive effects, perceptual
    profiles, and chemical compositions
    URL: <https://www.biorxiv.org/content/10.1101/759696v1.abstract>

    - Effects of cannabidiol in cannabis flower:
    Implications for harm reduction
    URL: <https://pubmed.ncbi.nlm.nih.gov/34467598/>

"""
# Internal imports.
import os

# External imports.
from cannlytics.utils.utils import snake_case # pip install cannlytics
import matplotlib.pyplot as plt
import pandas as pd
import pypatent # pip install pypatent
import seaborn as sns

# Decarboxylation rate. Source: <https://www.conflabs.com/why-0-877/>
DECARB = 0.877


#------------------------------------------------------------------------------
# Get cannabis plant patent data.
#------------------------------------------------------------------------------

# Get plant patent data.

# # Get most recently filed cannabis-related patents.
# query = pypatent.Search('cannabis', results_limit=10) 
# patents = query.as_dataframe()

# # FIXME: Get specific patent.
# # patent = pypatent.Search('', pn='US0PP027475')
# # this_patent = pypatent.Patent(title='Ecuadorian Sativa', url='https://pdfpiw.uspto.gov/.piw?PageNum=0&docid=PP027475&IDKey=3C857949F63E&HomeUrl=http%3A%2F%2Fpatft.uspto.gov%2Fnetacgi%2Fnph-Parser%3FSect1%3DPTO2%2526Sect2%3DHITOFF%2526u%3D%25252Fnetahtml%25252FPTO%25252Fsearch-adv.htm%2526r%3D22%2526f%3DG%2526l%3D50%2526d%3DPTXT%2526p%3D1%2526S1%3D%28%28%252522plant%252522.TI.%29%252BAND%252Bcannabis.TI.%29%2526OS%3Dttl%2F%252522plant%252522%252Band%252Bttl%2Fcannabis%2526RS%3D%28TTL%2F%252522plant%252522%252BAND%252BTTL%2Fcannabis%29')
# # patent = this_patent.fetch_details()
# base = 'http://patft.uspto.gov/netacgi/nph-Parser'
# url = f'{base}?Sect1=PTO2&Sect2=HITOFF&u=%2Fnetahtml%2FPTO%2Fsearch-adv.htm&r=4&p=1&f=G&l=50&d=PTXT&S1=aaa&OS=aaa&RS=aaa'
# patent = this_patent = pypatent.Patent(title='Patent', url=url)






#------------------------------------------------------------------------------


def curate_lab_results(
        data_dir,
        compound_folder='Terpene and Cannabinoid data',
        cannabinoid_file='rawDATACana',
        terpene_file='rawDATATerp',
        max_cannabinoids=35,
        max_terpenes=8,
):
    """Curate lab results for effects prediction model."""

    # Read terpenes.
    terpenes = None
    if terpene_file:
        file_path = os.path.join(data_dir, compound_folder, terpene_file)
        terpenes = pd.read_csv(file_path, index_col=0)
        terpenes.columns = [snake_case(x).strip('x_') for x in terpenes.columns]
        terpene_names = list(terpenes.columns[3:])
        compounds = terpenes

    # Read cannabinoids.
    cannabinoids = None
    if cannabinoid_file:
        file_path = os.path.join(data_dir, compound_folder, cannabinoid_file)
        cannabinoids = pd.read_csv(file_path, index_col=0)
        cannabinoids.columns = [snake_case(x).strip('x_') for x in cannabinoids.columns]
        cannabinoid_names = list(cannabinoids.columns[3:])
        compounds = cannabinoids

    # Merge terpenes and cannabinoids.
    if terpene_file and cannabinoid_file:
        compounds = pd.merge(
            left=cannabinoids,
            right=terpenes,
            left_on='file',
            right_on='file',
            how='left',
            suffixes=['', '_terpene']
        )

    # Rename any oddly named columns.
    rename = {
        'cb_da': 'cbda',
        'cb_ga': 'cbda',
        'delta_9_th_ca': 'delta_9_thca',
        'th_ca': 'thca',
    }
    compounds.rename(columns=rename, inplace=True)

    # Combine `delta_9_thca` and `thca`.
    # FIXME: Ensure that this is combining the two fields correctly.
    compounds['delta_9_thca'].fillna(compounds['thca'], inplace=True)
    compounds.drop(columns=['thca'], inplace=True)
    cannabinoid_names.remove('thca')

    # FIXME: Combine any additional compounds.
    # compounds['delta_9_thca'].fillna(compounds['thca'], inplace=True)
    # compounds.drop(columns=['thca'], inplace=True)
    # cannabinoid_names.remove('thca')

    # FIXME: Calculate totals.
    compounds['total_terpenes'] = compounds[terpene_names].sum(axis=1).round(2)
    compounds['total_cannabinoids'] = compounds[cannabinoid_names].sum(axis=1).round(2)
    compounds['total_thc'] = (compounds['delta_9_thc'] + compounds['delta_9_thca'].mul(DECARB)).round(2)
    compounds['total_cbd'] = (compounds['cbd'] + compounds['cbda'].mul(DECARB)).round(2)

    # Exclude outliers.
    compounds = compounds.loc[
        (compounds['total_cannabinoids'] < max_cannabinoids) &
        (compounds['total_terpenes'] < max_terpenes)
    ]

    # Return compounds with nulls as 0.
    compounds = compounds.fillna(0)
    return compounds


# Read in all lab results from subjective effects paper.
data_dir = '../.datasets/subjective-effects'
lab_results = curate_lab_results(data_dir)

# Group lab results by strain.
strains = lab_results.groupby('tag').mean()
strains['tests'] = lab_results.groupby('tag')['cbd'].count()


#------------------------------------------------------------------------------

# Look at the distribution of terpenes.
terpenes = [x for x in list(lab_results.columns) if not x.startswith('total')]
exclude_vars = ['tag', 'tag_terpene', 'file', 'type', 'type_terpene']
for exclude in exclude_vars:
    terpenes.remove(exclude)

# Find samples in the top percentile for each terpene.
quantile = .99
number = 2
for terpene in terpenes:
    high_conc = strains.loc[strains[terpene] >= strains[terpene].quantile(quantile)]
    high_conc = high_conc.head(number)
    print(f'1st percentile of {terpene}:') # , ', '.join(list(high_conc.index))
    print(high_conc[terpene])
    lab_results[terpene].hist(bins=100)
    plt.vlines(
        high_conc[terpene],
        ymin=0,
        ymax=100,
        color='red'
    )
    plt.show()


# Given a sample, return the percentile of it's terpenes.


# Rank the strains by the percentile of their terpene.


#------------------------------------------------------------------------------
# Look at where patents in comparison to dataset average.
#------------------------------------------------------------------------------


# # Read manually collected plant patent data.
# datafile = '../.datasets/plant-patents/plant-patents.xlsx'
# patent_data = pd.read_excel(datafile)

# # Look at the first plant patent.
# first_patent = patent_data.loc[
#     patent_data['patent_number'] == 'US0PP027475'
# ]
# sns.regplot(
#     x='d_limonene',
#     y='beta_myrcene',
#     data=lab_results,
# )
# plt.scatter(
#     first_patent['beta_myrcene'],
#     first_patent['d_limonene'],
#     color='red',
#     label='First Cannabis Plant Patent'
# )
# plt.legend(loc='best')
# plt.show()


#------------------------------------------------------------------------------

# Look at terpene ratios.


# Identify the ratios with the highest correlation.


# Look at the distribution of terpene ratios.


# Find samples in the top percentile for each ratio.


# Rank the strains by the percentile of their terpene.


#------------------------------------------------------------------------------

# TODO: Statistically identify outliers? --> These would be good strains for patents!


#------------------------------------------------------------------------------


# TODO: Report accuracy and precision.



#------------------------------------------------------------------------------


# TODO: Get chemical data from ChemSpider, etc.
