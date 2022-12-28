"""
Cultivar Prediction
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 6/23/2022
Updated: 6/25/2022
License: MIT License <https://opensource.org/licenses/MIT>
"""

# Conduct a MANOVA test for each strain (or average?).
import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA

# Read in all the patent data.
patents = pd.read_excel(
    '../.datasets/plant-patents/plant-patents.xlsx',
    sheet_name='Patent Lab Results',
)

# Test MANOVA by creating two fictitious strains.
compounds = [
    'delta_9_thc',
    'thca',
    'thcva',
    'thcv',
    'cbd',
    'cbda',
    'cbga',
    'cbg',
    'cbca',
    'cbc',
    'cbn',
    'alpha_bisabolol',
    'alpha_caryophyllene',
    'alpha_humulene',
    'alpha_pinene',
    'alpha_terpinene',
    'alpha_terpineol',
    'beta_caryophyllene',
    'beta_myrcene',
    'beta_pinene',
    'camphene',
    'caryophyllene_oxide',
    'd_limonene',
    'gamma_terpinene',
    'linalool',
    'ocimene',
    'terpinolene',
    'borneol',
    'camphor',
    'citronellol',
    'citral',
]

# Average by strain.
avgs = patents.groupby('strain_name').mean()
stds = patents.groupby('strain_name').std()

# Create mock strain profiles.
strain_1 = {}
strain_2 = {}
for compound in compounds:
    try:
        avg = avgs.loc[avgs[compound] > 0].sample(2, random_state=420).round(4)
        std = stds.loc[stds[compound] > 0].sample(2, random_state=420).round(4)
        strain_1[compound] = {
            'mean': avg.iloc[0][compound],
            'std': std.iloc[0][compound],
        }
        strain_2[compound] = {
            'mean': avg.iloc[1][compound],
            'std': std.iloc[1][compound],
        }
    except ValueError:
        pass

# Create mock lab result data with distributions
# for each terpene.
N = 300
np.random.seed(420)
strain_1_data = pd.DataFrame()
strain_2_data = pd.DataFrame()
for key, values in strain_1.items():
    obs = pd.Series(np.random.normal(values['mean'], values['std'], size=N))
    obs.loc[obs < 0] = 0
    strain_1_data[key] = obs
for key, values in strain_2.items():
    obs = pd.Series(np.random.normal(values['mean'], values['std'], size=N))
    obs.loc[obs < 0] = 0
    strain_2_data[key] = obs

# See how many samples are needed to begin to distinguish
# between the two different strains.
# Future work:See if the given lab result data are statistically
# different from all other patents.
manova = MANOVA(endog=strain_1_data, exog=strain_2_data)
print(manova.mv_test())
