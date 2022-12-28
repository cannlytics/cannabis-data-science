"""
Training Statistical Models
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 6/4/2022
Updated: 6/4/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description:

    Explore correlations between cannabinoid / terpene concentrations
    and reported effects and flavors.

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
import numpy as np
import pandas as pd
import pypatent # pip install pypatent
import seaborn as sns
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm

# Decarboxylation rate. Source: <https://www.conflabs.com/why-0-877/>
DECARB = 0.877


# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})


#------------------------------------------------------------------------------
# Read in the data.
#------------------------------------------------------------------------------

# Read the panel data.
panel = pd.read_excel('../.datasets/subjective-effects/strain-reviews-2022-06-01.xlsx')

# Get the variables.
effects = [x for x in list(panel.columns) if x.startswith('effect')]
aromas = [x for x in list(panel.columns) if x.startswith('aroma')]


#------------------------------------------------------------------------------
# Create the variable of interests.
#------------------------------------------------------------------------------

# Calculate beta-pinene to d-limonene ratio.
panel['log_pinene_to_limonene'] = np.log(panel['beta_pinene'].div(panel['d_limonene']))

# Calculate caryophyllene to humulene ratio.
panel['log_caryophyllene_to_humulene'] = np.log(panel['beta_caryophyllene'].div(panel['humulene']))

# Calculate total THC to total CBD ratio.
decarb = 0.877
panel.loc[panel['thca'] == 0, 'total_thc'] = panel['delta_9_thc']
panel.loc[panel['thca'] != 0, 'total_thc'] = panel['delta_9_thc'] + panel['thca'].mul(decarb)
panel.loc[panel['cbda'] == 0, 'total_cbd'] = panel['cbd']
panel.loc[panel['cbda'] != 0, 'total_cbd'] = panel['cbd'] + panel['cbda'].mul(decarb)
panel['log_thc_to_cbd'] = np.log(panel['total_thc'].div(panel['total_cbd']))

# Look at only observations with THC.
data = panel.loc[
    (np.isfinite(panel['log_thc_to_cbd'])) &
    (np.isfinite(panel['log_pinene_to_limonene'])) &
    (np.isfinite(panel['log_caryophyllene_to_humulene']))
]


#------------------------------------------------------------------------------
# Separate training data.
#------------------------------------------------------------------------------

# Create training and test data.
sample_size = 0.9
N = len(data)
n = round(sample_size * N)
sample = data.sample(n, random_state=420)
train = data.loc[~data.index.isin(sample.index)]


#------------------------------------------------------------------------------
# Estimate the model.
#------------------------------------------------------------------------------

# Define the variates.
Y = sample['effect_sleepy']
X = sample[[
    'log_thc_to_cbd',
    'log_pinene_to_limonene',
    'log_caryophyllene_to_humulene',
]]
X = sm.add_constant(X)

# Estimate the model.
model = sm.Probit(Y, X).fit(disp=0)
print(model.summary())

# Calculate the prediction threshold.
y_bar = Y.mean()
Y_hat = pd.Series(model.predict(X))
threshold = round(Y_hat.quantile(1 - y_bar), 4)

# Visualize the predictions.
Y_hat.hist(bins=100)
plt.show()

# Get the marginal effects.
print(model.get_margeff().summary())


# TODO: Exclude Type 2 and Type 3 cannabis and see if it improves accuracy.


#------------------------------------------------------------------------------
# Calculate accuracy statistics.
#------------------------------------------------------------------------------

# Create test predictions.
y = train['effect_sleepy']
x_hat = train[[
    'log_thc_to_cbd',
    'log_pinene_to_limonene',
    'log_caryophyllene_to_humulene',
]]
x_hat = sm.add_constant(x_hat)
y_hat = model.predict(x_hat)
predictions = pd.Series(y_hat > threshold).astype(int)

# Calculate the confusion matrix
cm = confusion_matrix(y, predictions)
tn, fp, fn, tp = cm.ravel()
pos = sum(y)
neg = len(y) - pos

# Calculate the false positive rate.
fpr = round(fp / neg, 4)

# Calculate the false negative rate.
fnr = round(fn / pos, 4)

# Calculate the true positive rate.
tpr = round(tp / pos, 4)

# Calculate the true negative rate.
tnr = round(tn / neg, 4)

# Calculate the accuracy rate.
accuracy = round((tp + tn) / (pos + neg), 4)

# Calculate the informedness.
informedness = round((tp / pos) / (tn / neg), 4)

# Calculate the predictive value.
pp = sum(y_hat)
pn = len(y_hat) - pp
ppv = round((tp / pp), 4)

# Calculate the false omission rate.
_for = round((fn / pn), 4)
