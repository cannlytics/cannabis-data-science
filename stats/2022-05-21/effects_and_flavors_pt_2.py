"""
Purported Effects and Flavors Probit Model
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 5/13/2022
Updated: 5/14/2022
License: MIT License <https://opensource.org/licenses/MIT>

Objectives:

    - Aid in the identification and characterization of cannabis strains.

    - Explore correlations between cannabinoid / terpene concnetrations
    and reported effects and flavors.

Data Sources:

    - Data from: Over eight hundred cannabis strains characterized
    by the relationship between their subjective effects, perceptual
    profiles, and chemical compositions
    https://data.mendeley.com/datasets/6zwcgrttkp/1 

Resources:

    - Over eight hundred cannabis strains characterized by the
    relationship between their psychoactive effects, perceptual
    profiles, and chemical compositions
    https://www.biorxiv.org/content/10.1101/759696v1.abstract

Setup:

    1. pip install cannlytics

"""
# Standard imports.
import os

# External imports.
from cannlytics.utils.utils import snake_case
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm


# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})
colors = sns.color_palette('Set2', n_colors=10)


#-----------------------------------------------------------------------
# Get the data.
#-----------------------------------------------------------------------

# Define directories.
DATA_DIR = '../.datasets/subjective-effects'
compound_folder = 'Terpene and Cannabinoid data'

#-----------------------------------------------------------------------

# Read terpenes.
file_path = os.path.join(DATA_DIR, compound_folder, 'rawDATATerp')
terpenes = pd.read_csv(file_path, index_col=0)
terpenes.columns = [snake_case(x) for x in terpenes.columns]

# Calculate total terpenes.
terpene_names = list(terpenes.columns[3:])
terpenes['total_terpenes'] = terpenes[terpene_names].sum(axis=1)

#-----------------------------------------------------------------------

# Read cannabinoids.
file_path = os.path.join(DATA_DIR, compound_folder, 'rawDATACana')
cannabinoids = pd.read_csv(file_path, index_col=0)
cannabinoids.columns = [snake_case(x) for x in cannabinoids.columns]

# Calculate total cannabinoids.
cannabinoid_names = list(cannabinoids.columns[3:])
cannabinoids['total_cannabinoids'] = cannabinoids[cannabinoid_names].sum(axis=1)

#-----------------------------------------------------------------------

# Merge terpenes and cannabinoids.
compounds = pd.merge(
    left=cannabinoids,
    right=terpenes,
    left_on='file',
    right_on='file',
    how='left',
    suffixes=['_cannabinoid', '_terpene']
)
print('Number of lab results:', len(compounds))

# Average results by strain.
concentrations = compounds.groupby('tag_cannabinoid').mean()
concentrations = concentrations.fillna(0)


#-----------------------------------------------------------------------
# Look at introductory data.
#-----------------------------------------------------------------------

# Plot total terpenes.
compounds['total_terpenes'].hist(bins=100)
plt.show()
print('Mean:', compounds['total_terpenes'].mean())
print('Median:', compounds['total_terpenes'].median())

# Plot total cannabinoids.
compounds['total_cannabinoids'].hist(bins=100)
plt.show()
print('Mean:', compounds['total_cannabinoids'].mean())
print('Median:', compounds['total_cannabinoids'].median())

# Visualize all compounds.
for compound in list(concentrations.columns):
    concentrations[compound].hist(bins=25)
    plt.title(compound)
    plt.show()
    print('Mean:', concentrations[compound].mean())
    print('ND:', len(concentrations.loc[
    concentrations[compound] == 0]
    ))

# TODO: Exclude outliers.


#-----------------------------------------------------------------------
# Curate the data.
#-----------------------------------------------------------------------

# Define an output file.
output_file = os.path.join(DATA_DIR, 'review-panel.xlsx')

# Create a panel of reviews of strains.
# Note: You can comment the code until the next section
# if you have already created the panel.
panel = pd.DataFrame()

# Iterate through all strain data.
stain_folder = 'Strain data/strains'
print('Creating panel...')
for index, row in concentrations.iterrows():

    # Read the strain's effects and flavors data.
    review_file = row.name + '.p'
    file_path = os.path.join(DATA_DIR, stain_folder, review_file)
    try:
        strain = pd.read_pickle(file_path)
    except FileNotFoundError:
        print("Couldn't find:", row.name)
        continue

    # Assign dummy variables for effects and flavors.
    reviews = strain['data_strain']
    for review in reviews:

        # Create panel observation, combining prior compound data.
        obs = row.copy()
        for flavor in review['sabores']:
            key = 'flavor_' + snake_case(flavor)
            obs[key] = 1
        for effect in review['efectos']:
            key = 'effect_' + snake_case(effect)
            obs[key] = 1
        
        # Record the observation.
        panel = panel.append(obs)
    
    # Original author: Estimate the probability of 
    # a review containing flavor or effect.

    # Future work: NLP of 'reporte'.
    # This is how original author classified hybrid, indica, and sativa.
    
    # Optional: Assign dummy variables for categories.
    # categories = list(strain['categorias'])

    # Optional: Assign dummy variables for mode flavors.

# Fill null effects and flavors.
panel = panel.fillna(0)

# Save the panel data.
panel.to_excel(output_file)


#-----------------------------------------------------------------------
# Look at the panel data.
#-----------------------------------------------------------------------

# Optional: Read back in the panel (useful if panel is already baked).
panel = pd.read_excel(output_file, index_col=0)

# Find all of the reported effects.
reported_effects = [x for x in panel.columns if x.startswith('effect')]

# Find all of the reported flavors.
reported_flavors = [x for x in panel.columns if x.startswith('flavor')]

# TODO: Add color!!!

# Estimate the unconditional probability of reporting an effect.
effect_probs = {}
for reported_effect in reported_effects:
    prob = panel[reported_effect].mean()
    name = reported_effect.replace('effect_', '').replace('_', ' ').title()
    effect_probs[name] = prob
effect_probs = pd.Series(effect_probs)
effect_probs.sort_values(inplace=True, ascending=False)
fig, ax = plt.subplots(figsize=(18.5, 10.5))
(effect_probs * 100).plot(kind='bar')
plt.xlabel('Effect')
plt.ylabel('Percent')
plt.title('Percent of cannabis strain reviews that report given effect')
plt.show()
print('N:', len(panel))

# Estimate the unconditional probability of reporting an flavor.
flavor_probs = {}
for reported_flavor in reported_flavors:
    prob = panel[reported_flavor].mean()
    name = reported_flavor.replace('flavor_', '').replace('_', ' ').title()
    flavor_probs[name] = prob
flavor_probs = pd.Series(flavor_probs)
flavor_probs.sort_values(inplace=True, ascending=False)
fig, ax = plt.subplots(figsize=(18.5, 10.5))
(flavor_probs * 100).plot(kind='bar')
plt.xlabel('Flavor')
plt.ylabel('Percent')
plt.title('Percent of cannabis strain reviews that report given flavor')
plt.show()
print('N:', len(panel))


#-----------------------------------------------------------------------
# Optional: Perform literature review methodology.
#-----------------------------------------------------------------------

# Original author: Implement random forest algorithm with scikit-learn.


#-----------------------------------------------------------------------
# Perform methodology: Estimate the probability of a consumer reporting
# a given flavor or effect given cannabinoid and terpene data!
#-----------------------------------------------------------------------

# Define the flavor / effect of interest.
independent_variable = 'flavor_skunk' # Try `effect_happy`, etc.

# Define explanatory variables.
explanatory_variables = cannabinoid_names + terpene_names
panel['const'] = 1

# Estimate a probit model.
Y = panel[independent_variable]
X = panel[explanatory_variables]
X = X.loc[:, (X != 0).any(axis=0)]
X = sm.add_constant(X)
variates = X.columns
model = sm.Probit(Y, X).fit()
print(model.summary())

#-----------------------------------------------------------------------

# See how well the model predicts the sample.
y_hat = model.predict(X)
y_bar = panel[independent_variable].mean()
threshold = y_hat.quantile(1 - y_bar)
predictions = pd.Series(y_hat > threshold).astype(int)
print('Observed proportion:', y_bar)
print('Estimated proportion:', predictions.mean())

# Calculate the confusion matrix.
train = panel[independent_variable]
cm = confusion_matrix(train, predictions)
tn, fp, fn, tp = cm.ravel()
print('False positive count:', fp)
print('False negative count:', fn)
print('True positive count:', tp)
print('True negative count:', tn)

# Strains predicted correctly with the flavor / effect:
actual = list(panel.loc[panel[independent_variable] == 1].index.unique())
predicted = list(predictions.loc[predictions == 1].index.unique())
correct = list(set(predicted) & set(actual))
accuracy = round(len(correct) / len(actual) * 100, 2)
print('Accuracy: %.2f%%' % accuracy)

#-----------------------------------------------------------------------

# Predict a single sample (below are mean concentrations).
x = pd.Series({
    'const': 1,
    'delta_9_thc': 10.85,
    'cbd': 0.29,
    'cbn': 0.06,
    'cbg': 0.54,
    'cbc': 0.15,
    'thcv': 0.07,
    'cb_da': 0.40,
    'delta_8_thc': 0.00,
    'cb_ga': 0.40,
    'th_ca': 8.64,
    'delta_9_th_ca': 1.12,
    'limonene': 0.22,
    'beta_ocimene': 0.05,
    'beta_myrcene': 0.35,
    'beta_pinene': 0.12,
    'linalool': 0.07,
    'alpha_pinene': 0.10,
    'caryophyllene': 0.18,
    'camphene': 0.01,
    '3_carene': 0.00,
    'alpha_terpinene': 0.00,
    'ocimene': 0.00,
    'cymene': 0.00,
    'eucalyptol': 0.00,
    'gamma_terpinene': 0.00,
    'terpinolene': 0.08,
    'isopulegol': 0.00,
    'geraniol': 0.00,
    'humulene': 0.06,
    'trans_nerolidol_1': 0.00,
    'trans_nerolidol_2': 0.01,
    'guaiol': 0.01,
    'caryophyllene_oxide': 0.00,
    'alpha_bisabolol': 0.03,
    'beta_caryophyllene': 0.11,
    'alpha_humulene': 0.03,
    'p_cymene': 0.00,
    'trans_nerolidol': 0.00,
    'terpinene': 0.00,
})
prediction = model.predict(x)
if prediction.values[0] > threshold:
    print('Good likelihood of ', independent_variable)
else:
    print('Low likelihood of being', independent_variable)

# Crank the jack, i.e. `terpinolene`!
x['terpinolene'] = concentrations['terpinolene'].quantile(.9)
prediction = model.predict(x)
if prediction.values[0] > threshold:
    print('Good likelihood of ', independent_variable)
else:
    print('Low likelihood of being', independent_variable)


#-----------------------------------------------------------------------
# Present the results.
#-----------------------------------------------------------------------

# Plot logistic regression curve (a probit curve would be better!).
sns.regplot(
    x='terpinolene',
    y='flavor_skunk',
    data=panel,
    logistic=True,
    ci=None
)
plt.show()


#-----------------------------------------------------------------------
# Future work: Build a re-usable model that predicts ALL
# flavors and effects given lab results.
#-----------------------------------------------------------------------
