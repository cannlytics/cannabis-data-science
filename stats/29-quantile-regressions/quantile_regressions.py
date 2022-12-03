"""
Quantile Regression Models
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 6/10/2022
Updated: 6/11/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description:

    Explore relationships between terpenes with quantile regressions.

Data Sources:

    - Data from: Over eight hundred cannabis strains characterized
    by the relationship between their subjective effects, perceptual
    profiles, and chemical compositions
    URL: <https://data.mendeley.com/datasets/6zwcgrttkp/1>
    License: CC BY 4.0. <https://creativecommons.org/licenses/by/4.0/>

    - Curated Cannabis Strains, their Average Chemical Compositions, and
    Reported Effects and Aromas
    URL: https://cannlytics.page.link/reported-effects
    License: CC BY 4.0. <https://creativecommons.org/licenses/by/4.0/>

Resources:

    - Over eight hundred cannabis strains characterized by the
    relationship between their psychoactive effects, perceptual
    profiles, and chemical compositions
    URL: <https://www.biorxiv.org/content/10.1101/759696v1.abstract>

    - Summary Statistics for Skewed Distributions
    URL: <https://web.ma.utexas.edu/users/mks/statmistakes/skeweddistributions.html>

    - What is Quantile Regression?
    URL: <http://www.econ.uiuc.edu/~roger/research/rq/rq.html>

"""
# External imports.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

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
datafile = '../.datasets/reported-effects-and-aromas.xlsx'
results = pd.read_excel(datafile, sheet_name='Lab Results')


#------------------------------------------------------------------------------
# Clean the data.
#------------------------------------------------------------------------------

# Add a constant.
data = sm.add_constant(results)

# Create log-normal variables.
data['log_beta_pinene'] = np.log(data['beta_pinene'])
data['log_d_limonene'] = np.log(data['d_limonene'])
data['log_beta_caryophyllene'] = np.log(data['beta_caryophyllene'])
data['log_humulene'] = np.log(data['humulene'])

# Define variates.
y_key = 'log_beta_pinene'
x_key = 'log_d_limonene'

# keeping only samples with finite values.
data = data.loc[(np.isfinite(data[y_key])) & (np.isfinite(data[x_key]))]

# Add ad-hoc "Sativa" / "Indica" type.
data['type'] = 'Sativa'
data.loc[(data['beta_pinene'].div(data['d_limonene'])) < 0.25, 'type'] = 'Indica'

#------------------------------------------------------------------------------
# Estimate the models.
#------------------------------------------------------------------------------

# Look at a OLS regression.
Y = data[y_key]
X = data[['const', x_key]]
ols = sm.OLS(Y, X).fit()
print(ols.summary())

# Keep the OLS results for visualization.
ols_ci = ols.conf_int().loc[x_key].tolist()
ols_model = dict(
    const=ols.params['const'],
    beta=ols.params[x_key],
    lb=ols_ci[0],
    ub=ols_ci[1],
)

# Estimate a quantile regression (Least Absolute Deviation model when q=0.5).
lad_model = QuantReg(Y, X).fit(q=.5)
print(lad_model.summary())


def fit_model(Y, X, q):
    res = QuantReg(Y, X).fit(q=q)
    stats = [q, res.params['const'], res.params[x_key]]
    stats += res.conf_int().loc[x_key].tolist()
    return stats


# Estimate a myriad of quantile regressions.
quantiles = np.arange(0.05, 0.96, 0.1)
models = [fit_model(Y, X, q) for q in quantiles]
models = pd.DataFrame(
    models,
    columns=['q', 'const', 'beta', 'lb', 'ub']
)


#------------------------------------------------------------------------------
# Visualize the models.
#------------------------------------------------------------------------------

# Visualize the regressions with the data.
fig, ax = plt.subplots(figsize=(11.5, 8))

# Plot the quantile regressions.
for _, model in models.iterrows():
    x_hat = X[x_key]
    y_hat = model['const'] + model['beta'] * x_hat
    ax.plot(x_hat, y_hat, linestyle='dotted', color='grey', alpha=0.6)

# Plot the OLS regression.
y_hat = sm.OLS(Y, X).fit().predict(X)
ax.plot(x_hat, y_hat, color='red', label='OLS')

# Plot the data.
sns.scatterplot(
    x=x_key,
    y=y_key,
    data=data,
    hue='type'
)
legend = ax.legend()
ax.set_xlabel(x_key, fontsize=16)
ax.set_ylabel(y_key, fontsize=16)
plt.show()

#------------------------------------------------------------------------------

# Visualize the estimated parameters.
n = models.shape[0]
plt.plot(models['q'], models['beta'], color='black', label='Quantile Reg.')
plt.plot(models['q'], models['ub'], linestyle='dotted', color='black')
plt.plot(models['q'], models['lb'], linestyle='dotted', color='black')
plt.plot(models['q'], [ols_model['beta']] * n, color='red', label='OLS')
plt.plot(models['q'], [ols_model['lb']] * n, linestyle='dotted', color='red')
plt.plot(models['q'], [ols_model['ub']] * n, linestyle='dotted', color='red')
plt.ylabel(r'$\beta$')
plt.xlabel('Quantiles of the conditional distribution')
plt.legend()
plt.show()
