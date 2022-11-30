"""
Estimate a Bayesian Linear Regression with Gibbs Sampling
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 2/25/2022
Updated: 2/26/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script analyzes the relationship between price and
sales tax by county in Washington State using a Bayesian linear regression.

Data sources:

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd

    - WA State random sample of sales
    https://cannlytics.page.link/cds53

    - WA State augmented licensees
    https://github.com/cannlytics/cannabis-data-science/blob/main/2022-01-26/geocode_licensees.py

"""
# External imports.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as S
import statsmodels.api as sm
import seaborn as sns # pip install seaborn==0.11.0

# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
})


#--------------------------------------------------------------------------
# Read the sample sales data.
#--------------------------------------------------------------------------

# Specify where your data lives.
DATA_DIR = 'D:\\leaf-data'

# Read in the data.
DATA_FILE = f'{DATA_DIR}/samples/random-sales-items-2022-02-16.csv'
data = pd.read_csv(DATA_FILE)


#------------------------------------------------------------------------------
# Get the retail city and county.
# Licensees data: https://cannlytics.com/data/market/augmented-washington-state-licensees
# Augment yourself: https://github.com/cannlytics/cannabis-data-science/blob/main/2022-01-26/geocode_licensees.py
#------------------------------------------------------------------------------

# Augment fields from the licensees data.
licensee_fields = {
    'global_id': 'string',
    'city': 'string',
    'county': 'string',
}
licensees = pd.read_csv(
    f'{DATA_DIR}/augmented/augmented-washington-state-licensees.csv',
    usecols=list(licensee_fields.keys()),
    dtype=licensee_fields,
)
data = pd.merge(
    left=data,
    right=licensees,
    how='left',
    left_on='mme_id',
    right_on='global_id',
)
data.drop(['global_id'], axis=1, inplace=True, errors='ignore')
print('Augmented the sales data with city and county.')


#--------------------------------------------------------------------------
# Clean the data.
#--------------------------------------------------------------------------

# Determine wholesale vs retail transactions.
# Note: It is necessary to find out if there are any medical sales and
# exclude any medical sales from taxes.
data = data.loc[data['sale_type'] != 'wholesale']

# Drop observations with negative prices and prices in the upper quantile.
data = data.loc[data.price_total > 0]
data = data[data.price_total < data.price_total.quantile(.95)]

# Add a date, quarter column and clean the city and county columns.
data = data.assign(date=pd.to_datetime(data['created_at']))
data = data.assign(
    city=data['city'].str.title(),
    county=data['county'].str.title().str.replace(' County', ''),
    day=data['date'].dt.date,
    quarter=data['date'].dt.to_period('Q'),
)

# TODO: Calculate price per total cannabinoids ($/mg)

# Identify the time period for analysis.
timeperiod_start = '2021-01-01'
timeperiod_end = '2021-10-31'
data = data.loc[
    (data['date'] >= pd.to_datetime(timeperiod_start)) &
    (data['date'] <= pd.to_datetime(timeperiod_end))
]
print('Data cleaned and limited to the period of analysis.')
print('{:,} observations.'.format(len(data)))


#--------------------------------------------------------------------------
# Estimate a linear regression of price on total cannabinoid percent.
#--------------------------------------------------------------------------

# Identify a sample of sales.
sample_type = 'concentrate_for_inhalation'
sample_data = data.loc[data.intermediate_type == sample_type]

# Add trend variable before creating the panel.
sample_data = sample_data.assign(
    ln_price=np.log(sample_data['price_total'])
)
sample_data = sample_data.loc[
    ~sample_data['total_cannabinoid_percent'].isnull()
]

# Define variables for OLS.
y = sample_data['ln_price']
x = sample_data[['total_cannabinoid_percent']]

# Estimate OLS.
x = sm.add_constant(x)
ols_results = sm.OLS(y, x).fit()
print(ols_results.summary())

# Find the covariance matrix.
cov = ols_results.cov_params()
print('Covariance matrix:')
print(cov)


#--------------------------------------------------------------------------
# Estimate a Bayesian linear regression with Gibbs Sampling.
#--------------------------------------------------------------------------

# 1. Define the number of draws and the prior hyperparameters.
R = 50_000
R_0 = 5000
draws = R + R_0
K = np.shape(x)[1]
mu = 0.0 * np.ones(K) # K x 1 matrix.
V_inv = 100 * np.identity(K) # K x K matrix.
a = 8.0
b = 70.0
n = len(y) # N X 1 matrix.
beta = np.zeros((draws, K)) # R x K matrix.
sigma_sq = np.ones(draws) # R x 1 matrix.
X = x.to_numpy() # N x K matrix.
Y = y.to_numpy() # K x 1 matrix.

# Gibbs Sampler
print('Gibbs sampling...')
for i in np.arange(0, draws, 1):

    # 2. Draw beta, β*, from the posterior using the updating rules for the
    # multivariate-normal distribution.
    updated_V = np.linalg.inv(np.dot(X.T, X) / sigma_sq[i - 1] + V_inv)
    updated_mu = np.dot(updated_V, (np.dot(X.T, Y) / sigma_sq[i - 1] + np.dot(V_inv, mu)))
    beta[i] = np.random.multivariate_normal(updated_mu, updated_V)

    # 3. Draw variance, σ*2, from the posterior given  β* and the updating
    # rules for the gamma distribution,
    B = beta[i, :] # k x 1 matrix
    updated_a = a + (n / 2)
    updated_b = 1 / ((1 / b) + .5 * (np.dot((Y - np.dot(X, B)).T, Y - np.dot(X, B))))
    sigma_sq[i] = S.invgamma.rvs(updated_a, updated_b)

    # 4. Repeat for R + R_0 draws.

# 5.Drop the burn-in draws, keeping the desired number of draws of β* and σ*2.
beta_draws = beta[R_0:]
sigma_draws = sigma_sq[R_0:]

# Collect the results into a DataFrame.
estimates = pd.DataFrame(beta_draws, columns=x.columns)
estimates['variance'] = pd.Series(sigma_draws)

# Print Bayes estimates under quadratic loss.
for column in estimates:
    bayes_estimate = estimates[column].mean()
    print('Bayes estimate for %s: %3.4f' % (column, bayes_estimate))


#--------------------------------------------------------------------------
# Visualize the random draw of parameters from the posterior with Gibbs sampling.
#--------------------------------------------------------------------------

# # TODO: Create a better visualization.
# # Create multiple subplots for each parameter.
# # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(19.8, 12))
# colors = sns.color_palette('Set1', n_colors=10)

# for i, column in enumerate(estimates):
#     print(column)

#     bayes_estimate = estimates[column].mean()
#     # estimates[column].hist(bins=100, density=1)

#     fig, ax = plt.subplots(figsize=(19.8, 12))
#     sns.displot(
#         data=estimates,
#         x=column,
#         kde=True,
#         ax=ax
#     )

#     # TODO: Title, xlabel, annotate bayes estimate.

#     plt.show()


