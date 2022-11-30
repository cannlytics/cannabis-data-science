


# plt.rc('text', usetex=True) # Proccess figure text with LaTeX (slow, but pretty).



# Aggregate quantities by producer (and by retailer).
# producer_data = data.groupby(['month', 'producer_mme_id'], ).sum()['weight']
# cultivator_data = flower_data.groupby(['month', 'producer_mme_id'], ).sum()['weight']
# processor_data = concentate_data.groupby(['month', 'producer_mme_id'], ).sum()['weight']
# retailer_data = data.groupby(['month', 'mme_id'], ).sum()['weight']

#--------------------------------------------------------------------------
# Yield appears to be the name of the game. Does the amount a producer produces
# affect the number of periods a producer has operated or if a producer has exited.
#--------------------------------------------------------------------------

# 1. Measure yields: Aggregate quantities by producer (and by retailer).
# producer_data = data.groupby(['day', 'producer_mme_id'], ).sum()['weight']
# producer_data = producer_data.groupby(pd.Grouper(freq='M')).sum()

# retailer_data = data.groupby(['day', 'mme_id'], ).sum()['weight']
# retailer_data = retailer_data.groupby(pd.Grouper(freq='M')).sum()

# sample = data.loc[
#     (data['date'] >= pd.to_datetime('2020-02-01')) &
#     (data['date'] <= pd.to_datetime('2021-10-31')) &
#     (data['parsed_uom'] != 'ea') &
#     (data['total_cannabinoid_percent'] > 0)
# ]

# 2. Determine when producers are operating by start and end of their data.


# import pymc3 as pm
# Bayesian linear regression.
# X, y = linear_training_data()
# with pm.Model() as linear_model:
#     parameter_belief = pm.Normal('weights', mu=0, sigma=1)
#     variance_belief = pm.Gamma('noise', alpha=2, beta=1)
#     y_observed = pm.Normal(
#         'y_observed',
#         mu=X @ parameter_belief,
#         sigma=variance_belief,
#         observed=y,
#     )
#     prior = pm.sample_prior_predictive()
#     posterior = pm.sample()
#     posterior_pred = pm.sample_posterior_predictive(posterior)


#--------------------------------------------------------------------------
# Conduct survival analysis.
#--------------------------------------------------------------------------

# Calculate the percent of retailers / producers that have exited.
# The percent that have not exited are the percent of censored data.

#--------------------------------------------------------------------------
# Explore various factors
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# Get Census Data
# API Key: https://api.census.gov/data/key_signup.html
# Attribution: This product uses the Census Bureau Data API but is not
# endorsed or certified by the Census Bureau.
#--------------------------------------------------------------------------

# from dotenv import dotenv_values
# from census import Census # pip install census
# from us import states # pip install us
# import requests

# # Create a request session.
# session = requests.session()
# session.headers.update({'User-Agent': 'census-demo/0.0'})

# # Read your Census API key.
# config = dotenv_values('../.env')
# census_api_key = api_key=config['CENSUS_API_KEY']

# # Make requests to the Census API.
# client = Census(census_api_key, session=session)
# census_data = client.acs5.get(
#     ('NAME', 'B25034_010E'),
#     {'for': 'state:{}'.format(states.MD.fips)}
# )

# # Examples:
# # c.acs5.get('B01001_004E', {'for': 'state:*'})
# # c.acs5.state('B01001_004E', Census.ALL)
# # c.sf1.state_county_tract('NAME', states.AK.fips, '170', Census.ALL)


# TODO: Use a Chi square test to determine the better model.

# TODO: Visualize the regression results.


#--------------------------------------------------------------------------
# For next week: Fit a Cox's proportional hazards model
#--------------------------------------------------------------------------

# from lifelines import CoxPHFitter

# # Fit a Cox proportional hazards model.
# cph = CoxPHFitter()
# cph.fit(panel, duration_col='duration', event_col='exit')
# cph.print_summary()

# # Fit a linear model with an interaction term.
# cph.fit(
#     panel.drop(columns=['mme_id']),
#     duration_col='duration',
#     event_col='exit',
#     # formula="fin + wexp + age * prio"
# )

# import numpy as np

# # Prediction.
# X = np.array([24, 0])
# cph.predict_survival_function(X)
# cph.predict_median(X)
# cph.predict_partial_hazard(X)

# # Visualizing effects.
# cph.plot_partial_effects_on_outcome(
#     covariates='prio',
#     values=[0, 2, 4, 6, 8, 10], 
#     cmap='coolwarm'
# )