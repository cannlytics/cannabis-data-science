"""
Draft Material | Predicting Lab Profitability in Washington State
Copyright (c) 2022 Cannlytics

Author: Keegan Skeate <keegan@cannlytics.com>
Created: 1/10/2022
Updated: 1/12/2022
License: MIT License <https://opensource.org/licenses/MIT>
"""

# # Combine the data sets.
# # data = pd.merge(
# #     left=lab_data,
# #     right=licensee_data,
# #     how='left',
# #     left_on='for_mme_id',
# #     right_on='global_id'
# # )

# # Identify all of the labs.
# labs = licensee_data.loc[licensee_data['code'].str.startswith('L')]

# # Sort by lab number.
# lab_number = labs['code'].str.replace('L', '') \
#                          .apply(pd.to_numeric, errors='coerce').dropna()
# labs = labs.assign(lab_number=lab_number)
# labs = labs.sort_values(ascending=True, by='lab_number')

# Run a regrssion of revenue on price, etc.
# revenue_data = five_year_forecast_data.loc[
#     five_year_forecast_data['revenue'] > 0
# ]
# X = revenue_data['price']
# X = sm.add_constant(X)
# regression = sm.OLS(revenue_data['revenue'], X).fit()
# print(regression.summary())

