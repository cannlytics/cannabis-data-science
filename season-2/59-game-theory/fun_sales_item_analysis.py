"""
Quick Sales Item Analysis
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 3/29/2022
Updated: 3/30/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description: Perform a quick sales items analysis
using a sample of observed sales items.
"""
import pandas as pd
import statsmodels.api as sm


#--------------------------------------------------------------------------
# 1. Wrangle the data.
#--------------------------------------------------------------------------

# Read observed sales.
DATA_DIR = '../.datasets'
DATA_FILE = f'{DATA_DIR}/observed_sales.xlsx'
data = pd.read_excel(DATA_FILE, index_col=0)


#--------------------------------------------------------------------------
# 2. Curate the data.
#--------------------------------------------------------------------------

# Calculate price per gram.
data['price_per_gram'] = data['price'].div(data['weight'])

# Calculate price per mg of TAC.
tac = data['weight'] * data['tac'] * 0.01 * 1000
data['price_per_mg_tac'] = data['price'].div(tac)


#--------------------------------------------------------------------------
# 3. Analyze the data.
#--------------------------------------------------------------------------

# Regress price per gram on TAC, type.
Y = data['price_per_gram']
X = data[['tac']]
dummies = pd.get_dummies(data['type'])
dummies.drop('Sativa', axis=1, inplace=True)
X = pd.concat([X, dummies], axis=1)
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
print(model.summary())

# Predict price given TAC and strain type.
x_hat = pd.DataFrame([{
    'const': 1,
    'tac': 20,
    'Hybrid': 0,
    'Indica': 0,
}])
y_hat = model.predict(x_hat)
print('Predicted price:', y_hat.iloc[0])

# Regress price per mg of TC on type.
Y = data['price_per_mg_tac']
X = dummies
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
print(model.summary())


#--------------------------------------------------------------------------
# ***Saturday Morning Statistics Teaser***
#--------------------------------------------------------------------------

# Probit regression of type on TAC.
# Y = pd.get_dummies(data['type'])['Indica']
# X = data[['tac']]
# X = sm.add_constant(X)
# model = sm.Probit(Y, X).fit()
# print(model.summary())
# print(model.predict())

# # Predict type given TAC.
# x_hat = pd.DataFrame([{
#     'const': 1,
#     'tac': 15,
#     'price_per_gram': 11,
# }])
# y_hat = model.predict(x_hat)
# print('Predicted Indica?', 'yes' if y_hat.iloc[0] else 'no')
