# Examples:

# poisson_mod = sm.Poisson(rand_data.endog, rand_exog)
# poisson_res = poisson_mod.fit(method="newton")
# print(poisson_res.summary())

# panel_data = panel.count()
# panel['retailer_county'] = panel_data.index.get_level_values('retailer_county')
# m1 = poisson('price_total ~ C(retailer_county)', data=panel_data).fit()
# print(m1.summary())

# preds_1 = m1.predict()
# model_fit1['preds'] = preds_1

# Adding dummies manually.
# col_dummies = pd.get_dummies(crab['C']).rename(columns = lambda x: 'Col' + str(x))
# dataWithDummies = pd.concat([crab, col_dummies], axis = 1) 
# dataWithDummies.drop(['C', 'Col1'], inplace = True, axis = 1)
# dataWithDummies = dataWithDummies.applymap(np.int)

# m2 = sm.Poisson(Y, X).fit()
# print m2.summary()