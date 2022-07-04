# def avg_concentration(row, results):
#     licensee_results = results.loc[results.code == row.code]
#     return licensee_results.total_cannabinoids.mean()


# def std_concentration(row, results):
#     licensee_results = results.loc[results.code == row.code]
#     return licensee_results.total_cannabinoids.std()


# def total_samples(row, results):
#     licensee_results = results.loc[results.code == row.code]
#     return licensee_results.total_cannabinoids.std()


# licensees.apply(lambda row : avg_concentration(row, results), axis=1) 



# TODO: Calculate avg concentration, std concentration, total_samples for each licensee.
# results.set_index('code')
# avg_concentration = results.groupby('code')['total_cannabinoids'].mean()
# licensees['avg_concentration'] = results['total_cannabinoids'].where(
#     results['code']==licensees['code'],
#     results['total_cannabinoids'].mean()
# )

# import numpy as np
# results.groupby('code')['total_cannabinoids'].agg({'total_cannabinoids': np.mean})



# top_10 = products.groupby(['mme_id']).count()
# top_10.sort_values(by='mme_id', ascending=False, inplace=True)
# top_10 = top_10[:10]


# FIXME: Box and Swarm Plot
# plt.figure(figsize=(12, 8))
# ax = sns.boxplot(
#     x='avg_concentration',
#     y='std_deviation',
#     data=licensees.loc[
#         (licensees.type == 'cultivator') |
#         (licensees.type == 'cultivator_production')
#     ]
# )
# # ax = sns.swarmplot(x="day", y="total_bill", data=tips, color=".25")
# plt.show()