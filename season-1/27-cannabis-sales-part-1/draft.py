# # Read in the inventories data.
# inventories_data = pd.read_csv(
#     f'../Inventories_0.csv',
#     sep='\t',
#     encoding='utf-16',
#     nrows=10000, # FIXME: Read in all the data!
# )

# # Combine the data sets.
# data = pd.merge(
#     left=lab_data,
#     right=inventories_data,
#     how='left',
#     left_on='for_mme_id',
#     right_on='global_id'
# )