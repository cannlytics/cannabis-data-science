# Draft material

# Optional: Optimize with list comprehension.
# distances, durations = pd.Series(
#     combine_distances(row, combinations)
#     for index, row in panel.iterrows()
# )

# Optional: Optimize with map
# distances, durations =  map(combine_distances, panel, combinations)

# Optional: Optimize matching with vectorization.
# def match_combinations(df, combinations):
#     return (
#         combinations.loc[
#             (combinations['sender'] == df['sender']) &
#             (combinations['recipient'] == df['recipient']) 
#         ].iloc[0][['distance', 'duration']]
#     )
# distances = match_combinations(panel, combinations)

# for index, row in monthly_transfers.iterrows():
#     print(index, row)
#     plt.bar(
#         day_transfers['date'],
#         day_transfers[name],
#         # left=left,
#         # color=colors[idx]
#     )
# fig = plt.figure(figsize=(15, 10))
# time_range = pd.date_range(start='2018-04-01', end='2021-10-31')
# for date in time_range:
#     day_transfers = panel.loc[panel['date'] == date]
#     day_transfers_by_type = day_transfers.groupby('combination')['count'].sum()
#     day_transfers_by_type.plot(
#         kind='bar',
#         stacked=True,
#         title='Transfers by Type in Washington State'
#     )
