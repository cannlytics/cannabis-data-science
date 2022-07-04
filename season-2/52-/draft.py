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


# Optional: Add data block.
# summary_stats = series.describe()
# summary_stats = summary_stats[['count', 'mean', 'std']]
# replacement_keys = {
#     'count': 'Count',
#     'mean': 'Avg.',
#     'std': 'Std. Dev.',
#     # 'min': 'Min',
#     # '25%': 'Q1',
#     # '50%': 'Median',
#     # '75%': 'Q3',
#     # 'max': 'Max',
# }
# summary_stats.index = list(replacement_keys.values())
# for key, value in summary_stats.items():
#     summary_stats[key] = str(round(value)).replace('.0', '')
# ax.annotate(
#     summary_stats.to_string(),
#     xy=(0.7, 0.6),
#     xycoords='axes fraction',
#     fontsize=32,
#     horizontalalignment='left',
#     verticalalignment='bottom',
# )
