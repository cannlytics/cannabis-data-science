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