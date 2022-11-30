
# Top licensees.
most_central_licensees = stats.loc[
    stats['centrality'] >= stats['centrality'].quantile(0.95)
]
top_licensee_counts = counts.loc[
    counts['origin_license_number'].isin(list(most_central_licensees.index))
]

# Create a networkx graph.
G = nx.from_pandas_edgelist(
    top_licensee_counts,
    source='origin_license_number',
    target='destination_license_number',
    edge_attr='serial',
)