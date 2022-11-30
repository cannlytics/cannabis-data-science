"""
Network Visualization
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 4/15/2022
Updated: 4/17/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script visualizes relationships between businesses
in Washington State.

Data sources:

    - CCRS PRR All Data Up To 3-12-2022
    https://lcb.app.box.com/s/7pi3wqrmkuo3bh5186s5pqa6o5fv8gbs

Setup:

    1. pip install cannlytics

"""
# External imports. Using Cannlytics!
from cannlytics.utils.utils import snake_case
import pandas as pd
import seaborn as sns

# Internal imports. Using the new CCRS interface!
from ccrs import CCRS
from ccrs.constants import datasets

# Create a place for your data to live.
DATA_DIR = 'D:\\data\\washington\\CCRS PRR All Data Up To 3-12-2022'


#------------------------------------------------------------------------------
# Read the data.
#------------------------------------------------------------------------------

# Read in the first 1000 transfers.
dataset = 'Transfers_0'
transfers = pd.read_excel(
    f'{DATA_DIR}/{dataset}/{dataset}.xlsx',
    usecols=datasets['transfers']['fields'],
    parse_dates=datasets['transfers']['date_fields'],
    nrows=1000,
    skiprows=2,
)


#------------------------------------------------------------------------------
# Clean the data.
#------------------------------------------------------------------------------

# Format the field names.
transfers.columns = [snake_case(x) for x in transfers.columns]

# Clean the license numbers.
transfers['origin_license_number'] = transfers['origin_license_number'].str.strip()
transfers['destination_license_number'] = transfers['destination_license_number'].str.strip()

# Remove test transfers.
tests = ['1234', 'test', 'TEST', 'OPENTHC', 'OPENTH',
         'The Slow Burn - 40th', '123445', '123456',
         'JWPP', 'JGFT3', '999999','adsasdasdasd',
         'x', 'ss', 'werwer', 'na']
transfers = transfers.loc[~(transfers['origin_license_number'].isin(tests))]
transfers = transfers.loc[~(transfers['destination_license_number'].isin(tests))]


#------------------------------------------------------------------------------
# Organize the data.
#------------------------------------------------------------------------------

# Format the data as network data: source, target, type (optional), weight.
origins = transfers.groupby('origin_license_number')
groups = [
    'origin_license_number',
    'destination_license_number',
    # 'transportation_type',
]
group = transfers.groupby(groups, as_index=False)
counts = group['serial'].count()


#------------------------------------------------------------------------------
# Augment the data.
#------------------------------------------------------------------------------

# Initialize a CCRS client.
ccrs = CCRS()

# Read licensee data.
licensees = ccrs.read_licensees(DATA_DIR)

# Remove sensitive fields.
redacted = ['email_address', 'phone_number', 'created_by']
licensees.drop(columns=redacted, inplace=True)


#------------------------------------------------------------------------------
# Look at the data.
#------------------------------------------------------------------------------

import networkx as nx
import matplotlib.pyplot as plt

# Create a networkx graph.
G = nx.from_pandas_edgelist(
    counts,
    source='origin_license_number',
    target='destination_license_number',
    edge_attr='serial',
)

# Visualize who's transferring samples with who.
nx.draw_circular(G)
plt.show()


#------------------------------------------------------------------------------
# Summarize the data.
#------------------------------------------------------------------------------

# Calculate the degree of centrality for each licensee.
centrality = {}
for k, v in sorted(nx.degree_centrality(G).items(), key=lambda x: -x[1]):
    centrality[k] = v
stats = pd.DataFrame.from_dict(
    centrality,
    orient='index',
    columns=['centrality'],
)

# Calculate a "hub" metric.
hub = {}
for k, v in sorted(nx.eigenvector_centrality(G).items(), key=lambda x: -x[1]):
    hub[k] = v
stats['hub'] = pd.Series(hub)

# Calculate a "between" metric.
between = {}
for k, v in sorted(nx.betweenness_centrality(G).items(), key=lambda x: -x[1]):
    between[k] = v
stats['between'] = pd.Series(between)

# Calculate a "closeness" metric.
closeness = {}
for k, v in sorted(nx.closeness_centrality(G).items(), key=lambda x: -x[1]):
    closeness[k] = v
stats['closeness'] = pd.Series(closeness)

# Visualize the statistics.
correlation = stats.corr()
sns.heatmap(correlation, annot=True, cmap='vlag_r')
plt.show()

# Visualize a metric.
sns.displot(
    stats['centrality'],
    bins=int(len(stats) / 5),
    color='blue',
)
plt.title('Centrality of Licensees in Washington')
plt.show()


#------------------------------------------------------------------------------
# Build a model for the data: Community Detection.
#------------------------------------------------------------------------------

from networkx.algorithms.community import greedy_modularity_communities
from matplotlib import colors as mcolors

# Define colors.
css4_colors = list(mcolors.CSS4_COLORS.values())
rgb = [mcolors.to_rgb(c) for c in css4_colors]
hsv = [mcolors.rgb_to_hsv(c) for c in rgb]

# Assign colors by a statistic.
maximum = stats['centrality'].max()
positions = nx.spring_layout(G)
sets = greedy_modularity_communities(G)
sets.reverse()
node_colors = []
for c in sets:
    for n in c:
        x = stats.loc[n]['centrality']
        color = hsv[0]
        color[1] = x / maximum
        shade = mcolors.hsv_to_rgb(color)
        node_colors.append(shade)
sorted_colors = sorted(node_colors, key=lambda x: (x[1], x[0], x[2]))
sorted_colors.reverse()
nx.draw_networkx_edges(G, positions)
nx.draw_networkx_nodes(G, positions, node_color=sorted_colors)
plt.axis('off')
plt.title('Greedy Modularity Communities')
plt.show()


#------------------------------------------------------------------------------
# Analyze the data to gain actionable insights.
#------------------------------------------------------------------------------

# Identify the most influent node in the group.
# This "top dog" has won victory through hard-fought competition and knowing
# how they got there seems of utmost importance.
top_dog = stats.loc[stats['centrality'] == stats['centrality'].max()]
top_dog_licensee = licensees.loc[licensees['license_number'] == top_dog.index[0]]
top_dog_licensee.index = top_dog_licensee.license_number
top_dog_data = top_dog.join(top_dog_licensee)
print('Top Dog:', top_dog_data.iloc[0]['name'])

# Identify the movers and shakers in the industry for future study.
q = 0.975
movers_and_shakers = stats.loc[stats['centrality'] >= stats['centrality'].quantile(q)]
top_licensee = licensees.loc[licensees['license_number'].isin(movers_and_shakers.index)]
top_licensee.index = top_licensee.license_number
top_licensee_data = movers_and_shakers.join(top_licensee)

# Define characteristics of groups of users.
print('Movers and shakers:')
print(top_licensee_data[['name', 'county', 'centrality']])
spokane = top_licensee_data.loc[top_licensee_data.county == 'Spokane']
king = top_licensee_data.loc[top_licensee_data.county == 'King']
print('Percent of top players in King county:', len(king) / len(top_licensee_data))
print('Percent of top players in Spokane county:', len(spokane) / len(top_licensee_data))

# Identify licensees who could use some networking help,
# i.e. the licensees with the fewest connections and the
# likely higher marginal return from new connections.
marketing_aid = stats.loc[
    stats['centrality'] <= stats.centrality.quantile(.15)
]


#------------------------------------------------------------------------------
# Future work: Predict the data!
#------------------------------------------------------------------------------

# Future work: Use a Poisson regression to estimate the number of transfers
# from nearby partners to plan potential profitable routes for
# transport companies.


# Future work: See if network effects may help predict exit rates.
