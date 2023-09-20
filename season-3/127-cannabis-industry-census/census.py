"""
Cannabis Industry Census
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 9/20/2023
Updated: 9/20/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# External imports:
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Setup plotting style.
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
})


# Read all licenses.
historic_licenses = pd.read_csv('./data/licenses-2022-10-06.csv')
licenses = pd.read_csv('./data/licenses-all-latest.csv')

# Hot-fix: recode mis-coded states.
to_replace = ["IL.", "Il.", "Illinois", "Lake", "IL.62301", "60608", "Data Not Available"]
licenses['premise_state'] = licenses['premise_state'].replace(to_replace, 'IL')
historic_licenses['premise_state'] = historic_licenses['premise_state'].replace(to_replace, 'IL')

# Count licenses by state.
state_counts = licenses['premise_state'].value_counts().reset_index()
state_counts.columns = ['state', 'count']

# Create a bar chart of licenses by state.
plt.figure(figsize=(15, 10))
plt.bar(state_counts['state'], state_counts['count'], color='teal')
plt.title('Number of Licenses by State')
plt.xlabel('State')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Get historic and latest license counts by state.
historic_state_counts = historic_licenses['premise_state'].value_counts().reset_index()
historic_state_counts.columns = ['state', 'historic_count']
latest_state_counts = licenses['premise_state'].value_counts().reset_index()
latest_state_counts.columns = ['state', 'latest_count']
merged_counts = pd.merge(historic_state_counts, latest_state_counts, on='state', how='outer').fillna(0)

# Plot historic versus current licenses.
bar_width = 0.35
index = np.arange(len(merged_counts['state']))
plt.figure(figsize=(15, 10))
bars1 = plt.bar(index, merged_counts['historic_count'], bar_width, color='skyblue', label='Historic Licenses')
bars2 = plt.bar(index + bar_width, merged_counts['latest_count'], bar_width, color='teal', label='Latest Licenses')
plt.xlabel('State')
plt.ylabel('Counts')
plt.title('Number of Licenses by State: Historic vs. Latest')
plt.xticks(index + bar_width / 2, merged_counts['state'], rotation=45)  # place the tick in the middle of the bars
plt.legend()
plt.tight_layout()
plt.show()


def create_retailer_map(
        df,
        color='crimson',
        filename=None,
        lat='premise_latitude',
        long='premise_longitude',
    ):
    """Create a map of licensed retailers."""
    m = folium.Map(
        location=[39.8283, -98.5795],
        zoom_start=3,
        control_scale=True,
    )
    for _, row in df.iterrows():
        folium.Circle(
            radius=5,
            location=[row[lat], row[long]],
            color=color,
        ).add_to(m)
    if filename:
        m.save(filename)
    return m


# Create a map of all licenses.
data = licenses.loc[
    (~licenses['premise_latitude'].isnull()) &
    (~licenses['premise_longitude'].isnull())
]
map_file = './figures/cannabis-licenses-map.html'
m = create_retailer_map(data, filename=map_file)
print('Saved map to', map_file)


# Create a choropleth map of licenses by state.
# import plotly.express as px

# FIXME: Create Chloropleth
# fig = px.choropleth(
#     state_counts, 
#     locations='state', 
#     locationmode="USA-states", 
#     color='count',
#     hover_name='state',
#     scope="usa",
#     color_continuous_scale="Viridis",
#     title="Licenses by State"
# )
# fig.show()


# TODO: Use NLP to look at popular naming conventions for cannabis businesses.
