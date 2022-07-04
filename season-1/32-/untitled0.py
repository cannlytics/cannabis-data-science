"""
Crispy Bar Chart | Cannlytics for Cannabis Data Science Meetup Group

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 9/30/2021
Updated: 9/30/2021
License: MIT License <https://opensource.org/licenses/MIT>
"""

import matplotlib.pyplot as plt


def crispy_bar_chart():
    """Crispy plot of GDP per capita.
    Args:
        
    Returns:
        (figure): The chart figure for any post-processing.
    """
    raise NotImplementedError

# Create a figure.
fig, ax = plt.subplots(figsize=(7, 7))

# Set the chart font, style, and text color.
plt.style.use('fivethirtyeight')
plt.rcParams['axes.edgecolor'] = '#1a1a1a'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['text.color'] = '#1a1a1a'
plt.rcParams['xtick.color'] = '#1a1a1a'
plt.rcParams['ytick.color'] = '#1a1a1a'

# Format the X-axis labels.
x_ticks = [x.strftime('%Y') for x in list(gdp_per_capita.index)]

# Define the color.
deep_purple = (0.3, 0.1, 0.4, 0.6)

# Plot the data.
plt.bar(
    x_ticks,
    gdp_per_capita.values,
    align='center',
    alpha=.5,
    color=deep_purple,
    width=.625,
)

# Add a title.
plt.title(
    'GDP per Capita from Adult-Use\nCannabis in Massachusetts',
    fontsize=24,
    pad=15
)

# Format Y axis labels.
ax.yaxis.set_major_formatter(tick) 

# Add notes.
notes = """Data: 1,069 daily cannabis sales totals between 10/15/2018
and 9/27/2021 in Massachusetts.

Data Sources: Cannabis sales data published by the Massachusetts
Cannabis Control Commission and annual population data published
by the U.S. Census Bureau and retrieved from FRED, Federal Reserve
Bank of St. Louis.

Notes: The timeframes of 10/16/2018 through 12/25/2018 and
3/28/2020 through 5/24/2020 were missing sales data. The population
for 2020 is used in the calculation for GDP per capita in 2021.
"""
plt.figtext(0.05, -0.475, notes, ha='left', fontsize=18)

# Annotate each bar.
for i in range(len(gdp_per_capita)):
    plt.text(
        x=i,
        y=gdp_per_capita[i] + 2,
        s='$%.2f' % gdp_per_capita[i],
        size=18,
        ha='center',
)

# Hide unnecessary spines and ticks.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.gca().xaxis.grid(False)

# Save the figure.
plt.savefig(
    'figures/gdp_per_capita_ma',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.5,
    transparent=False,
)

# Show the figure.
plt.show()
