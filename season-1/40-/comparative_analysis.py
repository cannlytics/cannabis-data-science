"""
Comparative Analysis | Cannabis Data Science Meetup Group

Author: Keegan Skeate
Contact: <keegan@cannlytics.com>
Created: 11/17/2021
Updated: 11/24/2021
License: MIT License <https://opensource.org/licenses/MIT>

Objective:
    
    Analyze the structure and performance of the cannabis markets in
    various states by estimating the relationship between dispensaries
    per capita and revenue per dispensary.

Data Sources:
        
    Nevada
    
        - Nevada Cannabis Dispensary Market Overview Technical Memorandum
        https://www.leg.state.nv.us/App/NELIS/REL/81st2021/ExhibitDocument/OpenExhibitDocument?exhibitId=50417&fileDownloadName=SB%20235%20RCG%20Economic%20impact%20report%202021-3-30%20NV%20Cannabis%20Dispensary%20Mkt%20Analysis-Final%20%281%29.pdf

    Fed Fred
    
        - State populations
        https://fred.stlouisfed.org/

"""
# External imports.
from dotenv import dotenv_values
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import abline_plot

# Internal imports.
from get_state_data import get_state_current_population
from utils import format_millions, format_thousands

#--------------------------------------------------------------------------
# Estimate the relationship between dispensaries per capita and
# sales per dispensary.
#--------------------------------------------------------------------------

# Read in retailer statistics (from Nevada's technical memorandum).
retailer_stats = pd.read_excel('./data/retailer_stats.xlsx')

# Look at only observations with revenue per retailer.
stats = retailer_stats[~retailer_stats['revenue_per_retailer'].isnull()]

# Run a regression of sales per retailer on retailers per 100,000 adults.
Y = stats['revenue_per_retailer']
X = stats['retailers_per_100_000']
X = sm.add_constant(X)
regression = sm.OLS(Y, X).fit()
print(regression.summary())

# Interpret the relationship.
beta = regression.params.values[1]
statement = """If retailers per 100,000 adults increases by 1,
then everything else held constant one would expect
revenue per retailer to change by {}.
""".format(format_thousands(beta))
print(statement)

# Visualize the regression.
ax = stats.plot(
    x='retailers_per_100_000',
    y='revenue_per_retailer',
    kind='scatter'
)
abline_plot(
    model_results=regression,
    ax=ax
)
plt.show()

#--------------------------------------------------------------------------
# Create a beautiful visualization.
#--------------------------------------------------------------------------

# Set chart defaults.
plt.style.use('fivethirtyeight')
plt.rcParams['font.family'] = 'Times New Roman'

# Create the figure. 
fig, ax = plt.subplots(figsize=(15, 5))

# Write the text.
title = """The Relationship Between
Dispensaries per Capita and
Sales per Dispensary"""
notes = """Data: Annual dispensary statistics for {} observations in 2020.
Data Source: Nevada Cannabis Dispensary Market Overview Technical Memorandum."""
notes = notes.format(len(stats))

# Plot the points.
stats.plot(
    x='retailers_per_100_000',
    y='revenue_per_retailer',
    kind='scatter',
    ax=ax
)

# Annotate each point.
for index, row in stats.iterrows():
    point = (
        row['retailers_per_100_000'] + .15,
        row['revenue_per_retailer']
    )
    ax.annotate(row['state'], point, fontsize=14)
    
# Plot the regression line.
abline_plot(model_results=regression, ax=ax)

# Format the Y-axis.
yaxis_format = FuncFormatter(format_millions)
ax.yaxis.set_major_formatter(yaxis_format)
plt.gca().set(ylim=0)
plt.setp(ax.get_yticklabels()[0], visible=False)

# Plot the title, labels, and notes.
plt.ylabel('Annual Revenue per\n Dispensary ($)', fontsize=16)
plt.xlabel('Retailers per 100,000 Adults', fontsize=16)
plt.title(title, fontsize=21, pad=10)
plt.figtext(0.05, -0.125, notes, ha='left', fontsize=12)

# Format the plot by removing unnecessary ink.
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
# ax.tick_params(axis='both', which='major', labelsize=18)

# Save and show the figure.
plt.margins(1, 1)
plt.savefig(
    'figures/revenue_per_retailer_to_retailers_per_100_000.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.75,
    transparent=False,
)
plt.show()




#--------------------------------------------------------------------------
# FUTURE WORK: Get public cannabis data for several states.
#--------------------------------------------------------------------------

# TODO: Get California data.


# TODO: Get Oregon data.


# TODO: Get Colorado data.


# TODO: Read Illinois sales and retailer data.


# TODO: Get Massachusetts data.


#--------------------------------------------------------------------------
# FUTURE WORK: Calculate market performance statistics.
#--------------------------------------------------------------------------

# Read FRED API key.
config = dotenv_values('../.env')
fred_api_key = config.get('FRED_API_KEY')

# TODO: Get the population for each state.
state_populations = {}
states = ['MA']
for state in states:
    population = get_state_current_population(state, fred_api_key)
    state_populations[state] = population['population']

# TODO: Calculate dispensaries per capita in each state.

# TODO: Calculate sales per dispensary in each state.



#--------------------------------------------------------------------------
# FUTURE WORK: Compare sales per retailer across the various states.
#--------------------------------------------------------------------------

# TODO: Plot sales per retailer together.


# TODO: Calculate mean for each state.


# TODO: See if there is a statistical difference in the growth of sales in any state.
