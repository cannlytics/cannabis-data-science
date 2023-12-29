"""
Retailers per Capita | Cannabis Data Science

Author: Keegan Skeate
Contact: <keegan@cannlytics.com>
Created: 11/17/2021
Updated: 12/26/2023
License: MIT License <https://opensource.org/licenses/MIT>

Data Sources:
   
    Illinois Department of Financial and Professional Regulation
    
        - Illinois adult use cannabis monthly sales figures
        https://www.idfpr.com/Forms/AUC/2021%2011%2002%20IDFPR%20monthly%20adult%20use%20cannabis%20sales.pdf
        
        - Licensed Adult Use Cannabis Dispensaries
        https://www.idfpr.com/LicenseLookup/AdultUseDispensaries.pdf


    Massachusetts Cannabis Control Commission
    
        - Approved Massachusetts Licensees
        https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/hmwt-yiqy
        
        - Plant Activity and Volume
        https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/j3q7-3usu

    
    Nevada
    
        https://ccb.nv.gov/ccb-dot-release-annual-cannabis-taxable-sales-data-fy21/
        https://ccb.nv.gov/media/#item-3
        
    
        - Nevada Cannabis Dispensary Market Overview Technical Memorandum
        https://www.leg.state.nv.us/App/NELIS/REL/81st2021/ExhibitDocument/OpenExhibitDocument?exhibitId=50417&fileDownloadName=SB%20235%20RCG%20Economic%20impact%20report%202021-3-30%20NV%20Cannabis%20Dispensary%20Mkt%20Analysis-Final%20%281%29.pdf


    Fed Fred
    
        - State populations   
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

#-----------------------------------------------------------------------
# Estimate the relationship between retailers per capita and
# sales per retailer.
#-----------------------------------------------------------------------

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

# TODO: Look at the normality of the regression errors.


#-----------------------------------------------------------------------
# Create a beautiful visualization.
#-----------------------------------------------------------------------

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


#-----------------------------------------------------------------------
# Calculate market performance statistics.
#-----------------------------------------------------------------------

# Read FRED API key.
config = dotenv_values('../.env')
fred_api_key = config.get('FRED_API_KEY')

# Get the population for each state.
state_populations = {}
states = ['MA']
for state in states:
    population = get_state_current_population(state, fred_api_key)
    state_populations[state] = population['population']
