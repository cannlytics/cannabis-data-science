"""
Get Massachusetts Data | Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 9/20/2021
Updated: 9/30/2021
License: MIT License <https://opensource.org/licenses/MIT>

Data Sources:

    MA Cannabis Control Commission
    - Retail Sales by Date and Product Type: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/xwf2-j7g9
    - Approved Massachusetts Licensees: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/hmwt-yiqy
    - Average Monthly Price per Ounce for Adult-Use Cannabis: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/rqtv-uenj
    - Plant Activity and Volume: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/j3q7-3usu
    - Weekly sales by product type: https://dev.socrata.com/foundry/opendata.mass-cannabis-control.com/87rp-xn9v

    Fed Fred
    - MA Gross Domestic Product: https://fred.stlouisfed.org/series/MANQGSP
    - MA Civilian Labor Force: https://fred.stlouisfed.org/series/MALF
    - MA All Employees: https://fred.stlouisfed.org/series/MANA
    - MA Avg. Weekly Wage: https://fred.stlouisfed.org/series/LES1252881600Q
    - MA Minimum Wage: https://fred.stlouisfed.org/series/STTMINWGMA
    - MA Population: https://fred.stlouisfed.org/series/MAPOP
"""
from dotenv import dotenv_values
from fredapi import Fred
import pandas as pd
import requests
from scipy.stats import pearsonr


def end_of_period_timeseries(df, period='M'):
    """Convert a DataFrame from beginning-of-the-period to
    end-of-the-period timeseries.
    Args:
        df (DataFrame): The DataFrame to adjust timestamps.
        period (str): The period of the time series, monthly "M" by default.
    Returns:
        (DataFrame): The adjusted DataFrame, with end-of-the-month timestamps.
    """
    df.index = df.index.to_period(period).to_timestamp(period)
    return df


def reverse_dataframe(df):
    """Reverse the ordering of a DataFrame.
    Args:
        df (DataFrame): A DataFrame to re-order.
    Returns:
        (DataFrame): The re-ordered DataFrame.
    """
    return df[::-1].reset_index(drop=True)


#--------------------------------------------------------------------------
# Get the data.
#--------------------------------------------------------------------------

# Setup Socrata API, get the App Token, and define the headers.
config = dotenv_values('../.env')
app_token = config.get('APP_TOKEN', None)
headers = {'X-App-Token': app_token}
base = 'https://opendata.mass-cannabis-control.com/resource'

# Get production stats (total employees, total plants, etc.) j3q7-3usu
url = f'{base}/j3q7-3usu.json'
params = {'$limit': 2000, '$order': 'activitysummarydate DESC'}
response = requests.get(url,  headers=headers, params=params)
production = pd.DataFrame(response.json(), dtype=float)
production = reverse_dataframe(production)
variables = [
    'activitysummarydate',
    'total_plantimmaturecount',
    'total_planttrackedcount',
    'total_plantfloweringcount',
    'total_plantvegetativecount',
    'total_plantdestroyedcount',
    'total_plantharvestedcount',
    'total_plantcount',
    'salestotal',
    'total_active_harvestcount',
    'total_active_packagecount',
    'total_plantbatchcount',
    'total_activeproducts',
    'total_activestrains',
    'total_employees'
]

#--------------------------------------------------------------------------
# Clean the data, standardize variables, and get supplemental data.
#--------------------------------------------------------------------------

# Initialize Fed Fred.
config = dotenv_values('../.env')
fred = Fred(api_key=config.get('FRED_API_KEY'))

# Find the observation time start.
start = production.activitysummarydate.min()
observation_start = start.split('T')[0]

# Calculate percent of the civilian labor force in Massachusetts.
labor_force = fred.get_series('MALF', observation_start=observation_start)
labor_force.index = labor_force.index.to_period('M').to_timestamp('M')

# Calculate sales difference.
production['sales'] = production['salestotal'].diff()

# Aggregate daily production data into totals.
production['date'] = pd.to_datetime(production['activitysummarydate'])
production.set_index('date', inplace=True)
monthly_avg_production = production.resample('M').mean()
quarterly_avg_production = production.resample('Q').mean()

# Calculate total employees as a percent of all employees in MA.
total_ma_employees = fred.get_series('MANA', observation_start=observation_start)
total_ma_employees = end_of_period_timeseries(total_ma_employees)
total_ma_employees = total_ma_employees.multiply(1000) # Thousands of people

# Get MA population (conjecturing that population remains constant in 2021).
population = fred.get_series('MAPOP', observation_start=observation_start)
population = end_of_period_timeseries(population, 'Y')
population = population.multiply(1000) # Thousands of people
new_row = pd.DataFrame([population[-1]], index=[pd.to_datetime('2021-12-31')])
population = pd.concat([population, pd.DataFrame(new_row)], ignore_index=False)

#--------------------------------------------------------------------------
# Calculate (estimate) GDP from the cannabis industry in MA.
# Correlate cannabis GDP and total MA GDP.
# Estimate and plot the percent of GDP that the cannabis industry institutes.
#--------------------------------------------------------------------------

# Calculate GDP from consumption (sales).
cannabis_gdp = production['sales']
quarterly_cannabis_gdp = cannabis_gdp.resample('Q').sum()
quarterly_cannabis_gdp = quarterly_cannabis_gdp.divide(1000000) # Millions of dollars
ma_gdp = fred.get_series('MANQGSP', observation_start=observation_start)
ma_gdp.index = ma_gdp.index.to_period('Q').to_timestamp('Q')

# Plot quarterly GDP.
quarterly_cannabis_gdp[:len(ma_gdp)].plot()
ma_gdp.plot()

# Calculate the correlation coefficient between cannabis GDP and MA GDP.
corr, _ = pearsonr(quarterly_cannabis_gdp[:len(ma_gdp)], ma_gdp)
print('Pearsons correlation: %.3f' % corr)

# Calculate cannabis percent of GDP.
cannabis_percent_of_gdp = (quarterly_cannabis_gdp / ma_gdp) * 100
cannabis_percent_of_gdp.plot()

# Calculate percent of annual GDP.
annual_cannabis_gdp = cannabis_gdp.resample('Y').sum()
annual_cannabis_gdp = annual_cannabis_gdp.divide(1000000) # Millions of dollars
annual_ma_gdp = ma_gdp.resample('Y').sum()
annual_percent_of_gdp = annual_cannabis_gdp / annual_ma_gdp * 100

# Plot the GDP and percent of GDP.
annual_cannabis_gdp.plot(
    kind='bar',
    title='Annual Cannabis GDP (Millions of Dollars) in MA'
)
annual_percent_of_gdp.plot()

# Calculate (Estimate) the cannabis GDP per Capita in MA.
gdp_per_capita = (annual_cannabis_gdp / population[0]) * 1000000

#--------------------------------------------------------------------------
# Calculate (estimate) Gross domestic income
#--------------------------------------------------------------------------

# Calculate the percent of the labor force that are cannabis employees.
cannabis_portion_of_labor_force = monthly_avg_production.total_employees \
                                  / labor_force * 100

# Get the average weekly wage in MA.
avg_weekly_wage = fred.get_series('LES1252881600Q', observation_start=observation_start)
avg_weekly_wage = end_of_period_timeseries(avg_weekly_wage, 'Q')

# Calculate GDI from labor (this excludes capital rents).
quarterly_gdi = quarterly_avg_production.total_employees * avg_weekly_wage * 4 * 3
(quarterly_gdi / 1000000).plot()

#--------------------------------------------------------------------------
# Visualize labor data.
#--------------------------------------------------------------------------

# Plot employees.
monthly_employee_avg = production['total_employees'].resample('M').mean()
monthly_employee_avg.plot()
total_ma_employees.plot()

# Calculate as a percent of all employees.
cannabis_employees_percent = monthly_employee_avg / total_ma_employees * 100
cannabis_employees_percent.plot()

# Correlate total employees with cannabis employees.
corr, _ = pearsonr(monthly_employee_avg[:len(total_ma_employees)], total_ma_employees)
print('Pearsons correlation: %.3f' % corr)

# Crispy plot of GDP per capita.
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

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
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
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
