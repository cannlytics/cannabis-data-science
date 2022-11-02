
#-----------------------------------------------------------------------
# Get supplementary data.
#-----------------------------------------------------------------------

# from bs4 import BeautifulSoup
# from cannlytics.data.gis import get_state_population
# from cannlytics.utils.constants import state_names
# from dotenv import dotenv_values
# from fredapi import Fred
# import requests

# # Read your FRED API key.
# config = dotenv_values('../.env')
# fred_api_key = config['FRED_API_KEY']

# # Get the population for each state (in 2021).
# state_data = {}
# for state, abbv in state_names.items():
#     population = get_state_population(
#         abbv,
#         fred_api_key=fred_api_key,
#         obs_start='2021-01-01',
#     )
#     state_data[state] = {'population': population['population']}

# # Get the square miles of land for each state.
# url = 'https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_area'
# response = requests.get(url).text
# soup = BeautifulSoup(response, 'lxml')
# table = soup.find('table', class_='wikitable')
# for items in table.find_all('tr'):
#     data = items.find_all(['th', 'td'])
#     if data:
#         try:
#             rank = int(data[1].text)
#         except:
#             continue
#         state = data[0].text.replace('\n', '')
#         land_area = float(data[5].text.replace('\n', '').replace(',', ''))
#         state_data[state]['land_area_sq_mi']

# # Get the change in GDP for each state in 2022 Q1.
# code = 'NQGSP'
# fred = Fred(api_key=fred_api_key)
# for state, abbv in state_names.items():
#     try:
#         series = fred.get_series(abbv + code, '2021-10-01')
#     except ValueError:
#         continue
#     current, past = series[-1], series[-2]
#     change_gdp = ((current - past) / past) * 100
#     state_data[state]['change_gdp_2022_q1'] = change_gdp


#-----------------------------------------------------------------------
# Analyze the data.
#-----------------------------------------------------------------------

# import statsmodels.api as sm

# # FIXME: Compile all of the state statistics.
# stats = pd.DataFrame()

# # TODO: Count the number of retailers by state.


# # TODO: Calculate retailers per capita (100,000) by state.


# # TODO: Calculate retailers per 100 square miles by state.


# # TODO: Create `adult_use` dummy variable. Assign 0 `retailers_per_capita`.


# # Regress GDP on adult-use status and retailers per capita.
# Y = stats['change_gdp_2022_q1']
# X = stats[['adult_use', 'retailers_per_capita']]
# X = sm.add_constant(X)
# regression = sm.OLS(Y, X).fit()
# print(regression.summary())

# # Interpret the relationships.
# beta = regression.params.adult_use
# statement = """If a state permitted adult-use at the start of 2022,
# then everything else held constant one would expect
# GDP in 2022 Q1 to change by {}.
# """.format(beta)
# print(statement)

# # Interpret the relationships.
# beta = regression.params.retailers_per_capita
# statement = """If retailers per 100,000 adults increases by 1,
# then everything else held constant one would expect
# GDP in 2022 Q1 to change by {}.
# """.format(beta)
# print(statement)
