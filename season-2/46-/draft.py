"""
Forecasting Draft

Author: Keegan Skeate <keegan@cannlytics.com>
Created: 12/15/2021
Updated: 1/5/2022
License: MIT License <https://opensource.org/licenses/MIT>
"""
#--------------------------------------------------------------------------
# Begin to visualize the data.
#--------------------------------------------------------------------------

# # Plot sales by state.
# fig, ax = plt.subplots(figsize=(16, 8))
# panel_data.groupby('state')['total_sales'].plot()
# yaxis_format = FuncFormatter(format_millions)
# ax.yaxis.set_major_formatter(yaxis_format)
# plt.title('Monthly Cannabis Sales by State')
# plt.legend()
# plt.show()

# # Plot prices by state.
# fig, ax = plt.subplots(figsize=(16, 8))
# panel_data.groupby('state')['avg_price_per_gram'].plot()
# plt.title('Average Monthly Price per Gram')
# plt.legend()
# plt.show()


#--------------------------------------------------------------------------
# Forecast total sales into 2022.
#--------------------------------------------------------------------------

# # Iterate over the states, forecasting sales for each state.
# all_forecasts = []
# states = list(panel_data['state'].unique())
# states.sort()
# for state in states:

#     # Define the series.
#     series = panel_data.loc[panel_data.state == state]

#     # Create month fixed effects.
#     excluded_month = 1
#     month_effects = pd.get_dummies(series.index.month)
#     month_effects.index = series.index
#     del month_effects[excluded_month]

#     # Define forecast horizon.
#     forecast_start = series.index.max() + relativedelta.relativedelta(months=1)
#     forecast_horizon = pd.date_range(
#         start=forecast_start,
#         end=pd.to_datetime('2023-01-01'),
#         freq='M',
#     )

#     # Define forecasted fixed effects.
#     forecast_month_effects = pd.get_dummies(forecast_horizon.month)
#     try:
#         del forecast_month_effects[excluded_month]
#     except KeyError:
#         pass
    
#     # Build a forecasting model.
#     sales_model = pm.auto_arima(
#         series.total_sales,
#         X=month_effects,
#         start_p=0,
#         d=0,
#         start_q=0,
#         max_p=6,
#         max_d=6,
#         max_q=6,
#         seasonal=True,
#         start_P=0,
#         D=0,
#         start_Q=0,
#         max_P=6,
#         max_D=6,
#         max_Q=6,
#         information_criterion='bic',
#         alpha=0.2,
#     )

#     # Forecast sales.
#     sales_forecast, sales_conf = forecast_arima(
#         sales_model,
#         forecast_horizon,
#         X=forecast_month_effects
#     )
    
#     # Record the forecast.
#     all_forecasts.append(sales_forecast)

# # Aggregate all of the forecasts.
# forecasts = pd.concat(all_forecasts, axis=1)
# forecasts.columns = ['{}_forecast'.format(x.lower()) for x in states]


#--------------------------------------------------------------------------
# Visualize the 2022 forecasts.
#--------------------------------------------------------------------------

# # Define a unique color for each state.
# state_colors = {}
# palette = sns.color_palette('tab10')
# for index, state in enumerate(states):
#     state_colors[state] = palette[index]

# # Plot historic and predicted sales by state.
# fig, ax = plt.subplots(figsize=(16, 8))
# for state in states:
#     color = state_colors[state]
#     panel_data.loc[panel_data.state == state]['total_sales'].plot(color=color, label=f'{state} Sales')
#     forecasts['{}_forecast'.format(state.lower())].plot(color=color, linestyle='--', label=f'{state} Forecast')
# yaxis_format = FuncFormatter(format_millions)
# ax.yaxis.set_major_formatter(yaxis_format)
# plt.title('Monthly Forecasted Cannabis Sales by State')
# plt.legend()
# plt.savefig(
#     'figures/state_cannabis_sales_forecast.png',
#     dpi=300,
#     bbox_inches='tight',
#     pad_inches=0.75,
#     transparent=False,
# )
# plt.show()


#--------------------------------------------------------------------------
# Analyze 2022 forecasts.
#--------------------------------------------------------------------------

# # Calculate the expected annual sales in 2022 in each state.
# print('Predicted total cannabis sales by state:')
# total_cannabis_sales = 0
# for state in states:
    
#     # Estimate 2022 total sales.
#     state_forecast = forecasts['{}_forecast'.format(state.lower())]
#     next_year_forecast = state_forecast.loc[
#         (state_forecast.index >= pd.to_datetime('2022-01-01')) &
#         (state_forecast.index < pd.to_datetime('2023-01-01'))
#     ]
#     next_year_sales = next_year_forecast.sum()
#     print(state, '≈', format_millions(next_year_sales))
#     total_cannabis_sales += next_year_sales

#     # Estimate number of schools that could be built with a given tax rate.
#     tax = 0.07
#     elementary_school = 7_393_000 # https://www.rsmeans.com/model-pages/elementary-school
#     high_school = 20_592_000 # https://www.rsmeans.com/model-pages/high-school
#     number_of_new_elementary_schools = (next_year_sales * tax) / elementary_school
#     number_of_new_high_schools = (next_year_sales * tax) / high_school
#     print(state, 'could build', round(number_of_new_elementary_schools), 'new elementary schools.')
#     print(state, 'could build', round(number_of_new_high_schools), 'new high schools.')

#     # Calculate sales per capita.
#     series = panel_data.loc[panel_data.state == state]
#     latest_pop = series.population.iloc[-1]
#     cannabis_gdp_per_capita = round(next_year_sales/ latest_pop, 2)
#     print(state, 'GDP per Capita from cannabis:', cannabis_gdp_per_capita)

# total_sales = format_millions(total_cannabis_sales)
# print('A forecast of cannabis sales in 2022 in the U.S. is at least', total_sales)


# TODO: Calculate the expected percent growth from 2021.

#--------------------------------------------------------------------------
# TODO: Forecast states without available or historic data with
# a structural regression model.
#--------------------------------------------------------------------------

# Fit a regression of sales on population, months since legalization, and
# a month dummy variable.


#--------------------------------------------------------------------------
# TODO: Forecast additional variables:
# - prices, consumption, patients
#--------------------------------------------------------------------------



# Plot in one-shot.
# colors = sns.color_palette(
#     # 'Set2',
#     cmap='Spectral',
#     # n_colors=len(state_data),
# )
# state_forecasts.groupby('state')['sales_forecast'].plot(
#     color=palette
# )
