"""
Predicting with Panel Data
Cannabis Data Science Meetup Group
Copyright (c) 2021 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 12/15/2021
Updated: 12/18/2021
License: MIT License <https://opensource.org/licenses/MIT>
"""
# External imports.
from dotenv import dotenv_values
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import pmdarima as pm
import requests
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import abline_plot

# Internal imports
from utils import (
    forecast_arima,
    format_millions,
    format_thousands,
    reverse_dataframe,
    set_training_period,
)

#--------------------------------------------------------------------------
# Read the data.
#--------------------------------------------------------------------------

panel_data = pd.read_excel('./data/state_panel_data.xlsx')

# Assign a time index.
panel_data.index = pd.to_datetime(panel_data['Unnamed: 0'])

variables = list(panel_data.columns)

#--------------------------------------------------------------------------
# Explore the data.
#--------------------------------------------------------------------------

# grouped = panel_data.groupby('state')

fig, ax = plt.subplots(figsize=(18, 10))
states = list(panel_data.state.unique())
for state in states:
    
    state_data = panel_data.loc[panel_data.state == state]
    ax = state_data['retailers_per_capita'].plot(label=state)
    # yaxis_format = FuncFormatter(format_millions)
    # ax.yaxis.set_major_formatter(yaxis_format)

plt.legend(fontsize=20)
plt.show()

#--------------------------------------------------------------------------
# Forecast into 2022.
#--------------------------------------------------------------------------

# Specifiy training time periods.
# train_start = '2020-06-01'
# train_end = '2021-10-25'

# Define forecast horizon.
# forecast_horizon = pd.date_range(
#     pd.to_datetime(train_end),
#     periods=60,
#     freq='w'
# )

# Create month fixed effects (dummy variables),
# excluding 1 month (January) for comparison.
# month_effects = pd.get_dummies(weekly_sales.index.month)
# month_effects.index = weekly_sales.index
# month_effects = set_training_period(month_effects, train_start, train_end)
# forecast_month_effects = pd.get_dummies(forecast_horizon.month)
# del month_effects[1]
# try:
#     del forecast_month_effects[1]
# except:
#     pass

# TODO: Create state fixed effects (dummy variables).

# Forecast sales.
# sales_model = pm.auto_arima(
#     set_training_period(weekly_sales, train_start, train_end),
#     X=month_effects,
#     start_p=0,
#     d=0,
#     start_q=0,
#     max_p=6,
#     max_d=6,
#     max_q=6,
#     seasonal=True,
#     start_P=0,
#     D=0,
#     start_Q=0,
#     max_P=6,
#     max_D=6,
#     max_Q=6,
#     information_criterion='bic',
#     alpha=0.2,
# )
# sales_forecast, sales_conf = forecast_arima(
#     sales_model,
#     forecast_horizon,
#     X=forecast_month_effects
# )

# def plot_forecast(
#         ax,
#         forecast,
#         historic=None,
#         conf=None,
#         title=None,
#         color=None,
#         formatter=None,
# ):
#     """Plot a time series forecast.
#     Args:
#         ax (): The axes on which to plot the forecast.
#         forecast (Series): The forecast to plot.
#         historic (Series): Optional historic time series to plot.
#         conf (Array): An optional 2xN array of lower and upper confidence
#             bounds for the forecast series.
#         title (str): An optional title to place above the chart.
#         color: (str): An optional color hex code.
#         formatter (func): An optional formatting function for the Y axis.
#     """
#     forecast.plot(color=color, style='--', label='Forecast')
#     if conf is not None:
#         plt.fill_between(
#             forecast.index,
#             conf[:, 0],
#             conf[:, 1],
#             alpha=0.1,
#             color=color,
#         )
#     if historic is not None:
#         historic.plot(color=color, label='Historic')
#     if title is not None:
#         plt.title(title, fontsize=24, pad=10)
#     if formatter is not None:
#         yaxis_format = FuncFormatter(formatter)
#         ax.yaxis.set_major_formatter(yaxis_format)
#     plt.gca().set(ylim=0)
#     plt.setp(ax.get_yticklabels()[0], visible=False)
#     plt.xlabel('')
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)

# # Define the plot style.
# plt.style.use('fivethirtyeight')
# plt.rcParams['font.family'] = 'Times New Roman'
# palette = sns.color_palette('tab10')

# # Plot all series.
# fig = plt.figure(figsize=(40, 25))

# # Plot sales.
# ax1 = plt.subplot(3, 2, 1)
# plot_forecast(
#         ax1,
#         sales_forecast,
#         historic=weekly_sales,
#         conf=sales_conf,
#         title='Cannabis Sales',
#         color=palette[0],
#         formatter=format_millions,
# )
