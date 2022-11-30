"""
Price to Cannabinoid Regression Analysis
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 2/17/2022
Updated: 2/18/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description: This script uses the Bayesian approach to the linear
regression to estimate the relationship between price and cannabinoid
content in cannabis flower in Washington State.

Data sources:

    - WA State Traceability Data January 2018 - November 2021
    https://lcb.app.box.com/s/e89t59s0yb558tjoncjsid710oirqbgd

Resources:
    
    - Introduction to Bayesian Linear Regression
    https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7
    
    - Basic Bayesian Linear Regression Implementation
    https://github.com/WillKoehrsen/Data-Analysis/blob/master/bayesian_lr/Bayesian%20Linear%20Regression%20Demonstration.ipynb
    
    -Bayesian Linear Regression Lecture
    https://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/slides/lec19-slides.pdf

    - Introduction to Bayesian Regression
    https://statswithr.github.io/book/introduction-to-bayesian-regression.html

    - Implementation of Bayesian Regression
    https://www.geeksforgeeks.org/implementation-of-bayesian-regression/

"""

# External imports.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # pip install basemap
import pandas as pd
import requests
import seaborn as sns


# Define the plot style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
})
sns.set_palette('Set2', 10, .75)


#--------------------------------------------------------------------------
# Get Washington State price data.
#--------------------------------------------------------------------------

# Specify where the data lives.
DATA_DIR = 'D:\\leaf-data'
DATA_FILE = f'{DATA_DIR}/samples/random-sales-items-2022-02-16.csv'

# Read in the data.
data = pd.read_csv(DATA_FILE)

# Determine wholesale vs retail transactions.
data = data.loc[data['sale_type'] != 'wholesale']

# Drop observations with negative prices and prices in the upper quantile.
data = data.loc[data.price_total > 0]
data = data[data.price_total < data.price_total.quantile(.95)]

# Add a date column.
data['date'] = pd.to_datetime(data['created_at'])
data['day'] = data['date'].dt.date

# Estimate the average price of flower.
sample_type = 'usable_marijuana'
sample_type_data = data.loc[data.intermediate_type == sample_type]

# Identify the time period.
series = sample_type_data.loc[
    (sample_type_data['date'] >= pd.to_datetime('2019-01-01')) &
    (sample_type_data['date'] <= pd.to_datetime('2022-01-01'))
]


#--------------------------------------------------------------------------
# Analysis of price by cannabinoids.
#--------------------------------------------------------------------------

# TODO: Regress price on cannabinoid content.



