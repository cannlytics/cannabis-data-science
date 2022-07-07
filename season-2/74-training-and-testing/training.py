"""
Training and Testing Prediction Models
Cannabis Data Science #74 | 2022-07-13
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 7/7/2022
Updated: 7/7/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    Analyze all of the PSI Labs test results, separating
    training and testing data to use for prediction models.

Data Sources:

    - PSI Labs Test Results
    URL: <https://results.psilabs.org/test-results/>

Resources:

    - Split / Explode a column of dictionaries into separate columns with pandas
    URL: https://stackoverflow.com/questions/38231591/split-explode-a-column-of-dictionaries-into-separate-columns-with-pandas

    - Tidyverse: Wide and Long Data Tables
    URL: https://rstudio-education.github.io/tidyverse-cookbook/tidy.html

"""
# Standard imports.
from datetime import datetime
from hashlib import sha256
import hmac
import os
from time import sleep

# External imports.
from cannlytics.utils.utils import snake_case
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import abline_plot


ANALYSES = {
    'cannabinoids': ['potency', 'POT'],
    'terpenes': ['terpene', 'TERP'],
    'residual_solvents': ['solvent', 'RST'],
    'pesticides': ['pesticide', 'PEST'],
    'microbes': ['microbial', 'MICRO'],
    'heavy_metals': ['metal', 'MET'],
}
ANALYTES = {
    # TODO: Define all of the known analytes from the Cannlytics library.
}
DECODINGS = {
    '<LOQ': 0,
    '<LOD': 0,
}


#-----------------------------------------------------------------------
# Preprocessing the Data
#-----------------------------------------------------------------------

# Read in the saved results.
datafile = f'../../.datasets/michigan/psi-lab-results-sample-2022-07-06.xlsx'
data = pd.read_excel(datafile)

# Optional: Drop rows with no analyses at this point.

# Create both wide and long data for ease of use.
# See: https://rstudio-education.github.io/tidyverse-cookbook/tidy.html
# TODO: Normalize and clean the data. In particular, flatten:
# - `analyses`
# - `results`
# - `images`
# - `coa_urls`
wide_data = pd.DataFrame()
long_data = pd.DataFrame()
for index, row in data.iterrows():
    series = row.copy()
    analyses = series['analyses']
    images = series['images']
    results = series['results']
    series.drop(['analyses', 'images', 'results'], inplace=True)
    if not analyses:
        continue

    # TODO: Iterate over results, cleaning results and adding columns.
    # Future work: Augment results with key, limit, and CAS.


# Optional: Create data / CoA NFTs for the lab results!!!


# TODO: Save the curated data, both wide and long data.



#------------------------------------------------------------------------------
# Exploring the data.
#------------------------------------------------------------------------------

# TODO: Count the number of lab results scraped!


# TODO: Count the number of unique data points scraped!


# TODO: Look at cannabinoid concentrations over time.


# TODO: Look at cannabinoid distributions by type.


# TODO: Look at terpene distributions by type!


# TODO: Find the first occurrences of famous strains.


#-----------------------------------------------------------------------
# Modeling the data.
#-----------------------------------------------------------------------

# TODO: Given a lab result, predict if it's in the Xth percentile.


# TODO: Use in ARIMA model to approach the question:
# Are terpene or cannabinoid concentrations increasing over time by sample type?
# - total_terpenes
# - D-limonene
# - beta-pinene
# - myrcene
# - caryophyllene
# - linalool
# - cbg
# - thcv
# - total_thc
# - total_cbd
# - total_cannabinoids


#-----------------------------------------------------------------------
# Training and testing the model.
#-----------------------------------------------------------------------

# TODO: Separate results after 2020 as test data.


# TODO: Estimate a large number of ARIMA models on the training data,
# use the models to predict the test data, and measure the accuracies.


# TODO: Pick the model that predicts the test data the best.


#-----------------------------------------------------------------------
# Evaluating the model.
#-----------------------------------------------------------------------

# TODO: Re-estimate the model with the entire dataset.


# TODO: Predict if cannabinoid and terpene concentrations are trending
# up or down and to what degree if so.


# TODO: Take away an insight: Is there statistical evidence that
# cannabis cultivated in Michigan is successfully being bred to, on average,
# have higher levels of cannabinoids or terpenes? If so, which compounds?


# TODO: Forecast: If the trend continues, what would cannabis look like
# in 10 years? What average cannabinoid and terpene concentration can
# we expect to see in Michigan in 2025 and 2030?
