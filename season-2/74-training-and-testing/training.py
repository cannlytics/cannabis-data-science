"""
Training and Testing Prediction Models
Cannabis Data Science #74 | 2022-07-13
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 7/7/2022
Updated: 7/12/2022
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
import ast

# External imports.
from cannlytics.utils.utils import snake_case
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


#-----------------------------------------------------------------------
# Preprocessing the Data
#-----------------------------------------------------------------------

# Read in the saved results.
datafile = f'../../.datasets/lab_results/aggregated-cannabis-test-results.xlsx'
data = pd.read_excel(datafile, sheet_name='psi_labs_raw_data')

# Optional: Drop rows with no analyses at this point.
unneeded_columns = ['coa_urls', 'date_received', 'method', 'qr_code',
                    'sample_weight']
data.drop(unneeded_columns, axis=1, inplace=True)

# Isolate a training sample.
sample = data.sample(1_000, random_state=420)

# Create both wide and long data for ease of use.
# See: https://rstudio-education.github.io/tidyverse-cookbook/tidy.html
# Normalize and clean the data. In particular, flatten:
# - `analyses`
# - `results`
# - `images`
# - `coa_urls`
wide_data = pd.DataFrame()
long_data = pd.DataFrame()
for index, row in sample.iterrows():
    series = row.copy()
    analyses = ast.literal_eval(series['analyses'])
    images = ast.literal_eval(series['images'])
    results = ast.literal_eval(series['results'])
    series.drop(['analyses', 'images', 'results'], inplace=True)
    
    # Code analyses.
    if not analyses:
        continue
    for analysis in analyses:
        series[analysis] = 1
    
    # Add to wide data.
    wide_data = pd.concat([wide_data, pd.DataFrame([series])])

    # Iterate over results, cleaning results and adding columns.
    # Future work: Augment results with key, limit, and CAS.
    for result in results:

        # Clean the values.
        analyte_name = result['name']
        measurements = result['value'].split(' ')
        try:
            measurement = float(measurements[0])
        except:
            measurement = None
        try:
            units = measurements[1]
        except:
            units = None
        key = snake_case(analyte_name)
        try:
            margin_of_error = float(result['margin_of_error'].split(' ')[-1])
        except:
            margin_of_error = None

        # Format long data.
        entry = series.copy()
        entry['analyte'] = key
        entry['analyte_name'] = analyte_name
        entry['result'] = measurement
        entry['units'] = units
        entry['margin_of_error'] = margin_of_error

        # Add to long data.
        long_data = pd.concat([long_data, pd.DataFrame([entry])])


# Fill null observations.
wide_data = wide_data.fillna(0)

# Rename columns
analyses = {
    'POT': 'cannabinoids',
    'RST': 'residual_solvents',
    'TERP': 'terpenes',
    'PEST': 'pesticides',
    'MICRO': 'micro',
    'MET': 'heavy_metals',
}
wide_data.rename(columns=analyses, inplace=True)
long_data.rename(columns=analyses, inplace=True)


#------------------------------------------------------------------------------
# Exploring the data.
#------------------------------------------------------------------------------

# TODO: Count the number of lab results scraped!


# TODO: Count the number of unique data points scraped!

# Count the number of various tests.
terpenes = wide_data.loc[wide_data['terpenes'] == 1]

# Find all of the analytes.
analytes = list(long_data.analyte.unique())

# Find all of the product types.
product_types = list(long_data.product_type.unique())


# TODO: Look at cannabinoid concentrations over time.


# Look at cannabinoid distributions by type.
flower = long_data.loc[long_data['product_type'] == 'Flower']
flower.loc[flower['analyte'] == '9_thc']['result'].hist(bins=100)

concentrates = long_data.loc[long_data['product_type'] == 'Concentrate']
concentrates.loc[concentrates['analyte'] == '9_thc']['result'].hist(bins=100)


# Look at terpene distributions by type!
terpene = flower.loc[flower['analyte'] == 'dlimonene']
terpene['result'].hist(bins=100)

terpene = concentrates.loc[concentrates['analyte'] == 'dlimonene']
terpene['result'].hist(bins=100)


# Find the first occurrences of famous strains.
gorilla_glue = flower.loc[
    (flower['product_name'].str.contains('gorilla', case=False)) |
    (flower['product_name'].str.contains('glu', case=False))    
]

# Create strain fingerprints: histograms of dominant terpenes.
compound = gorilla_glue.loc[gorilla_glue['analyte'] == '9_thc']
compound['result'].hist(bins=100)


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


# Optional: Create data / CoA NFTs for the lab results!!!
