# -*- coding: utf-8 -*-
"""
Cannabis Data Science | Meetup 2021-02-10

Prepared by: Keegan Skeate <keeganskeate@gmail.com>
Created on 2/7/2021

License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Description:
    Examples of way to work with WA cannabis traceability data.

Data sources:
    https://lcb.wa.gov/records/make-public-records-request
    
Resources:
    https://stackoverflow.com/questions/42358259/how-to-parse-tsv-file-with-python
    https://hackersandslackers.com/series/data-analysis-pandas/
    https://hackersandslackers.com/plotting-data-seaborn-pandas/

"""
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import abline_plot


#-----------------------------------------------------------------------------#
# Read in lab results from the .csv file obtained through public records request.
# https://stackoverflow.com/questions/18171739/unicodedecodeerror-when-reading-csv-file-in-pandas-with-python
# https://stackoverflow.com/questions/18039057/python-pandas-error-tokenizing-data
#-----------------------------------------------------------------------------#
file_name = "LabResults_0.csv"
data = pd.read_csv(
    file_name,
    encoding="utf-16",
    sep='\t',
)
print("Number of observations:", len(data))


#-----------------------------------------------------------------------------#
# Look at the data
#-----------------------------------------------------------------------------#
obs = data.iloc[0].to_dict()
print(list(data.columns))

#-----------------------------------------------------------------------------#
# Sum tests per day
#-----------------------------------------------------------------------------#
LIMIT = 10000
sample = data[-LIMIT:]
print(list(sample.iloc[0].keys()))
print(list(sample["intermediate_type"].unique()))

#-----------------------------------------------------------------------------#
# Create some variables
#-----------------------------------------------------------------------------#
sample["time"] = pd.to_datetime(sample["tested_at"])
sample["lab"] = sample["global_id"].str.slice(2, 4)
print("Number of labs:", len(sample["lab"].unique()))

#-----------------------------------------------------------------------------#
# Estimate test per day in February
#-----------------------------------------------------------------------------#
flower = sample.loc[sample["intermediate_type"] == "flower_lots"]

#-----------------------------------------------------------------------------#
# Perform some simple statistics.
#-----------------------------------------------------------------------------#
high_thc = flower.loc[(flower["cannabinoid_d9_thca_percent"] <= 35) & 
                      (flower["cannabinoid_d9_thca_percent"] > 20) &
                      (flower["cannabinoid_cbda_percent"] < 0.5) ]


high_cbd = flower.loc[flower["cannabinoid_cbda_percent"] > 5]
high_cbd["moisture_content_water_activity_rate"].plot()

#-----------------------------------------------------------------------------#
# Perform a regression
# https://stackoverflow.com/questions/44325017/how-to-build-a-regression-model-in-python
# https://stackoverflow.com/questions/42261976/how-to-plot-statsmodels-linear-regression-ols-cleanly
# https://scikit-learn.org/stable/modules/linear_model.html
#-----------------------------------------------------------------------------#

# Independent variables.
X = high_thc[[
    "cannabinoid_cbda_percent",
]].fillna(0)


# Dependent variable.
Y = high_thc["cannabinoid_d9_thca_percent"].fillna(0)

# Fit a regression model.
X = sm.add_constant(X)
model = sm.OLS(Y, X)
regression_results = model.fit()
print(regression_results.summary())

# Plot the regression
ax = high_thc.plot(
    x='cannabinoid_cbda_percent',
    y='cannabinoid_d9_thca_percent',
    kind='scatter'
)
abline_plot(model_results=regression_results, ax=ax)


#-----------------------------------------------------------------------------#
# Trend an analyte (butane) over time.
# https://stackoverflow.com/questions/36410075/select-rows-from-a-dataframe-based-on-multiple-values-in-a-column-in-pandas
# https://stackoverflow.com/questions/17706109/summing-the-number-of-occurrences-per-day-pandas
# https://apps.leg.wa.gov/wac/default.aspx?cite=314-55-102
# https://stackoverflow.com/questions/10998621/rotate-axis-text-in-python-matplotlib
# https://stackoverflow.com/questions/7917107/add-footnote-under-the-x-axis-using-matplotlib
#-----------------------------------------------------------------------------#
concentrate_types = [
    "hydrocarbon_concentrate",
    "concentrate_for_inhalation",
    "non-solvent_based_concentrate",
    "co2_concentrate",
    "food_grade_solvent_concentrate",
    "ethanol_concentrate",
]
concentrates = sample.loc[sample["intermediate_type"].isin(concentrate_types)]

# Aggregate data by day.
daily_concentrates = concentrates.groupby(concentrates.time.dt.date).mean()
daily_concentrates = daily_concentrates.loc[daily_concentrates.index > pd.to_datetime("2020-12-01")]

# Look at the data!
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(9, 5)
ax.plot(daily_concentrates.index, daily_concentrates.solvent_butanes_ppm)

# Fit a trend line.
X = daily_concentrates.index.map(datetime.date.toordinal)
Y = daily_concentrates["solvent_butanes_ppm"].fillna(0).values
X = sm.add_constant(X)
model = sm.OLS(Y, X)
results = model.fit()
print(results.summary())

# Plot the trend line with the daily data points.
ax.plot(daily_concentrates.index, results.fittedvalues, c='r')
ax.set_ylabel('ppm')
ax.set_title("Average Butange levels in WA Concentrates", fontsize=18)
fig.autofmt_xdate()
