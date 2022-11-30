"""
Retail Analysis with Panel Data | Saturday Morning Statistics
Copyright (c) 2021 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 11/27/2021
Updated: 11/27/2021
License: MIT License <https://opensource.org/licenses/MIT>

References:
    
    - Fixed Effect Regression — Simply Explained
    https://towardsdatascience.com/fixed-effect-regression-simply-explained-ab690bd885cf
    
    - Three ways to run Linear Mixed Effects Models inPython Jupyter Notebooks
    https://towardsdatascience.com/how-to-run-linear-mixed-effects-models-in-python-jupyter-notebooks-4f8079c4b589
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

#-----------------------------------------------------------------------------
# Read in panel data.
#-----------------------------------------------------------------------------

# load the data
df = pd.read_csv("Fatalities.csv")


#-----------------------------------------------------------------------------
# Estimate a fixed effects model.
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# Estimate a random effects model.
#-----------------------------------------------------------------------------

# Run LMER
# md = smf.mixedlm(
#     "Weight ~ Time"
#     data,
#     groups=data["Pig"],
#     re_formula="~Time"
# )
# mdf = md.fit(method=["lbfgs"])
# print(mdf.summary())

# md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"], re_formula="~Time")
# mdf = md.fit(method=["lbfgs"])
# print(mdf.summary())

