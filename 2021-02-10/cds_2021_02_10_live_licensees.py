# -*- coding: utf-8 -*-
"""
Title | Project

Author: Keegan Skeate <keeganskeate@gmail.com>
Created: Wed Feb 10 09:09:08 2021

License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Description:
    

Resources:

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
file_name = "./Licensees_0/Licensees_0.csv"
data = pd.read_csv(
    file_name,
    encoding="utf-16",
    sep='\t',
)
print("Number of observations:", len(data))

active_licenses = data.loc[(data.suspended == False) &
                           (data.is_live == True)]

active_licenses.to_excel("active_licenses.xlsx")


