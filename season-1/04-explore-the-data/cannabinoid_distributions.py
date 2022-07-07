"""
Cannabinoid Distributions | Cannabis Data Science

Author: Keegan Skeate
Created: Wednesday, March 3, 2021

License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Description:
    Create cannabinoid distributions for different sample types.

Resources:
    https://lcb.wa.gov/sites/default/files/publications/Marijuana/traceability/WALeafDataSystems_UserManual_v1.37.5_AddendumC_LicenseeUser.pdf
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#-----------------------------------------------------------------------------#
# Read in lab results from the .csv file
# obtained through public records request.
# https://lcb.app.box.com/s/fnku9nr22dhx04f6o646xv6ad6fswfy9?page=1
# https://stackoverflow.com/questions/18171739/unicodedecodeerror-when-reading-csv-file-in-pandas-with-python
# https://stackoverflow.com/questions/18039057/python-pandas-error-tokenizing-data
#-----------------------------------------------------------------------------#
file_name = directory + "/LabResults_0.csv"
data = pd.read_csv(
    file_name,
    encoding="utf-16",
    sep='\t',
)
print("Number of observations:", len(data))


# Plot a histogram.
sample = data.sample(1000)
analyte = "cannabinoid_d9_thca_percent"
# sns.displot(data, x="cannabinoid_d9_thca_percent")
sns.kdeplot(
    data=sample[analyte],
    hue="intermediate_type",
    # fill=True,
    # common_norm=False,
    # palette="crest",
    # alpha=.5,
    # linewidth=0,
)


# Plot conditional historgam.