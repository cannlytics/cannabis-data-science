"""
Title | Project

Author: Keegan Skeate
Created: Mon Mar  1 23:00:35 2021

License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Description:
    

Resources:
    https://seaborn.pydata.org/tutorial/distributions.html
    https://seaborn.pydata.org/generated/seaborn.kdeplot.html
    https://python-graph-gallery.com/42-custom-linear-regression-fit-seaborn/

"""
from random import randrange
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Density plot
# tips = pd.Series([randrange(100) for x in range(0, 100)])
sns.kdeplot(
    data=tips,
    # hue="size",
    # fill=True,
    # common_norm=False,
    # palette="crest",
    # alpha=.5,
    # linewidth=0,
)

# Regression scatter plot
df = sns.load_dataset('iris')
sns.regplot(x=df["sepal_length"], y=df["sepal_width"], line_kws={"color":"r","alpha":0.7,"lw":5})
plt.show()

