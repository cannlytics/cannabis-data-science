"""
Parse CT COAs with AI
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 10/26/2023
Updated: 10/26/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# External imports:
from datetime import datetime
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib import cm
import numpy as np
import pandas as pd
import re
import seaborn as sns
from scipy import stats

# Read CT results.
ct_results = pd.read_excel('./data/ct-lab-results-2023-10-27.xlsx')
ct_results['date'] = pd.to_datetime(ct_results['date_tested'])
ct_results['month_year'] = ct_results['date'].dt.to_period('M')


# TODO: Parse a sample of CT COAs with AI.
