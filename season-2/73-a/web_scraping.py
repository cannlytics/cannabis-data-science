"""
Web Scraping | Cannabis Data Science #73 | 2022-07-06
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: July 4th, 2022
Updated: 7/4/2022
License: MIT License <https://opensource.org/licenses/MIT>

Description:

    Archive all of the PSI Labs test results.

"""
# Standard imports.
import datetime

# External imports.
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import pandas as pd
import  requests
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import abline_plot


BASE = 'https://results.psilabs.org/test-results/?page={}'
PAGES = 4921
print('Expected treasure:', PAGES * 10, 'observations!')


#------------------------------------------------------------------------------
# Getting the data.
#------------------------------------------------------------------------------



# Optional: Pull Michigan state limits? Or else use static.
pages = range(1, PAGES)
for page in pages:

    url = BASE.format(page)
    response = requests.get(url))
    soup = BeautifulSoup(r.content)
    print(soup.find("table",{"class": "gf-table historical_price"}).get_text())

    break

    # except Exception as e:
    #     print(e)
    #     break
    # start += 30

# Get:
# - sample_id (generated)
# - sample_name
# - sample_type
# - producer
# - lab_results_url
# - analyses
# - images
# - coa_urls
# - QR code
# - date_received
# - date_tested
# - method
# - results
#    * units
#    * value
#    * margin_of_error
#    * name
#    * key
#    * cas (augmented)
#    * limit (augmented)


# TODO: Count the number of lab results scraped!


# TODO: Count the number of unique data points scraped!


#-----------------------------------------------------------------------
# Preprocessing the Data
#-----------------------------------------------------------------------

# TODO: Create a data NFT for the lab results!!!



#------------------------------------------------------------------------------
# Exploring the data.
#------------------------------------------------------------------------------

# Look at cannabinoid concentrations over time.


# Look at cannabinoid distributions by type.


# Look at terpene distributions by type!


#-----------------------------------------------------------------------
# Modeling the Data
#-----------------------------------------------------------------------

# Given a lab result, predict if it's in the Xth percentile.

