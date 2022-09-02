"""
CoA Parsing for Consumer Product Recommendation
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 8/23/2022
Updated: 8/31/2022
License: <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    Parse a producer's corpus of CoAs, create a standardized datafile, then
    use the data, augmented with data about consumer's prior purchases, to
    create product recommendations for each consumer.

Data Source:

    - Raw Garden Lab Results
    URL: <https://rawgarden.farm/lab-results/>

    - Strain Reviews
    https://cannlytics.page.link/reported-effects

"""
# Standard imports.
from datetime import datetime
import math
import os
from time import sleep

# External imports.
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

# Internal imports.
from cannlytics.data.coas import CoADoc
from cannlytics.utils.constants import DEFAULT_HEADERS

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.family': 'Times New Roman',
    'font.size': 24,
})

# Specify where your data lives.
DATA_DIR = '.datasets'
COA_DATA_DIR = f'{DATA_DIR}/lab_results/raw_garden'
COA_PDF_DIR = f'{COA_DATA_DIR}/pdfs'
REVIEW_DATA_DIR = f'{DATA_DIR}/effects'

# Create directories if they don't already exist.
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
if not os.path.exists(COA_DATA_DIR): os.makedirs(COA_DATA_DIR)
if not os.path.exists(COA_PDF_DIR): os.makedirs(COA_PDF_DIR)
if not os.path.exists(REVIEW_DATA_DIR): os.makedirs(REVIEW_DATA_DIR)


#-----------------------------------------------------------------------
# Get the data!
# URL: <https://rawgarden.farm/lab-results/>
#-----------------------------------------------------------------------

# Get Raw Garden's lab results page.
base = 'https://rawgarden.farm/lab-results/'
response = requests.get(base, headers=DEFAULT_HEADERS)
soup = BeautifulSoup(response.content, 'html.parser')

# Get all of the PDF URLs.
urls = []
for i, link in enumerate(soup.findAll('a')):
    try:
        href = link.get('href')
        if href.endswith('.pdf'):
            urls.append(href)
    except AttributeError:
        continue

# Download all of the PDFs.
pause = 0.24 # Pause to respect the server serving the PDFs.
total = len(urls)
print('Downloading PDFs, ETA > %.2fs' % (total * pause))
start = datetime.now()
for i, url in enumerate(urls):
    name = url.split('/')[-1]
    outfile = os.path.join(COA_PDF_DIR, name)
    response = requests.get(url, headers=DEFAULT_HEADERS)
    with open(outfile, 'wb') as pdf:
        pdf.write(response.content)
    print('Downloaded %i / %i' % (i +  1, total))
    sleep(pause)
end = datetime.now()

# Count the number of PDFs downloaded.
files = [x for x in os.listdir(COA_PDF_DIR)]
print('Downloaded %i PDFs.' % len(files), 'Time:', end - start)

# TODO: Organize the PDFs into folder by type.


#-----------------------------------------------------------------------
# Parse and standardize the data with CoADoc
#-----------------------------------------------------------------------

# Parse lab results with CoADoc.
parser = CoADoc()

# Future work: Parse all CoAs in 1-shot.
# data = parser.parse(COA_PDF_DIR)

# Iterate over PDF directory.
all_data, recorded, unidentified = [], [], []
for path, subdirs, files in os.walk(COA_PDF_DIR):
    for name in files:
        if not name.endswith('.pdf'):
            continue
        if name in unidentified:
            continue

        # Parse CoA PDFs one by one.
        file_name = os.path.join(path, name)
        try:
            coa_data = parser.parse(file_name, max_delay=3)
            all_data.extend(coa_data)
            print('Parsed:', name)
            recorded.append(name)
        except:
            print('Error:', name)
            pass

        # Save the CoA data every 100 CoAs, just in case!
        if len(all_data) % 100 == 0:
            timestamp = datetime.now().isoformat()[:19].replace(':', '-')
            outfile = f'{COA_DATA_DIR}/rawgarden-coa-data-{timestamp}.xlsx'
            coa_data = parser.save(all_data, outfile)

# Save all of the CoA data.
timestamp = datetime.now().isoformat()[:19].replace(':', '-')
outfile = f'{COA_DATA_DIR}/rawgarden-coa-data-{timestamp}.xlsx'
coa_data = parser.save(all_data, outfile)

# DEV:
outfile = f'{COA_DATA_DIR}/rawgarden-coa-data-2022-08-31T14-05-09.xlsx'

# Read the CoA data back in.
coa_values = pd.read_excel(outfile, sheet_name='Values')

# Visualize the beta-pinene / d-limonene ratio of the CoA data.
coa_values['pine_lime_ratio'] = coa_values.eval('beta_pinene / d_limonene')
coa_values['pine_lime_ratio'].hist(bins=100)
plt.show()

# Visualize the log of the beta-pinene / d-limonene ratio.
coa_values['log_pine_lime_ratio'] = coa_values['pine_lime_ratio'].apply(np.log)
coa_values['log_pine_lime_ratio'].hist(bins=100)
plt.show()


#-----------------------------------------------------------------------
# Read in supplementary data: consumer reviews.
#-----------------------------------------------------------------------

# Read in the reviews.
datafile = f'{REVIEW_DATA_DIR}/strain-reviews-2022-06-15.xlsx'
reviews = pd.read_excel(datafile, index_col=0)

# Remove duplicates.
reviews.drop_duplicates(subset='review', inplace=True)

# TODO: Standardize review columns?

# Determine the unique number of users.
users = reviews.groupby('user', as_index=False)
user_averages = users.mean()
user_averages['n_reviews'] = users['review'].nunique()['review']

# Isolate a training sample.
sample = user_averages.loc[
    (user_averages['user'] != 'Anonymous') &
    (user_averages['beta_pinene'] > 0) &
    (user_averages['d_limonene'] > 0) &
    (user_averages['n_reviews'] > 10)
    # (train['total_thc'] > 0) &
    # (train['total_cbd'] > 0) &
    # (train['beta_caryophyllene'] > 0) &
    # (train['humulene'] > 0)
]

# Plot beta-pinene to d-limonene by user.
x, y = 'd_limonene', 'beta_pinene'
ax = sns.scatterplot(
    x=x,
    y=y,
    data=sample,
    hue='n_reviews',
    size='n_reviews',
    sizes=(100, 1000),
)
for index, row in sample.iterrows():
    if row['n_reviews'] < 30:
        continue
    ax.text(
        row[x],
        row[y],
        row['user'],
        horizontalalignment='center',
        verticalalignment='bottom',
        size='medium',
        color='#2A8936',
    )
plt.xlim(0)
plt.ylim(0)
plt.title('Average beta-Pinene to d-Limonene by User')
plt.show()

# Save the ratio for future use.
ratio = sample.eval(f'{y} / {x}').rename('ratio')
sample['pine_lime_ratio'] = ratio

# Visualize the distribution of ratios.
sample['pine_lime_ratio'].hist(bins=50)
plt.title('Distribution of Average beta-Pinene to d-Limonene by User')
plt.show()

# Visualize the distribution after a log transformation.
sample['pine_lime_ratio'].apply(np.log).hist(bins=50)
plt.title('Distribution of Log of Average beta-Pinene to d-Limonene by User')
plt.show()

sample['log_pine_lime_ratio'] = sample['pine_lime_ratio'].apply(np.log)


#-----------------------------------------------------------------------
# Analyze the data.
#-----------------------------------------------------------------------

# TODO: Recommend the nearest product in similarity of terpene ratios.

# Define the features to use.
features = ['log_pine_lime_ratio']
X = coa_values[features].dropna(how='all')

# Fit a nearest neighbors model.
model = NearestNeighbors(n_neighbors=4, algorithm='ball_tree')
model.fit(X)

# Specify the user.
x_hat = sample.sample(1, random_state=420)

# Predict the k-nearest neighbors given a user's chemotype profile.
distance, prediction = model.kneighbors(x_hat[['log_pine_lime_ratio']])
y_hat = X.iloc[prediction[0]]

# TODO: Map prediction to products.
recommendations = coa_values.loc[y_hat.index]
# recommendations = list(coa_values.index)
print('Recommendations:', recommendations['product_name'])

# Concat historic user average with recommendations.
observations = pd.concat([x_hat, recommendations])

# Look terpene ratio of recommended products in
# comparison of the user's historic ratio.
x, y = 'd_limonene', 'beta_pinene'
ax = sns.scatterplot(
    data=observations,
    x=x,
    y=y,
    s=400,
    palette='viridis_r',
)
for index, row in observations.iterrows():
    name = row['user']
    try:
        if math.isnan(name): name = row['product_name']
    except TypeError:
        continue
    ax.text(
        row[x],
        row[y],
        name,
        horizontalalignment='center',
        verticalalignment='bottom',
        size='medium',
        color='#2A8936',
    )
plt.xlim(0)
plt.ylim(0)
plt.title('beta-Pinene / d-Limonene Ratio of Recommendations to Historic Average')
plt.show()


#-----------------------------------------------------------------------
# Bonus: k-nearest neighbors Search with user-specified strain.
#-----------------------------------------------------------------------


# E.g. My favorite strain is Jack Herer, the average Jack Herer has
# chemical profile Xj. Therefore, the most similar raw Garden product
# based on factors B, is Yj.


# E.g. Mikki's favorite strain is Northern Lights. The average Northern Lights
# has chemical profile Xn. Therefore, the most similar raw Garden product
# based on factors B, is Yn.


# Super bonus: If it were possible to mix products, then what mix
# of products would match, or be the closest match, to a consumer's
# historic or desired chemical profile.

# def f(alpha, x, target):
#     matrix = x[0] * alpha + x[1] * (1- alpha) - target
#     return sum(matrix)

# def gradient_descent(gradient, start, learn_rate, n_iter, **kwargs):
#     vector = start
#     for _ in range(n_iter):
#         diff = -learn_rate * gradient(vector, **kwargs)
#         vector += diff
#     return vector

# # Get a user's target.
# strain = 'Jack Herer' # vs. Northern Lights
# x_hat = pd.DataFrame([strains.loc[strain]])

# # Predict an appropriate blend of two products.
# alpha = 0.5
# product1 = y_hat.iloc[0]
# product2 = y_hat.iloc[1]
# optimal_mix = gradient_descent(
#     gradient=f,
#     start=alpha,
#     learn_rate=0.01,
#     n_iter=100,
#     x=[product1, product2, alpha],
#     target=x_hat,
# )
