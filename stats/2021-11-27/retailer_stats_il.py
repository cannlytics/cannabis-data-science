"""
Get Retailer Statistics for Illinois
Cannabis Data Science Meetup Group
Saturday Morning Statistics
Copyright (c) 2021 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 11/17/2021
Updated: 11/27/2021
License: MIT License <https://opensource.org/licenses/MIT>

Data Sources:
    
    - Licensed Adult Use Cannabis Dispensaries
    <https://www.idfpr.com/LicenseLookup/AdultUseDispensaries.pdf>

    - Illinois adult use cannabis monthly sales figures
    <https://www.idfpr.com/Forms/AUC/2021%2011%2002%20IDFPR%20monthly%20adult%20use%20cannabis%20sales.pdf>
        

References:
    

"""

# External imports.
from dotenv import dotenv_values
from fredapi import Fred
import matplotlib.pyplot as plt
import pandas as pd
import PyPDF2
import requests

# Internal imports.
from utils import end_of_period_timeseries, format_millions


def clean_text(x, replacements=[]):
    """Remove certain characters from a given string, x,
    returning the clean value.
    Args:
        x (str): The value to clean.
        replacements (list): A list of text replacements to make.
    Returns:
        (str): The clean value.
    """
    for replacement in replacements:
        x = x.replace(replacement[0], replacement[1])
    return x.strip()

#-----------------------------------------------------------------------------
# Download the data.
#-----------------------------------------------------------------------------

# # Download the licensees PDF.
filename = './data/illinois_retailers.pdf'
# url = 'https://www.idfpr.com/LicenseLookup/AdultUseDispensaries.pdf'
# response = requests.get(url)
# with open(filename, 'wb') as f:
#     f.write(response.content)


#-----------------------------------------------------------------------------
# Automated cleaning of the data!
#-----------------------------------------------------------------------------

# Data to collect.
licensees = {}

# The fileds in the order that they appear. The order matters.
fields = [
    'organization',
    'trade_name',
    'street',
    'city',
    'state',
    'zip_code',
    'phone_number',
    'medical',
    'license_issue_date',
    'license_number', 
]
key = 'license_number'

# Open the PDF to get all of the text.
licensees_pdf_file = open(filename, 'rb')
pdfReader = PyPDF2.PdfFileReader(licensees_pdf_file)

# Extract the text from each page.
for i in range(pdfReader.numPages):
    
    # Get the page's text.
    pageObj = pdfReader.getPage(i)
    text = pageObj.extractText()

    # Remove title text and column names on the first page.
    try:
        text = text.split('\n \n \n \n')[1]
        text = text.split('\nCredential Number\n \n')[1]
    except IndexError:
        pass

    # Replace nuisance text.
    replacements = [
        (', Illinois ', '\n\nIL\n\n'), # Handle state abbriviations.
        (', IL ', '\n\nIL\n\n'),
        (', IL', '\n\nIL\n\n'),
        ('IL. ', '\n\nIL\n\n'),
        (')\n', ') '), # Handle split phone numbers
        ('\n \nN\n.\n \n', ' N. '), # Handle certain street addresses.
        ('Ste. C ', 'Ste. C\n'),
        (' \n \nSte.', ' Ste.'),
        ('\n \nSt.', 'St.'),
        ('Street \n', 'Street\n\n'),
        ('Ave.\n \n ', 'Ave. \n\n'),
        ('Zen\n \nL', 'Zen L'),
        ('L\ne\naf\n \n', 'Leaf '),
        ('Chicago, ', 'Chicago\n\n'),
        ('\n \n', '\n\n'), # Split lines.
        ('\n\nLLC', ' LLC'),
        ("’", "'"),
    ]
    text = clean_text(text, replacements)

    # Split text into observations.
    observations = text.split('\n\n')

    # Clean each observation.
    replacements = [
        (' \n', ' '),
        ('\n-\n' , '-'),
        ('\n*', ''),
        ('\n', ''),
        ('  ', ' '),
    ]
    observations = [clean_text(x, replacements) for x in observations]

    # Remove empty lines and the footer on the first page.
    observations = [x for x in observations if x]
    try:
        observations.remove('IDFPR -')
        observations.remove('LICENSED ADULT USE CANNABIS DISPENSARIES')
    except ValueError:
        pass

    # Identify each licensee's data.
    for n in range(0, len(observations), len(fields)):
        observation = observations[n:n + len(fields)]

        # Record the observation's fields.
        entry = {}
        for index, field in enumerate(fields):
            entry[field] = observation[index]
        licensees[entry[key]] = entry  
    
# Close the file
licensees_pdf_file.close()

# Turn licensees to dataframe and save as an Excel workbook.
licensees_data = pd.DataFrame.from_dict(licensees, orient='index')
licensees_data.to_excel('./data/licensees_il.xlsx')

#----------------

# Replace 'Yes ' with 'Yes\n'
# Replace 'No ' with 'No\n'



# Replace '-AUDO' with '-AUDO\n'
# Replace '\n-AUDO' with '-AUDO'

# Replace ', LLC ' with ', LLC\n'
# Replace '\nLLC' with ' LLC'

# Replace 'Compassionate\nCare' with 'Compassionate Care'

# Replace 'Sunnyside*' with 'Sunnyside'

# Replace '2019 ' with '2019\n'
# Replace '2020 ' with '2020\n'
# Replace '2021 ' with '2021\n'

# Delete 'IDFPR - LICENSED ADULT USE CANNABIS DISPENSARIES\n'

# Replace '\n-' with '-'

# Replace ',\nLLC' with ', LLC'

# Replace 'Tel: (TBD)' with 'None'
# Replace ' \(' with '\n\('

# Replace '284\n' with '284'

# FIXME:
    # Handle '2nd Site'
    # Handle 'Inc.'
    # 'IL.' with no space.
    # ' IL '

#-----------------------------------------------------------------------------
# Parse the licensees data.
#-----------------------------------------------------------------------------

# # Specify the fields.


# # Parse the licensee data from the text file.
# licensees = {}
# with open('./data/illinois_retailers.txt', 'r') as f:
#     text = f.read()
#     observations = text.split('\n\n')
#     for obs in observations:
#         values = obs.split('\n')
#         entry = {}
#         for index, field in enumerate(fields):
#             entry[field] = values[index]
#         licensees[entry['license_number']] = entry

# # Turn licensees to dataframe and save as an Excel workbook.
# licensees_data = pd.DataFrame.from_dict(licensees, orient='index')
# licensees_data.to_excel('./data/illinois_retailers.xlsx')

#-----------------------------------------------------------------------------
# Calculate Illinois retailer statistics.
#-----------------------------------------------------------------------------

# Read in the sales data.
# production = pd.read_excel('./data/retailer_data_il.xlsx')
# production.index = pd.to_datetime(production.month)

# # Get the Illinois population data.
# config = dotenv_values('../.env')
# fred_api_key = config.get('FRED_API_KEY')
# fred = Fred(api_key=fred_api_key)
# observation_start = production.index.min().isoformat()
# population = fred.get_series('ILPOP', observation_start=observation_start)
# population = end_of_period_timeseries(population, 'Y')
# population = population.multiply(1000) # thousands of people
# new_row = pd.DataFrame([population[-1]], index=[pd.to_datetime('2021-12-31')])
# population = pd.concat([population, pd.DataFrame(new_row)], ignore_index=False)

# # Format the license_issue_date.
# licensees_data['issue_date'] = pd.to_datetime(licensees_data['license_issue_date'])

# # Create total retailers by month series.
# production['total_retailers'] = 0
# for index, _ in production.iterrows():
#     production.at[index, 'total_retailers'] = len(licensees_data.loc[
#         (licensees_data['issue_date'] <= index)
#     ])


# # Calculate retailers per capita.
# monthly_retailers = end_of_period_timeseries(production['total_retailers'])
# monthly_population = population[0].resample('M').mean().pad()
# retailers_per_capita = monthly_retailers / (monthly_population.iloc[0] / 100_000)
# retailers_per_capita.plot()
# plt.show()

# # Calculate sales per retailer.
# total_sales = end_of_period_timeseries(production['total_sales'])
# sales_per_retailer = total_sales / production['total_retailers']
# sales_per_retailer.plot()
# plt.show()

# avg_2020_sales = total_sales.loc[
#     (total_sales.index >= pd.to_datetime('2020-01-01')) &
#     (total_sales.index < pd.to_datetime('2021-01-01'))
# ].sum()
# print('Sales per retailer in IL in 2020: %.2fM' % (avg_2020_sales / 1_000_000))


# TODO: Save retail statistics.

#--------------------------------------------------------------------------
# Estimate the relationship between dispensaries per capita and
# sales per dispensary.
#--------------------------------------------------------------------------
# import statsmodels.api as sm
# from utils import format_thousands
# from statsmodels.graphics.regressionplots import abline_plot

# # Run a regression of sales per retailer on retailers per 100,000 adults.
# Y = sales_per_retailer
# X = retailers_per_capita
# X = sm.add_constant(X)
# regression = sm.OLS(Y, X).fit()
# print(regression.summary())

# # Interpret the relationship.
# beta = regression.params.values[1]
# statement = """If retailers per 100,000 adults increases by 1,
# then everything else held constant one would expect
# revenue per retailer to change by {}.
# """.format(format_thousands(beta))
# print(statement)

# stats = pd.DataFrame([sales_per_retailer, retailers_per_capita])

# # Visualize the regression.
# ax = stats.plot(
#     x='retailers_per_100_000',
#     y='revenue_per_retailer',
#     kind='scatter'
# )
# abline_plot(
#     model_results=regression,
#     ax=ax
# )
# plt.show()










#-----------------------------------------------------------------------------
# SCRAP
#-----------------------------------------------------------------------------


# TODO: Upload licensees PDF to cloud storage.

# Extract the licensees data.
# import re

# from google.cloud import documentai_v1beta2 as documentai
# from google.cloud import storage


# # def batch_parse_table(
# #     project_id="YOUR_PROJECT_ID",
# #     input_uri="gs://cloud-samples-data/documentai/form.pdf",
# #     destination_uri="gs://your-bucket-id/path/to/save/results/",
# #     timeout=90,
# # ):
# #     """Parse a form"""

# # Set credentials.
# import os
# import environ
# env = environ.Env()
# env.read_env('../.env')
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = env('GOOGLE_APPLICATION_CREDENTIALS')

# project_id =  env('PROJECT_ID')
# storage_bucket = env('STORAGE_BUCKET')
# input_uri = f'gs://{storage_bucket}/data/state_data/illinois_retailers.pdf'
# destination_uri = f'gs://{storage_bucket}/data/state_data/parsed_'
# timeout = 90

# client = documentai.DocumentUnderstandingServiceClient()

# gcs_source = documentai.types.GcsSource(uri=input_uri)

# # mime_type can be application/pdf, image/tiff,
# # and image/gif, or application/json
# input_config = documentai.types.InputConfig(
#     gcs_source=gcs_source,
#     mime_type='application/pdf',
# )

# # where to write results
# output_config = documentai.types.OutputConfig(
#     gcs_destination=documentai.types.GcsDestination(uri=destination_uri),
#     pages_per_shard=1,  # Map one doc page to one output page
# )

# # Improve table parsing results by providing bounding boxes
# # specifying where the box appears in the document (optional)
# table_bound_hints = [
#     documentai.types.TableBoundHint(
#         page_number=1,
#         bounding_box=documentai.types.BoundingPoly(
#             # Define a polygon around tables to detect
#             # Each vertice coordinate must be a number between 0 and 1
#             normalized_vertices=[
#                 # Top left
#                 documentai.types.geometry.NormalizedVertex(x=0, y=0),
#                 # Top right
#                 documentai.types.geometry.NormalizedVertex(x=1, y=0),
#                 # Bottom right
#                 documentai.types.geometry.NormalizedVertex(x=1, y=1),
#                 # Bottom left
#                 documentai.types.geometry.NormalizedVertex(x=0, y=1),
#             ]
#         ),
#     )
# ]

# # Setting enabled=True enables form extraction
# table_extraction_params = documentai.types.TableExtractionParams(
#     enabled=True,
#     table_bound_hints=table_bound_hints,
# )

# # Location can be 'us' or 'eu'
# parent = "projects/{}/locations/us".format(project_id)
# request = documentai.types.ProcessDocumentRequest(
#     input_config=input_config,
#     output_config=output_config,
#     table_extraction_params=table_extraction_params,
# )

# api_requests = []
# api_requests.append(request)

# batch_request = documentai.types.BatchProcessDocumentsRequest(
#     parent=parent,
#     requests=api_requests,
# )

# operation = client.batch_process_documents(batch_request)

# # Wait for the operation to finish.
# operation.result(timeout)

# # Results are written to GCS. Use a regex to find output files.
# match = re.match(r"gs://([^/]+)/(.+)", destination_uri)
# output_bucket = match.group(1)
# prefix = match.group(2)

# storage_client = storage.client.Client()
# bucket = storage_client.get_bucket(output_bucket)
# blob_list = list(bucket.list_blobs(prefix=prefix))
# print('Output files:')
# for blob in blob_list:
#     print(blob.name)


# # TODO: Download created JSON.


# # Read in the JSON files.
# import json
# data_files = [
#     './data/parsed_illinois_retailers-output-page-1-to-1.json',
# ]
 
# # Opening JSON file
# f = open('./data/parsed_illinois_retailers-output-page-1-to-1.json')
# data = json.load(f)
 

# Parse the JSON.

