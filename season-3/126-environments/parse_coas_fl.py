"""
COA Parsing
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 9/5/2023
Updated: 9/5/2023
License: MIT License <https://github.com/cannlytics/cannlytics/blob/main/LICENSE>
"""
# Standard imports:
import glob
import os
import tempfile

# External imports:
from cannlytics.data.coas import CoADoc
from dotenv import load_dotenv
import pandas as pd


# Specify where your data lives.
data_dir = '.datasets/reddit-coas'
temp_path = tempfile.gettempdir()

# Create the output data directory before beginning.
datafile_dir = 'data/datafiles'
if not os.path.exists(datafile_dir):
    os.makedirs(datafile_dir)

# Get all COA PDFs.
all_files = glob.glob(os.path.join(data_dir, '*.*'))
pdf_files = [f for f in all_files if f.endswith('.pdf')]
non_pdf_files = [f for f in all_files if not f.endswith('.pdf')]

# Optional: Get your OpenAI API key.
# load_dotenv('../../.env')
# openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize a parser.
parser = CoADoc()
all_data = []

# FIXME:
url = 'https://portal.acslabcannabis.com/qr-coa-view?salt=QUFFRzMwMF81NTgxNjAwMDM3NzQ3MThfMDMyODIwMjNfNjQyMzU0ZWE1NDdlYg=='
coa_data = parser.parse(url, temp_path=temp_path)

# Try to get URLs from each non-PDF file.
urls = []
for f in non_pdf_files:
    try:
        url = parser.scan(f)
        if url:
            urls.append(url)
            print('Found URL:', url)
        else:
            print('No URL found:', f)
    except Exception as e:
        print('Error finding URL:', e)
        print(e)

# Parse COA URLs.
for url in urls:
    
    # First, try parsing normally.
    try:
        coa_data = parser.parse(url, temp_path=temp_path)
        all_data.extend(coa_data)
        print('Parsed:', url)

    # If normal parsing fails, try parsing with AI.
    except Exception as e:
        print('Failed to parse:', url)
        print(e)

# Parse COA PDFs.
pdf_files.reverse()
for pdf_file in pdf_files:
    
    # First, try parsing normally.
    try:
        coa_data = parser.parse(pdf_file, temp_path=temp_path)
        all_data.extend(coa_data)
        print('Parsed:', pdf_file)

        # If normal parsing fails, try parsing with AI.
    except Exception as e:
        print('Failed to parse:', pdf_file)

        # # try:
        # coa_data, prompts, cost = parser.parse_with_ai(
        #     pdf_file,
        #     openai_api_key=openai_api_key,
        #     user='cannlytics',
        #     verbose=True,
        #     max_tokens=4_000,
        #     max_prompt_length=1_000,
        #     temperature=0.0,
        # )
        # coa_data['filename'] = pdf_file
        # all_data.append(coa_data)
        # print('Parsed with AI:', pdf_file)
        # print('Prompts:', prompts)
        # print('Cost:', cost)
        # print('Data:', coa_data)
            
        # # If AI parsing also fails, append an empty row.
        # except Exception as e:
        all_data.append({'filename': pdf_file, 'error': str(e)})

# Remove error rows.
data = pd.DataFrame(all_data)
filtered_data = data[data['error'].isna()]
filtered_data = filtered_data.drop(columns=['filename', 'error'])

# Save the parsed COA data.
timestamp = pd.Timestamp.now().strftime('%Y-%m-%d-%H-%M-%S')
outfile = f'{datafile_dir}/fl-coa-data-{timestamp}.xlsx'
parser.save(filtered_data, outfile)
