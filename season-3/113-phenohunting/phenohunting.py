"""
Phenohunting with OpenAI's GPT-4 Model
Copyright (c) 2023 Cannlytics

Authors:
    Keegan Skeate <https://github.com/keeganskeate>
Created: 5/24/2023
Updated: 5/24/2023
License: <https://github.com/cannlytics/cannlytics/blob/main/LICENSE>
"""
# Standard imports:
import glob
import json
import os
from typing import Optional

# External imports:
from cannlytics.data.coas import CoADoc
from dotenv import dotenv_values
import openai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdfplumber
import seaborn as sns


#-----------------------------------------------------------------------
# Setup
#-----------------------------------------------------------------------

# Initialize OpenAI.
# API Key URL: <https://platform.openai.com/account/api-keys>
config = dotenv_values('../../.env')
openai.api_key = config['OPENAI_API_KEY']

# Initialize CoADoc.
parser = CoADoc()

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (6, 4),
    'font.family': 'Times New Roman',
    'font.size': 18,
})

# Specify where your data lives.
DATA_DIR = 'D://data/florida/lab_results/.datasets/pdfs/acs'


#-----------------------------------------------------------------------
# Get the COA data.
#-----------------------------------------------------------------------

import os
import requests
from time import sleep

# Read all COA URLs.
filename = 'florida-coa-urls.txt'
with open(filename, 'r') as file:
    urls = [line.strip() for line in file.readlines()]

# Download COA PDFs.
for i, url in enumerate(urls):
    sleep(1)
    response = requests.get(url)
    outfile = os.path.join(DATA_DIR, f'coa-{i}.pdf')
    with open(outfile, 'wb') as pdf:
        pdf.write(response.content)


#-----------------------------------------------------------------------
# Prompt engineering to extract data from COAs.
#-----------------------------------------------------------------------

# Read the text of a COA.
doc = os.path.join(DATA_DIR, 'coa-1.pdf')
pdf = pdfplumber.open(doc)
front_page_text = pdf.pages[0].extract_text()

# Get the QR code URL.
qr_code_url = parser.find_pdf_qr_code_url(pdf)

# Create prompt from front page text.
prompt = """Extract as many of these data points to JSON:

{
    "product_name": str,
    "product_type": str,
    "producer": str,
    "total_thc_percent": float,
    "total_cbd_percent": float,
    "beta_pinene_percent": float,
    "d_limonene_percent": float,
    "farnesene_percent": float,
}

From the following text: 
"""
prompt += front_page_text

# Query OpenAI's GPT model to extract data.
response = openai.ChatCompletion.create(
  model='gpt-4',
  temperature=0,
  max_tokens=1000,
  messages=[
        {'role': 'system', 'content': 'Return only JSON and always return at least an empty object if no data can be found.'},
        {'role': 'user', 'content': prompt},
    ]
)
content = response['choices'][0]['message']['content']  
print(response['choices'][0]['message']['content'])

# Clean the JSON afterwards.
start_index = content.find('{')
end_index = content.rfind('}') + 1
coa_data = json.loads(content[start_index:end_index])


#-----------------------------------------------------------------------
# Extract data from a sample of COAs.
#-----------------------------------------------------------------------

# Define the engineered prompts.
COA_PROMPT = """Extract as many of these data points to JSON:
{
    "product_name": str,
    "product_type": str,
    "producer": str,
    "total_thc_percent": float,
    "total_cbd_percent": float,
    "beta_pinene_percent": float,
    "d_limonene_percent": float,
}
From the following text:
"""
SYSTEM_PROMPT = 'Return only JSON and always return at least an empty object if no data can be found.'


def parse_coa_with_ai(
        doc: str,
        model: Optional[str] = 'gpt-4',
        max_tokens: Optional[int] = 1000,
        temperature: Optional[float] = 0,
    ) -> dict:
    """Parse a COA with OpenAI's GPT model and return the data as JSON."""
    pdf = pdfplumber.open(doc)
    front_page_text = pdf.pages[0].extract_text()
    prompt = COA_PROMPT
    prompt += front_page_text
    try:
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response['choices'][0]['message']['content'] 
        print('CONTENT:', content)
    except:
        print('AI query failed.')
        return {}
    try:
        start_index = content.find('{')
        end_index = content.rfind('}') + 1
        coa_data = json.loads(content[start_index:end_index])
    except:
        print('JSON parsing failed.')
        coa_data = {}
    return coa_data


# Get a sample of COA PDFs.
pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))

# Parse each COA PDF with AI.
parsed_coas = []
for pdf_file in pdf_files[:5]:
    coa_data = parse_coa_with_ai(pdf_file)
    coa_data['filename'] = pdf_file
    parsed_coas.append(coa_data)

# Format the data as a DataFrame.
coa_df = pd.DataFrame(parsed_coas)

# Remove any COAs that could not be parsed.
coa_df.dropna(inplace=True)


#-----------------------------------------------------------------------
# Visualize the COA data.
#-----------------------------------------------------------------------

def sample_scatterplot(results, x, y, label=None, outfile=None):
    """Sample results with valid values and plot."""
    sample = results.loc[(results[x].notna()) & (results[y].notna())]
    sample = sample.replace('', np.nan).dropna(subset=[x, y])
    sample['ratio'] = sample[y].div(sample[x]).replace(np.inf, 0)
    sns.scatterplot(
        x=x,
        y=y,
        data=sample,
        hue='ratio',
        size='ratio',
        sizes=(150, 400),
        legend='full',
        palette='viridis',
    )
    if label:
        for line in range(0, sample.shape[0]):
            plt.text(
                sample[x].iloc[line],
                sample[y].iloc[line],
                sample[label].iloc[line],
                horizontalalignment='left',
                size='medium',
            )
    plt.xlim(0)
    plt.ylim(0)
    plt.legend([],[], frameon=False)
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show()

# Visualize the results of the sample.
sample_scatterplot(
    coa_df,
    x='d_limonene_percent',
    y='beta_pinene_percent',
    outfile='predicted-fl-limonene-to-pinene.png',
)

# Visualize the results for a particular strain.
strain = 'Khalifa'
strain_data = coa_df.loc[coa_df['product_name'].str.contains(strain)]
sample_scatterplot(
    strain_data.sample(10, random_state=420),
    x='d_limonene_percent',
    y='beta_pinene_percent',
    label='product_name',
    outfile='predicted-khalifa-kush-limonene-to-pinene.png',
)
