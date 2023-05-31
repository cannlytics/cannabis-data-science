"""
Indoor vs. Outdoor Cannabis
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 5/29/2023
Updated: 5/31/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Data Files:

    - [CA Lab Results 2023-05-30](https://cannlytics.page.link/ca-lab-results-2023-05-30)
    - [CT Lab Results 2023-05-30](https://cannlytics.page.link/ct-lab-results-2023-05-30)
    - [FL Lab Results 2023-05-30](https://cannlytics.page.link/fl-lab-results-2023-05-30)
    - [MA Lab Results 2023-05-30](https://cannlytics.page.link/ma-lab-results-2023-05-30)
    - [WA Lab Results 2023-05-30](https://cannlytics.page.link/wa-lab-results-2023-05-30)

Data Sources:

    - [Glass House Farms Strains](https://glasshousefarms.org/strains/)
    - [MCR Labs Test Results](https://reports.mcrlabs.com)

References:

    - https://towardsdatascience.com/support-vector-regression-svr-one-of-the-most-flexible-yet-robust-prediction-algorithms-4d25fbdaca60

"""
# Standard imports:
import ast

# External imports:
from cannlytics.data.coas import CoADoc
from cannlytics.utils import convert_to_numeric
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'figure.figsize': (6, 4),
    'font.family': 'Times New Roman',
    'font.size': 18,
})


#-----------------------------------------------------------------------
# Get latest lab results.
#-----------------------------------------------------------------------

DATAFILES = {
    'ca': 'data/ca-lab-results-2023-05-30.xlsx',
    'ma': 'data/ma-lab-results-2023-05-30.xlsx',
}

# Read in CA data.
ca_results = pd.read_excel(DATAFILES['ca'])
ca_results['state'] = 'ca'

# Read in MA data.
ma_results = pd.read_excel(DATAFILES['ma'])
ma_results['state'] = 'ma'


#-----------------------------------------------------------------------
# Standardize the data.
#-----------------------------------------------------------------------

def get_result_value(
        results,
        analyte,
        key='key',
        value='value',
        method='list',
    ):
    """Get the value for an analyte from a list of standardized results."""
    # Ensure that the results are a list.
    try:
        result_list = json.loads(results)
    except:
        try:
            result_list = ast.literal_eval(results)
        except:
            result_list = []
    if not isinstance(result_list, list):
        return None

    # DataFrame method.
    if method == 'df':
        result_data = pd.DataFrame(result_list)
        if result_data.empty:
            return None
        result = result_data.loc[result_data[key] == analyte, value]
        try:
            return convert_to_numeric(result, strip=True)
        except:
            return result

    # List method.
    for result in result_list:
        if result[key] == analyte:
            try:
                return convert_to_numeric(result[value], strip=True)
            except:
                return result[value]


# Get terpenes of interest from the results.
ma_results['beta_pinene'] = ma_results['results'].apply(
    lambda x: get_result_value(x, 'beta_pinene')
)
ma_results['d_limonene'] = ma_results['results'].apply(
    lambda x: get_result_value(x, 'd_limonene')
)

# Convert to numeric.
ma_results['beta_pinene'] = pd.to_numeric(ma_results['beta_pinene'], errors='coerce')
ma_results['d_limonene'] = pd.to_numeric(ma_results['d_limonene'], errors='coerce')


#-----------------------------------------------------------------------
# Find matching strains in the data.
#-----------------------------------------------------------------------

# Find all well-defined strains.
known_strains = list(ca_results['strain_name'].unique())

# Match a strain.
STRAIN = 'GG4'
ca_strain = ca_results.loc[ca_results['strain_name'].str.contains(STRAIN, case=False)]
ma_strain = ma_results.loc[ma_results['product_name'].str.contains(STRAIN, case=False)]

# Plot differences.
ca_strain['beta_pinene'].hist(bins=25)
ma_strain['beta_pinene'].hist(bins=25)
plt.show()


#-----------------------------------------------------------------------
# Look at the data.
#-----------------------------------------------------------------------

def sample_and_plot(results, x, y, label='product_name', key=''):
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
    plt.savefig(f'figures/{key}-{x}-{y}.png', dpi=300, bbox_inches='tight')
    plt.show()


# Calculate the average beta-pinene and d-limonene ratios of strains.
group = ca_results.groupby('strain_name', as_index=False)
group = group[['beta_pinene', 'd_limonene', 'sativa_percentage']].mean()
group['sativa_indica_ratio'] = group['beta_pinene'] / group['d_limonene']
ratio = group.sort_values(by='sativa_indica_ratio', ascending=False)
ratio[['sativa_indica_ratio', 'sativa_percentage']]

# Plot a sample of strains.
sample = group.sample(25, random_state=420)
sample_and_plot(sample, 'd_limonene', 'beta_pinene', label=None, key='ca-strains')


#-----------------------------------------------------------------------
# Compare strain terpenes in CA to MA.
#-----------------------------------------------------------------------

# Get all well-defined strains.
strain_names = list(ca_results['strain_name'].unique())

# TODO: Compare terpene and cannabinoid concentrations.
ma_strains = []
for strain_name in strain_names:
    ma_strain = ma_results.loc[ma_results['product_name'].str.contains(strain_name, case=False)]
    ma_strains.append({
        'strain_name': strain_name,
        'beta_pinene': ma_strain['beta_pinene'].mean(),
        'd_limonene': ma_strain['d_limonene'].mean(),
    })

# Find all matching MA strains with terpene data.
ma_strains = pd.DataFrame(ma_strains)
ma_strains = ma_strains[~ma_strains['beta_pinene'].isna()]

# View MA strains.
sample_and_plot(ma_strains, 'd_limonene', 'beta_pinene', label='strain_name', key='ca-strains')

# TODO: Compare terpene and cannabinoid ratios.
comparison = pd.merge(
    group,
    ma_strains,
    on='strain_name',
    suffixes=('_ca', '_ma')
)

# Visualize the differences in beta-pinene.
comparison.plot(
    x='strain_name',
    y=['beta_pinene_ca', 'beta_pinene_ma'],
    kind='bar',
    rot=90,
    color = ['#009688', '#9C1E19']
)
plt.show()
print(comparison[['beta_pinene_ca', 'beta_pinene_ma']].mean())

# Visualize the differences in beta-pinene.
comparison.plot(
    x='strain_name',
    y=['d_limonene_ca', 'd_limonene_ma'],
    kind='bar',
    rot=90,
    color = ['#009688', '#9C1E19']
)
plt.show()
print(comparison[['d_limonene_ca', 'd_limonene_ma']].mean())


#-----------------------------------------------------------------------
# Future work: Model the data.
#-----------------------------------------------------------------------

# Build a statistical model to predict terpene concentrations,
# Y, given the strain and if it is grown indoors or outdoors, X.
from dotenv import dotenv_values
import numpy as np
import statsmodels.api as sm


# Define the sample.
sample = ca_results.loc[
    (ca_results['beta_pinene'].notna()) &
    (ca_results['d_limonene'].notna())
]
sample['ratio'] = sample['beta_pinene'].div(sample['d_limonene']).replace(np.inf, 0)

# TODO: Fit the model.
# X = sm.add_constant(sample[['ratio']])
X = sm.add_constant(sample[['beta_pinene', 'd_limonene']])
Y = sample['sativa_percentage']

# Fit the OLS model
model = sm.OLS(Y, X)
results = model.fit()
print(results.summary())


#-----------------------------------------------------------------------
# TODO: Fit a model of beta-pinene to d-limonene ratio as a predictor
# to sativa to indica dominance in an ordered probit model.
#-----------------------------------------------------------------------

from statsmodels.miscmodels.ordinal_model import OrderedModel

# Define the scale.
scale = {
    'Indica': 1,
    'Indica Dominant': 2,
    'Hybrid': 3,
    'CBD': None,
    'Sativa Dominant': 4,
    'Sativa': 5,
}
reversed_scale = {value: key for key, value in scale.items()}

# Apply the scale to strain type.
sample['scale'] = sample['strain_type'].map(scale)
sample = sample.dropna(subset=['scale'])

# Fit the ordered probit model
model = OrderedModel(
    sample['scale'],
    sample[['beta_pinene', 'd_limonene']],
    distr='probit'
)

fitted_model = model.fit(method='bfgs')
print(fitted_model.summary())

def match_strain_type(x, df):
    """Match the strain type."""
    return df.loc[df['strain_name'] == x, 'strain_type'].iloc[0]

group['strain_type'] = group['strain_name'].apply(lambda x: match_strain_type(x, ca_results))

# See the accuracy in the sample.
y_hat = fitted_model.predict(group[['beta_pinene', 'd_limonene']])
ca_results.loc[ca_results['strain_name'].isin(group['strain_name']), 'strain_type']
group['prediction'] = y_hat.idxmax(axis=1).map(reversed_scale)
correct = group.loc[group['strain_type'] == group['prediction']]
print('Accuracy:', len(correct) / len(group))

# Use the model for prediction.
x_hat = pd.DataFrame([
    {'beta_pinene': 0.01, 'd_limonene': 0.01},
    {'beta_pinene': 0.2, 'd_limonene': 0.1},
    {'beta_pinene': 0, 'd_limonene': 0.2},
])
y_hat = fitted_model.predict(x_hat)
prediction = y_hat.idxmax(axis=1).map(reversed_scale)
print('Predictions:')
print(prediction)


#-----------------------------------------------------------------------
# Optional: Strain timeseries (trending) analysis.
#-----------------------------------------------------------------------

# Plot a single strain over time.
strain = 'SUPER SILVER HAZE'
strain_data = ca_results.loc[ca_results['strain_name'] == strain]
strain_data['time'] = pd.to_datetime(strain_data['date_tested'])
strain_data.set_index('time', inplace=True)
strain_data['beta_pinene'].plot()
plt.title(f'Beta-pinene concentrations in CA {strain}')
plt.show()

# Plot a single strain over time (Super Silver Haze).
strain_data['d_limonene'].plot()
plt.title(f'D-limonene concentrations in CA {strain}')
plt.show()


#-----------------------------------------------------------------------
# Optional: Give a product description from a COA.
#-----------------------------------------------------------------------
import ast
from cannlytics.data.coas import CoADoc
import json
import openai
import zlib


# Specify a COA PDF.
doc = 'COA-SUPER-SILVER-HAZE-GHFSSH0515.pdf'

# Parse the COA with CoADoc.
parser = CoADoc()
coa_data = parser.parse(doc)

# Clean the json.
obs = coa_data[0]
del obs['sample_id']
del obs['images']
del obs['lab_image_url']
del obs['lab_results_url']
del obs['sample_hash']
del obs['results_hash']
obs['results'] = ast.literal_eval(obs['results'])
obs['results'] = [(x['name'], x['value']) for x in obs['results'] if 'analysis' in ['cannabinoid', 'terpene']]
coa_json = str(obs)
print('LEN:', len(coa_json))

# Convert the list to a JSON string.
json_data = json.dumps(obs)

# Compress the JSON string using zlib compression.
compressed_data = zlib.compress(json_data.encode())

# Print the length of the compressed data.
print("Compressed data length:", len(compressed_data))

def split_string(string, max_length):
    return [string[i:i+max_length] for i in range(0, len(string), max_length)]

# Example usage.
MAX_PROMPT_LENGTH = 3500

# Split the long string into smaller strings.
substrings = split_string(compressed_data, MAX_PROMPT_LENGTH - 100)

# Format the message.
messages = [
    {'role': 'system', 'content': 'You are a data scientist writing a description for a laboratory sample of a cannabis product.'},
]
for substring in substrings:
    content = """Part of data in bytes, wait to reply:"""
    content += str(substring)
    messages.append({'role': 'user', 'content': content})

# Close the message
messages.append({'role': 'system', 'content': 'PROMPT_COMPLETE'})

# Initialize OpenAI.
config = dotenv_values('../../.env')
openai.api_key = config['OPENAI_API_KEY']

# Alternative: Compress the PDF text to bytes.
# messages = [
#     {'role': 'system', 'content': 'You are a data scientist writing a description for a laboratory sample of a cannabis product.'},
#     {'role': 'user', 'content': 'Lab sample data in bytes: ' + str(compressed_data)},
#     {'role': 'user', 'content': 'Write a short paragraph dense with facts.'},
# ]

# Query OpenAI's GPT model.
response = openai.ChatCompletion.create(
    model='gpt-4',
    temperature=0,
    max_tokens=500,
    messages=messages
)
content = response['choices'][0]['message']['content']
print(content)

# TODO: Clean up the response.


#-----------------------------------------------------------------------
# Future work: Use LangChain to extract data from COAs.
#-----------------------------------------------------------------------

import os
from cannlytics.data.coas import CoADoc

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain import OpenAI, ConversationChain

# Define model parameters.
temperature = 0.0

# Set OpenAI credentials.
config = dotenv_values('../../.env')
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']

# Initialize LangChain.
llm = OpenAI(
    openai_api_key=config['OPENAI_API_KEY'],
    temperature=temperature,
)


def parse_coa_text(text: str):
    """Parse a COA text with `CoADoc` and OpenAI to fill in the gaps."""

    # See if the LIMS can be identified.
    parser = CoADoc()
    lims = parser.identify_lims(text)
    if lims:
        print('LIMS:', lims)
        return lims
    else:
        print('LIMS: None')
        return 'Unknown LIMS'
    
    # TODO: Parse the COA with CoADoc.

    # TODO: Turn response to JSON.


# TODO: Implement a method to read an image file, find any QR code,
# then parse the QR code with `CoADoc`.

from langchain.document_loaders.image import UnstructuredImageLoader

loader = UnstructuredImageLoader('coa-qr-code.jpg')
data = loader.load()
data[0]


# TODO: Create an LLM tool.


# TODO: Explore built-in tools:
# - llm-math (LLMMath
# - pal-math
# - requests
# - wolfram-alpha

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(["llm-math"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Now let's test it out!
agent.run("What is the average of all terpinolene compounds in GG4?")

# TODO: Have a conversation with the COA.
llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)
output = conversation.predict(input="Hi there!")
print(output)

# TODO: Talk with COA data!
# https://python.langchain.com/en/latest/modules/agents/toolkits/examples/pandas.html#pandas-dataframe-agent
# from langchain.agents import create_pandas_dataframe_agent


#-----------------------------------------------------------------------
# TODO: Add `parse_coa_with_ai` method to `CoADoc`.
#-----------------------------------------------------------------------
from typing import Optional
import pdfplumber


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
        openai_api_key: Optional[str] = None,
        model: Optional[str] = 'gpt-4',
        max_tokens: Optional[int] = 1000,
        temperature: Optional[float] = 0,
    ) -> dict:
    """Parse a COA with OpenAI's GPT model and return the data as JSON."""
    # Initialize OpenAI.
    if openai_api_key is None:
        openai_api_key = os.environ['OPENAI_API_KEY']

    # Get the text of the PDF.
    pdf = pdfplumber.open(doc)
    front_page_text = pdf.pages[0].extract_text()
    
    # Format the prompt.
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
    
    # Get the structured the data.
    try:
        start_index = content.find('{')
        end_index = content.rfind('}') + 1
        coa_data = json.loads(content[start_index:end_index])
    except:
        print('JSON parsing failed.')
        coa_data = {}
    return coa_data
