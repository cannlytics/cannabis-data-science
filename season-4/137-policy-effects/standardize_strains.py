
import json
import os
from dotenv import dotenv_values
from openai import OpenAI


def standardize_strains(
        df,
        env_file='.env',
        column='strain_name',
    ):
    """Standardize strain names."""

    # Initialize OpenAI.
    os.environ['OPENAI_API_KEY'] = dotenv_values(env_file)['OPENAI_API_KEY']
    client = OpenAI()

    # Use OpenAI to create a map of strain names.
    names = list(df[column].unique())
    prompt = """Return JSON. Given a list of strain names, return a mapping of similar strain names to a standard strain name, e.g.:
    
    {
        "Gorilla Glue #4": "Gorilla Glue",
        "Gorilla Glue": "Gorilla Glue",
        "GG4": "Gorilla Glue",
        "Wi-Fi Og 5Gram Pre-Roll": "Wi-Fi Og",
        "WiFi O.g.": "Wi-Fi Og"
    }
    
    Remove any weights, product types, or extraneous characters from strain names, e.g.:

    {
        "Og Chem [1G]": "Og Chem",
        "Og Chem ~ Og": "Og Chem",
        "Og Cookies Small Bud 1/2Oz": "Og Cookies",

    }
    
    Return strain crosses with " X ", e.g.:

    {
        "Og Chem X Tangie": "Og Chem X Tangie",
        "Og Chem/Lime Dream": "Og Chem X Lime Dream",
        "Silver Kush Bud Infused W/Durban Poison Oil": "Silver Kush X Durban Poison",
    }
    """
    # TODO: Iterate over chunks of strain names.
    prompt += '\n\Strain Names:\n'
    prompt += '\n'.join(names)
    prompt += '\n\nStrain Mapping:\n'
    messages = [{'role': 'user', 'content': prompt}]
    completion = client.chat.completions.create(
        model='gpt-4-1106-preview',
        messages=messages,
        max_tokens=4_096,
        temperature=0.0,
        user='cannlytics',
    )
    usage = completion.model_dump()['usage']
    cost = 0.01 / 1_000 * usage['prompt_tokens'] + 0.03 / 1_000 * usage['completion_tokens']
    content = completion.choices[0].message.content
    extracted_json = content.split('```json\n')[-1].split('\n```')[0]
    extracted_data = json.loads(extracted_json)
    print('Cost:', cost)
    print('Extracted:', extracted_data)

    # Merge DBAs with product data.
    dba = extracted_data['names']
    llc_to_dba = {llc: dba for llc, dba in zip(names, dba)}
    df['producer_dba_name'] = df['producer'].map(llc_to_dba)

