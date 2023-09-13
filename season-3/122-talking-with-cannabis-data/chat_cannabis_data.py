"""
Talk with a Cannabis Data Science Transcript
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 7/26/2023
Updated: 8/2/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# Standard imports:
import os

# External imports:
from cannlytics.ai import split_into_token_chunks
from dotenv import dotenv_values
import openai


# Initialize the OpenAi API.
config = dotenv_values('../../.env')
openai.api_key = config['OPENAI_API_KEY']
model = 'gpt-3.5-turbo'
temperature = 0.42
max_tokens = 1_000


# Read all of the transcripts.
video_dir = './transcripts'
text_data = []
for file in os.listdir(video_dir):
    with open(os.path.join(video_dir, file), 'r') as f:
        text_data.append(f.read())

# Format the prompt.
messages = []

# Add the system prompt.
messages.append({
    "role": "system",
    "content": "Let me tell you about a transcript. At the end, please summarize.",
})

# Add the transcripts from the videos.
for text in text_data:
    chunks = split_into_token_chunks(
        text,
        max_prompt_length=max_tokens
    )
    for chunk in chunks:
        messages.append({
            "role": "user",
            "content": "Transcript: " + chunk
        })

# Add summary prompt.
messages.append({
    "role": "user",
    "content": "Summary:"
})

# Talk with ChatGPT to get insights from the videos.
response = openai.ChatCompletion.create(
    model=model,
    temperature=temperature,
    max_tokens=max_tokens,
    messages=messages[:5]
)
content = response['choices'][0]['message']['content']  
print(content)
