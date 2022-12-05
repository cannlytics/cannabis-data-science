"""
Process Cannabis Data Science Recording
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 4/4/2022
Updated: 12/4/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Command-line example:

    ```
    python tools/process_recording.py <episode-number> <optional:volume>
    ```

Setup:

    ```
    pip install moviepy
    pip install git+https://github.com/openai/whisper.git
    pip install --upgrade git+https://github.com/huggingface/diffusers.git
    ```

"""
# Standard imports:
import os
import sys
from typing import Optional

# External imports:
from cannlytics.utils import snake_case
from datasets import load_dataset
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from dotenv import dotenv_values
import moviepy.editor as mp
import pandas as pd
import requests
import torch
import whisper


def video_to_audio(
        video_file: str,
        audio_file: str,
        volume: Optional[float] = 1,
    ) -> None:
    """Save the audio from a video as an audio file, incrementing the version
    as needed.
    Args:
        directory (str): The directory with `videos` and `audio` folders.
        episode (str): The name of the video to extract audio from.
        volume (float): A scale for the volume (optional).
    """
    clip = mp.VideoFileClip(video_file)
    if volume != 1:
        clip = clip.volumex(volume)
    clip.audio.write_audiofile(audio_file)

        
def audio_to_text(
        audio: str,
        model_name: Optional[str] = 'small',
    ) -> str:
    """Create text transcript from audio file.
    Args:
        An audio file.
    Returns:
        The parsed text.
    """
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    model = whisper.load_model(model_name)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)
    return result.text


def text_to_image(
        prompt: str,
        filename: Optional[str] = None,
        height: Optional[int] = 768,
        width: Optional[int] = 768,
    ) -> str:
    """Create an image given a text prompt."""
    model_id = 'stabilityai/stable-diffusion-2'
    scheduler = EulerDiscreteScheduler.from_pretrained(
        model_id,
        subfolder='scheduler',
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to('cuda')
    image = pipe(prompt, height=height, width=width).images[0]
    if filename is None:
        filename = snake_case(prompt[:32] + '.png')
    image.save(filename)
    return filename


def create_hyperlink(href, text):
    """Format a hyperlink as a link with given text."""
    return f'<a href="{href}">{text}</a>'


def create_readme_index(datafile):
    """Create a markdown table from given JSON."""
    columns = {
        'short_title': 'Topic',
        'description': 'Description',
        'video_url': 'Video',
        'source_code_url': 'Code',
    }
    videos = pd.read_json(datafile)
    videos = videos.loc[videos['video_url'] != '']
    videos['description'] = videos['description'].apply(lambda x: x.split('\n')[0])
    videos['video_url'] = videos['video_url'].apply(create_hyperlink, text='Video')
    videos['source_code_url'] = videos['source_code_url'].apply(create_hyperlink, text='Code')
    videos['short_title'] = videos.apply(
        lambda x: create_hyperlink(x['source_code_url'], x['short_title']),
        axis=1,
    )
    videos = videos[list(columns.keys())]
    videos.rename(columns=columns, inplace=True)
    html = videos.to_html(index=False, render_links=True, escape=False)
    return html


def update_readme_index(videos_file, readme_file):
    """Create the table of episodes and update the `readme.md`."""
    table = create_readme_index(videos_file)
    with open(readme_file, 'r') as f:
        text = f.readlines()
        readme = ''.join([line for line in text])
        intro = readme.split('<table')[0]
        outro = readme.split('</table>')[-1]
        updated_readme = ''.join([intro, table, outro])
    with open(readme_file, 'w') as f:
        f.write(updated_readme)


# === Test ===
if __name__ == '__main__':

    # Specify the video directory.
    try:
        config = dotenv_values('../.env')
        video_dir = config['VIDEO_DIR']
    except KeyError:
        config = dotenv_values('.env')
        video_dir = config['VIDEO_DIR']

    # Get the episode.
    episode_name = sys.argv[1]

    # Get the volume if specified.
    try:
        volume = sys.argv[2]
    except IndexError:
        volume = 1

    # Save the audio to a file.
    video_file = f'{video_dir}/videos/{episode_name}.mov'
    audio_file = f'{video_dir}/audio/{episode_name}.mp3'
    video_to_audio(
        video_file=video_file,
        audio_file=audio_file,
        volume=volume,
    )

    # Get a transcript of the audio file.
    text = audio_to_text(audio_file)

    # TODO: Summarize (perhaps with BERT) to get a title and description.
    short_title = ''
    description = ''

    # TODO: Create background art (perhaps with Stable Diffusion) with the title.
    # Save the background as needed to Firebase Storage.
    cover_image_url = ''

    # Update the `readme.md`.
    # update_readme_index('../assets/videos-stats.json', '../readme.md')
