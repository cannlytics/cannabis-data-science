"""
Process Cannabis Data Science Recording
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 4/4/2022
Updated: 11/30/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Command-line example:

    ```
    python tools/process_recording.py <episode-number> <optional:volume>
    ```

Setup:

    ```
    pip install moviepy
    pip install git+https://github.com/openai/whisper.git
    ```

"""
# Standard imports:
import os
import sys
from typing import Optional

# External imports:
from dotenv import dotenv_values
import moviepy.editor as mp # 
import whisper


model = whisper.load_model("small")


def video_to_audio(
        video_file: str,
        audio_file: str,
        volume: Optional[float] = 1,
):
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

        
def audio_to_text(audio):
    """Create text transcript from audio file.
    Args:
        An audio file.
    Returns:
        The parsed text.
    """
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)
    return result.text


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
    short_title, description = '', ''

    # TODO: Create background art (perhaps with Stable Diffusion) with the title.
    # Save the background as needed to Firebase Storage.
    cover_image_url = ''
