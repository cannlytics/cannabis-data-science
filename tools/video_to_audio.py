"""
Process Cannabis Data Science Recording
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 4/4/2022
Updated: 12/13/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Command-line example:

    ```
    python tools/video_to_audio.py cannabis-statistics/cannabis-statistics-01
    ```
"""
# Standard imports:
import os
import sys
from typing import Optional

# External imports:
from dotenv import dotenv_values
import moviepy.editor as mp


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

    # Save the audio to a file.
    video_file = f'{video_dir}/videos/{episode_name}.mp4'
    audio_file = f'{video_dir}/audio/{episode_name}.mp3'
    video_to_audio(
        video_file=video_file,
        audio_file=audio_file,
        volume=1,
    )
