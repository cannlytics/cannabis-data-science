
"""
Audio to Text
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 4/27/2023
Updated: 4/28/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Command-line example:

    ```
    python tools/audio_to_text.py <AUDIO_FILE> <TRANSCRIPT_FILE>
    ```

Note:
    
    1. Requires OpenAI Python v0.27.0.
    2. Depends on ``ffmpeg` under GPLv3 license.

"""
# Standard imports:
import math
import sys
from typing import List

# External imports:
import openai
from pydub import AudioSegment


COST_PER_MINUTE = 0.006


def segment_audio(
        audio_file: str,
        mins_per_segment = 2.5,
    ) -> List[str]:
    """Split audio into files that are less than 25 MB."""
        
    # Read the audio file.
    audio = AudioSegment.from_mp3(audio_file)

    # Determine the duration of the file.
    duration_minutes = audio.duration_seconds / 60
    number_of_segments = math.ceil(duration_minutes / mins_per_segment)
    cost = round(COST_PER_MINUTE * duration_minutes, 2)
    print(f'Estimated cost: ${cost}')

    # Iterate over the file and split it into smaller chunks.
    files = []
    for i in range(number_of_segments):
        start = i * mins_per_segment * 60 * 1000
        end = (i + 1) * mins_per_segment * 60 * 1000
        segment = audio[start:end]
        segment_file = audio_file.replace('.wav', f'_{i}.wav')
        segment.export(segment_file, format='wav')
        files.append(segment_file)

    # Return the split audio files.
    return files


def audio_to_text(audio_file: str, transcript_file: str) -> None:
    """Convert an audio file to text using OpenAI's API."""

    # Segment audio into small segments.
    snippets = segment_audio(audio_file)

    # Create a transcript for each segment.
    segments = []
    for n, snippet in enumerate(snippets):
        audio = open(snippet, 'rb')
        segment = openai.Audio.transcribe('whisper-1', audio)
        segments.append(segment)
        print('Transcribed:', n + 1, 'of', len(snippets), 'segments.')
    
    # Compile each transcript into a single file.
    transcript = segments.join('\n')

    # Save the transcript to a file.
    with open(transcript_file, 'w', encoding='utf-8') as file:
        file.write(transcript)
    
    # Return the transcript.
    return transcript


# === Test ===
if __name__ == '__main__':

    # Define the audio and transcript files.
    
    # Command-line usage.
    audio_file = sys.argv[1]
    transcript_file = sys.argv[2]

    # DEV:
    episode = 'cannabis-data-science-70'
    audio_file = f'../.datasets/audio/{episode}.wav'
    transcript_file = f'../.datasets/audio/transcripts/{episode}.txt'

    # Convert the audio to text.
    transcript = audio_to_text(audio_file, transcript_file)
    words = transcript.split(' ')
    print('Created transcript with', len(words), 'words.')
