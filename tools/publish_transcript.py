"""
Publish a transcript
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 7/23/2023
Updated: 7/24/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Command-line example:

    ```
    python tools/publish_transcript.py cannabis-statistics/cannabis-statistics-05
    ```

"""
# Standard imports:
import os
import re
import sys
from typing import Dict, Tuple

# External imports:
from dotenv import dotenv_values
from cannlytics import firebase


def get_env_vars(env_file: str = ".env") -> Dict[str, str]:
    """Fetch environment variables from a .env file."""
    if not os.path.exists(env_file):
        raise FileNotFoundError(f"Cannot find {env_file}")
    config = dotenv_values(env_file)
    required_vars = [
        "VIDEO_DIR",
        "FIREBASE_STORAGE_BUCKET",
        "FIREBASE_API_KEY",
        "FIREBASE_PROJECT_ID",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ]
    for var in required_vars:
        if var not in config:
            raise KeyError(f"ERROR: {var} not found in .env file.")
    return config


def clean_transcript(file_path: str) -> str:
    """Clean a transcript file by removing timestamps and unnecessary
    numbers/elements. Removes leading/trailing whitespace."""
    with open(file_path, "r") as file:
        transcript = file.read()
    cleaned_transcript = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d+ --> \d{2}:\d{2}:\d{2},\d+\n', '', transcript)
    cleaned_transcript = cleaned_transcript.strip()
    return cleaned_transcript


def save_transcript(file_path: str, transcript: str) -> None:
    """Save a cleaned transcript to a text file."""
    with open(file_path, 'w') as file:
        file.write(transcript)


def upload_transcript(file_path: str, bucket_name: str) -> Tuple[str, str]:
    """Initialize Firebase and upload a transcript."""
    file_ref = f'public/cannabis_data_science/transcripts/{os.path.basename(file_path)}'
    firebase.upload_file(destination_blob_name=file_ref, source_file_name=file_path, bucket_name=bucket_name)
    download_url = firebase.get_file_url(file_ref, bucket_name=bucket_name) 
    return file_ref, download_url


def update_firebase_document(
        doc_ref: str,
        file_ref: str,
        download_url: str,
        short_url: str,
        episode_name: str,
    ) -> Dict[str, str]:
    """Update a Firestore document with transcript data."""
    season, episode = episode_name.split('/')[-2:]
    entry = {
        'file_ref': file_ref,
        'download_url': download_url,
        'short_url': short_url,
        'episode': episode,
        'season': season,
    }
    firebase.update_document(doc_ref, entry)
    return entry


def publish_transcript(episode_name: str) -> Dict[str, str]:
    """Publish a transcript: clean text, save file, upload file, and save data."""
    env_vars = get_env_vars()
    video_dir = env_vars["VIDEO_DIR"]
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = env_vars["GOOGLE_APPLICATION_CREDENTIALS"]
    firebase.initialize_firebase()
    raw_transcript_file = f'{video_dir}/transcripts/raw/{episode_name}.txt'
    cleaned_transcript = clean_transcript(raw_transcript_file)
    transcript_file = f'{video_dir}/transcripts/{episode_name}.txt'
    save_transcript(transcript_file, cleaned_transcript)
    file_ref, download_url = upload_transcript(transcript_file, env_vars["FIREBASE_STORAGE_BUCKET"])
    short_url = firebase.create_short_url(api_key=env_vars["FIREBASE_API_KEY"], long_url=download_url, project_name=env_vars["FIREBASE_PROJECT_ID"])
    doc_ref = f'public/videos/transcripts/{os.path.basename(transcript_file)}'
    data = update_firebase_document(doc_ref, file_ref, download_url, short_url, episode_name)
    return data


# === Test ===
if __name__ == "__main__":

    # Publish a transcript.
    episode_name = sys.argv[1]
    print('Publishing transcript:', episode_name)
    transcript_data = publish_transcript(episode_name)
    print('Transcript published:', transcript_data)

    # # Publish a folder of transcripts.
    # transcript_dir = ''
    # season = 'season-3'
    # episodes = os.listdir(os.path.join(transcript_dir, season))
    # for episode in episodes:
    #     episode_name = f'{season}/{episode.replace(".txt", "")}'
    #     print('Publishing transcript:', episode_name)
    #     transcript_data = publish_transcript(episode_name)
    #     print('Transcript published:', transcript_data)
