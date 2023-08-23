"""
Video Archive
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 4/27/2023
Updated: 8/17/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    Miscellaneous code for managing the video archive.

Command-line example:

    ```
    python tools/archive_videos.py
    ```

"""
# Internal imports:
from datetime import datetime
import os

# External imports:
from cannlytics import firebase
from dotenv import dotenv_values
import pandas as pd
import numpy as np


def archive_videos():
    """Create an archive of the Cannabis Data Science videos."""

    # Load JSON data from a file.
    with open('../videos.json', 'r') as file:
        df = pd.read_json(file)

    # Clean the data.
    df['published_at'] = df['published_at'].where(pd.notna(df['published_at']), np.nan)
    df['recorded_at'] = df['recorded_at'].where(pd.notna(df['recorded_at']), np.nan)

    # TODO: Sort the data as desired.

    # TODO: Remove unused columns.

    # Save the JSON data to an .xlsx file.
    df.to_excel('../videos.xlsx', index=False, engine='openpyxl')

    # Upload the data to Firebase.
    upload_video_data(df)

    # Return the data.
    return df


def upload_video_data(df):
    """Upload video data from local `.datasets`."""

    DOC = 'public/videos'
    REF = f'{DOC}/video_data'
    ID_KEY = 'id'

    # Initialize the Firebase database.
    database = firebase.initialize_firebase()

    
    # Upload each item in the dataset.
    print('Uploading dataset...')
    for _, item in df.iterrows():
        item['updated_at'] = datetime.now().isoformat()
        doc_id = item[ID_KEY]
        ref = f'{REF}/{doc_id}'
        firebase.update_document(ref, item.to_dict(), database=database)
        print('Updated:', ref)
    
    # Update the stats document if required.
    if DOC:
        firebase.update_document(DOC, {'total': len(df)}, database=database)
        print('Updated:', DOC)

    print('Uploaded video data.')
    return df


# === Test ===
if __name__ == '__main__':

    # Set credentials.
    try:
        config = dotenv_values('../.env')
        credentials = config['GOOGLE_APPLICATION_CREDENTIALS']
    except KeyError:
        config = dotenv_values('.env')
        credentials = config['GOOGLE_APPLICATION_CREDENTIALS']
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials

    # Archive videos.
    videos = archive_videos()
