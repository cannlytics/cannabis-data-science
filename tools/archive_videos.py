"""
Video Archive
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 4/27/2023
Updated: 4/27/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description: Miscellaneous code for managing the video archive.
"""
import pandas as pd


def archive_videos():
    """Create an archive of the Cannabis Data Science videos."""

    # Load JSON data from a file.
    with open('../videos.json', 'r') as file:
        df = pd.read_json(file)

    # Clean the data.
    df['recorded_at'] = df['recorded_at'].dt.tz_convert(None)

    # TODO: Sort the data as desired.

    # TODO: Remove unused columns.

    # Save the JSON data to an .xlsx file.
    df.to_excel('../videos.xlsx', index=False, engine='openpyxl')


# === Test ===
if __name__ == '__main__':
    archive_videos()
