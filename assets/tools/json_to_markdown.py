"""
Repository Management Tools | Cannabis Data Science
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 7/3/2022
Updated: 7/3/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""

import pandas as pd


def format_link(href, text):
    """Format a hyperlink as a link with given text."""
    return f'<a href="{href}">{text}</a>'


def create_readme_index():
    """Create a markdown table from given JSON."""

    # Read the videos JSON.
    videos = pd.read_json('../videos.json')

    # Rename columns as desired.
    columns = {
        'short_title': 'Topic',
        'description': 'Description',
        'video_url': 'Video',
        'source_code_url': 'Notes',
    }
    videos.rename(columns=columns, inplace=True)

    # Keep only necessary columns.
    videos = videos[list(columns.values())]

    # Keep only videos that are posted.
    videos = videos.loc[videos['Video'] != '']

    # Shorten descriptions.
    videos['Description'] = videos['Description'].apply(lambda x: x.split('\n')[0])

    # Format links.
    videos['Video'] = videos['Video'].apply(format_link, text='Video')
    videos['Notes'] = videos['Notes'].apply(format_link, text='Notes')

    # Print out a table for the readme.
    print(videos.to_html(index=False, render_links=True, escape=False))


if __name__ == '__main__':

    create_readme_index()
