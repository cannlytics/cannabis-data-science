"""
Repository Management Tools | Cannabis Data Science
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 7/3/2022
Updated: 7/6/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
import json
import os
import pandas as pd


def format_link(href, text):
    """Format a hyperlink as a link with given text."""
    return f'<a href="{href}">{text}</a>'


def create_readme_index(datafile):
    """Create a markdown table from given JSON."""

    # Read the videos JSON.
    videos = pd.read_json(datafile)

    # Rename columns as desired.
    columns = {
        'short_title': 'Topic',
        'description': 'Description',
        'video_url': 'Video',
        'source_code_url': 'Code',
    }
    videos.rename(columns=columns, inplace=True)

    # Keep only necessary columns.
    videos = videos[list(columns.values())]

    # Keep only videos that are posted.
    videos = videos.loc[videos['Video'] != '']

    # Shorten descriptions.
    videos['Description'] = videos['Description'].apply(lambda x: x.split('\n')[0])

    # Format links.
    videos['Topic'] = videos.apply(
        lambda x: format_link(x['Code'], x['Topic']),
        axis=1,
    )
    videos['Video'] = videos['Video'].apply(format_link, text='Video')
    videos['Code'] = videos['Code'].apply(format_link, text='Code')

    # Print out a table for the readme.
    html = videos.to_html(index=False, render_links=True, escape=False)
    return html


def list_github_folder_urls(base_dir='../../', season='season-1'):
    """List GitHub source code URL from folders."""
    directory = '/'.join([base_dir, season]) + '/'
    folders = [f for f in os.listdir(directory)]
    urls = []
    for folder in folders:
        base = 'https://github.com/cannlytics/cannabis-data-science/tree/main/{}/{}'
        url = base.format(season, folder)
        urls.append(url)
    return urls


if __name__ == '__main__':
    
    # # List repository folder URLs, to manually update `videos.json`.
    # videos = pd.read_json('../../assets/videos.json')
    # season_1_urls = list_github_folder_urls(season='season-1')
    # season_2_urls = list_github_folder_urls(season='season-2')
    # for i, url in enumerate(season_1_urls):
    #     videos.loc[i, 'source_code_url'] = url
    # for i, url in enumerate(season_2_urls[:-1]):
    #     videos.loc[i + 47, 'source_code_url'] = url

    # # Fix published at date.
    # videos['published_at'] = videos['published_at'].apply(lambda x: x.date()).astype(str)

    # # Future work: Add keywords.
    # videos.loc[videos.keywords.isnull(), 'keywords'] = ''
    # videos['keywords'] = videos['keywords'].apply(list)

    # # Future work: Automate the update of `videos.json`.
    # print(json.dumps(videos.to_dict(orient='records'), indent=2))
    
    # Create the table of episodes and update the `readme.md`.
    filename = '../../readme.md'
    table = create_readme_index('../../assets/videos.json')
    with open(filename, 'r') as f:
        text = f.readlines()
        readme = ''.join([line for line in text])
        intro = readme.split('<table')[0]
        outro = readme.split('</table>')[-1]
        updated_readme = ''.join([intro, table, outro])
    with open(filename, 'w') as f:
        f.write(updated_readme)
