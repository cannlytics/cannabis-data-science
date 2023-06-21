"""
Get Cannabis Plant Patents
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 6/5/2023
Updated: 6/6/2023
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# Standard imports:
from datetime import datetime
import os
import re
from time import sleep
from typing import Any, Iterable, Optional

# External imports:
import pandas as pd
import pdfplumber
from pytesseract import image_to_string
import requests
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def get_patent_metadata(
        search_terms: Any,
        verbose: Optional[bool] = True
    ) -> pd.DataFrame:
    """Get metadata for patents matching the search terms."""

    # Initialize a web driver.
    try:
        service = Service()
        options = Options()
        # options.add_argument('--window-size=1920,1200')
        # options.add_argument('--headless')
        # options.add_argument('--disable-gpu')
        # options.add_argument('--no-sandbox')
        driver = webdriver.Chrome(options=options, service=service)
    except:
        options = EdgeOptions()
        # options.add_argument('--headless')
        driver = webdriver.Edge(options=options)

    # Get the patent search page.
    driver.get('https://ppubs.uspto.gov/pubwebapp/static/pages/ppubsbasic.html')

    # Search for patents.
    if isinstance(search_terms, str):
        search_terms = {'searchText1': search_terms, 'searchText2': ''}
    else:
        search_terms = {'searchText1': search_terms[0], 'searchText2': search_terms[-1]}
        # search_terms['searchText2'] = search_terms[-1] if len(search_terms) > 1 else ''
    print(search_terms)
    for element_id, term in search_terms.items():
        input_element = driver.find_element(By.ID, element_id)
        input_element.clear()
        input_element.send_keys(term)

    # Click the search button.
    search_button = driver.find_element(by=By.ID, value='searchText2')
    search_button.send_keys(Keys.ENTER)
    sleep(3)

    # Iterate over search result pages.
    iterate = True
    patents = []
    while iterate:

        # Wait until the overlay is no longer visible.
        WebDriverWait(driver, 3).until(EC.invisibility_of_element((By.CSS_SELECTOR, '.overlay.full-page')))

        # Extract the page number and total number of pages.
        page_info_span = driver.find_element(By.ID, 'pageInfo')
        page_info_text = page_info_span.text
        current_page, total_pages = page_info_text.split(" of ")
        current_page = current_page.replace('Page ', '')
        
        # Get the metadata for each patent.
        metadata = []
        table = driver.find_element(by=By.ID, value='searchResults')
        rows = table.find_elements(by=By.TAG_NAME, value='tr')[1:]
        if verbose:
            print('Getting %i patents on page' % len(rows), current_page, 'of', total_pages)
        for row in rows:
            metadata.append({
                'query_number': row.find_element(By.XPATH, './td[1]').text,
                'patent_number': row.find_element(By.XPATH, './td[2]').text,
                'patent_title': row.find_element(By.XPATH, './td[4]').text,
                'inventor_name': row.find_element(By.XPATH, './td[5]').text,
                'date_published': row.find_element(By.XPATH, './td[6]').text,
                'page_count': int(row.find_element(By.XPATH, './td[7]').text)
            })

        # Add URLs to the list of dictionaries.
        count = 0
        links = driver.find_elements(by=By.TAG_NAME, value='a')
        for link in links:
            href = link.get_attribute('href')
            if href and 'image-ppubs.uspto.gov' in href:
                metadata[count]['patent_url'] = href
                count += 1

        # Record the patent data.
        patents.extend(metadata)
        
        # Click the next button if it's not the last page.
        if current_page != total_pages:
            next_button = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.ID, 'paginationNextItem'))
            )
            next_button.click()
        else:
            iterate = False

    # Close the driver.
    if verbose:
        print('Found %i patents, closing driver.' % len(patents))
    driver.quit()

    # Save the patent metadata.
    if verbose:
        print('Saving patent metadata...')
    date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    metadata_file = f'D://data/strains/patents/patent-metadata-{date}.xlsx'
    patent_data = pd.DataFrame(patents)
    patent_data.to_excel(metadata_file)
    if verbose:
        print('Patent metadata saved:', metadata_file)

    # Download patent images.
    if verbose:
        print('Downloading patent images...')
    pdf_dir = 'D://data/strains/patents/pdfs/'
    for patent in patents:
        outfile = os.path.join(pdf_dir, patent['patent_number'] + '.pdf')
        if os.path.exists(outfile):
            if verbose:
                print('Cached:', outfile)
            continue
        response = requests.get(patent['patent_url'])
        with open(outfile, 'wb') as file:
            file.write(response.content)
        if verbose: 
            print('Downloaded:', outfile)
    
    # Return the patent data.
    return patent_data


#-----------------------------------------------------------------------
# Query cannabis plant patents.
#-----------------------------------------------------------------------
if __name__ == '__main__':

    # Iterate over search term queries.
    all_data = []
    queries = [
        # ('cannabis', ''),
        # ('cannabis', 'plant'),
        ('cannabis', 'cultivar'),
        ('cannabis', 'variety'),
        # ('hemp', ''),
        # ('hemp', 'plant'),
        ('hemp', 'cultivar'),
        ('hemp', 'variety'),
        # ('marijuana', ''),
        # ('marijuana', 'plant'),
        ('marijuana', 'cultivar'),
        ('marijuana', 'variety'),
    ]
    for query in queries:
        data = get_patent_metadata(query)
        all_data.extend(data)

    # Aggregate all patents from all queries.
    patent_data = pd.DataFrame(all_data)
    patent_data.drop_duplicates(subset='patent_number', inplace=True)

    # Save the aggregated patent data.
    date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    patent_data.to_excel(f'D://data/strains/patents/patents-{date}.xlsx')
