"""
CannPatent
Copyright (c) 2022
Created: 5/21/2022
Updated: 5/25/2022
Authors: Keegan Skeate <https://github.com/keeganskeate>
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    Find and curate data for cannabis patents. In particular, this
    script collects detailed data for plant patents. Subsequent
    intellectual property (IP) analytics provide actionable insights
    for cannabis cultivar inventors and consumers. For example,
    cultivators can use the methodology to predict if a particular
    cultivar would make a good patent candidate given its lab results.
    Consumers can find the nearest patented strain to a set of lab results
    printed on a cultivar's label.

Data Source:

    - United States Patent and Trademark Office
    URL: <www.uspto.gov>

Requirements

    - ImageMagick
    Download: <https://imagemagick.org/script/download.php#windows>
    Cloud: <https://stackoverflow.com/questions/43036268/do-i-have-access-to-graphicsmagicks-or-imagemagicks-in-a-google-cloud-function>

"""
# Standard imports.
import os
from datetime import datetime
import re
from typing import Any, Optional, Union

# External imports.
from bs4 import BeautifulSoup
from cannlytics.utils.data import to_excel_with_style
from cannlytics.utils.utils import (
    camel_to_snake,
    clean_dictionary,
    kebab_case,
    snake_case,
)
import math
import pandas as pd
import requests
from time import sleep


def curate_lab_results(
        data_dir,
        compound_folder='Terpene and Cannabinoid data',
        cannabinoid_file='rawDATACana',
        terpene_file='rawDATATerp',
        max_cannabinoids=35,
        max_terpenes=8,
    ):
    """Curate lab results for effects prediction model."""

    # Decarboxylation rate. Source: <https://www.conflabs.com/why-0-877/>
    decarb = 0.877

    # Read terpenes.
    terpenes = None
    if terpene_file:
        file_path = os.path.join(data_dir, compound_folder, terpene_file)
        terpenes = pd.read_csv(file_path, index_col=0)
        terpenes.columns = [snake_case(x).strip('x_') for x in terpenes.columns]
        terpene_names = list(terpenes.columns[3:])
        compounds = terpenes

    # Read cannabinoids.
    cannabinoids = None
    if cannabinoid_file:
        file_path = os.path.join(data_dir, compound_folder, cannabinoid_file)
        cannabinoids = pd.read_csv(file_path, index_col=0)
        cannabinoids.columns = [snake_case(x).strip('x_') for x in cannabinoids.columns]
        cannabinoid_names = list(cannabinoids.columns[3:])
        compounds = cannabinoids

    # Merge terpenes and cannabinoids.
    if terpene_file and cannabinoid_file:
        compounds = pd.merge(
            left=cannabinoids,
            right=terpenes,
            left_on='file',
            right_on='file',
            how='left',
            suffixes=['', '_terpene']
        )

    # Rename any oddly named columns.
    rename = {
        'cb_da': 'cbda',
        'cb_ga': 'cbda',
        'delta_9_th_ca': 'delta_9_thca',
        'th_ca': 'thca',
    }
    compounds.rename(columns=rename, inplace=True)

    # Combine `delta_9_thca` and `thca`.
    # FIXME: Ensure that this is combining the two fields correctly.
    compounds['delta_9_thca'].fillna(compounds['thca'], inplace=True)
    compounds.drop(columns=['thca'], inplace=True)
    cannabinoid_names.remove('thca')

    # FIXME: Combine any additional compounds.
    # compounds['delta_9_thca'].fillna(compounds['thca'], inplace=True)
    # compounds.drop(columns=['thca'], inplace=True)
    # cannabinoid_names.remove('thca')

    # FIXME: Calculate totals.
    compounds['total_terpenes'] = compounds[terpene_names].sum(axis=1).round(2)
    compounds['total_cannabinoids'] = compounds[cannabinoid_names].sum(axis=1).round(2)
    compounds['total_thc'] = (compounds['delta_9_thc'] + compounds['delta_9_thca'].mul(decarb)).round(2)
    compounds['total_cbd'] = (compounds['cbd'] + compounds['cbda'].mul(decarb)).round(2)

    # Exclude outliers.
    compounds = compounds.loc[
        (compounds['total_cannabinoids'] < max_cannabinoids) &
        (compounds['total_terpenes'] < max_terpenes)
    ]

    # Return compounds with nulls as 0.
    compounds = compounds.fillna(0)
    return compounds


def search_patents(
        query: str,
        limit: Optional[int] = 50,
        details: Optional[bool] = False,
        pause: Optional[float] = None,
        term: Optional[str] = '',
    ) -> Any:
    """Search for patents.
    Args:
        query (str): The search term.
        limit (int): The number of patents to retrieve, 50 by default.
            The algorithm requests patents in batches of 50.
        details (bool): Whether or not to return extensive details for
            each patent. The default is `False`.
    Returns:
        (DataFrame): The patents' data.
    """

    # Define the query URL.
    query = query.replace(' ', '+')
    base = 'http://patft.uspto.gov/netacgi/nph-Parser'
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'}
    url = f'{base}?Sect1=PTO2&Sect2=HITOFF&u=%2Fnetahtml%2FPTO%2Fsearch-adv.htm&r=0&p=1&f=S&l=50&Query='
    url += f'{term}"{query}"&d=PTXT'
    print(url)

    # Future work: Allow user to specify date ranges for the search query.
    # ISD/11/1/1997->5/12/1998

    # Iterate over found items, 50 per page.
    patents = pd.DataFrame()
    pages = math.ceil(limit / 50)
    page_range = range(0, int(pages))
    for increment_page in page_range:

        # Get cannabinoid patents.
        if increment_page:
            url += f'&OS={query}&RS={query}'
            url += f'&TD=6080&Srch1={query}&NextList{increment_page + 1}=Next+50+Hits'
        # FIXME:
        try:
            response = requests.get(url, headers=headers)
        except:
            print('Error on page', increment_page)
            sleep(62)
            response.connection.close()
            response = requests.get(url, headers=headers)
            # try:
            #     response = requests.get(url, headers=headers)
            # except:
            #     continue
        if pause:
            sleep(pause)

        # Create lists of patents, titles, and URLs.
        soup = BeautifulSoup(response.content, features='html.parser')
        ids = []
        patent_numbers = []
        titles = []
        links = []
        for link in soup.find_all('a', href=True):
            if link.text and (link.text.startswith('PP') or (len(link.text) <= 10 and ',' in link.text)):
                ids.append(link.text)
                patent_number = link.text.replace(',', '')
                patent_numbers.append(patent_number)
                title_link = link.findNext('a')
                titles.append(title_link.text.strip())
                links.append('http://patft.uspto.gov' + title_link['href'])

        # Format the patents as a DataFrame.
        if patent_numbers == 0:
            break
        patents = pd.concat([patents, pd.DataFrame({
            'patent_number': patent_numbers,
            'patent_number_formatted': ids,
            'patent_title': titles,
            'patent_url': links,
        })])

        # Optionally get details for each patent.
        # Note: This can probably be done more efficiently.
        if details:
            patent_details = []
            for _, patent in patents[:limit].iterrows():
                patent_detail = get_patent_details(patent)
                patent_details.append(patent_detail)
            patents = pd.concat(patent_details, axis=1)
            if isinstance(patents, pd.Series):
                patents = patents.to_frame()
            patents = patents.transpose()

    # Return the patents found.
    # FIXME: Only return the limit?
    # patents = patents[:limit]
    return patents


def get_patent_details(
        data: Optional[Any] = None,
        patent_number: Optional[str] = None,
        patent_url: Optional[str] = None,
        user_agent: Optional[str] = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
        fields: Optional[Union[str, list]] = None,
        search_field: Optional[str] = 'patentNumber',
        search_fields: Optional[str] = 'patentNumber',
        query: Optional[str] = 'patentNumber',
    ) -> Any:
    """Get details for a given patent, given it's patent number and URL.
    Args:
        data (Series): Existing patent data with `patent_number` and
            `patent_url` fields (optional). If not specified, then
            pass `patent_number` and `patent_url` arguments.
        patent_number (str): A specific patent number (optional).
        patent_url (str): A specific patent URL (optional).
        user_agent (str): Your browser agent (optional). A typical Chrome
            agent by default.
        fields (list): A list of fields to return. You can use '*' for
            all fields. A curated selection is used by default.
    Returns:
        (Series): The detailed patent data.
    References:
        - https://ped.uspto.gov/peds/#!/#%2FapiDocumentation
    """

    # Ensure that a patent number and patent URL are passed.
    if data is None and patent_number:
        patent = pd.Series({
            'patent_number': patent_number,
            'patent_url': patent_url,
        })
    elif data is not None:
        patent = data
        patent_number = patent['patent_number']
    else:
        raise ValueError

    # Specify the fields to return.
    if fields is None:
        fields = ['appType', 'appFilingDate', 'inventorName',
                  'inventors', 'patentIssueDate'] # 'attrnyAddr',
    if isinstance(fields, str):
        field_list = fields
    else:
        field_list = ','.join(fields)
        
    # Request fields for the patent.
    base = 'https://ped.uspto.gov/api/queries'
    headers = {'User-Agent': user_agent}
    data = {
        'searchText': f'{query}:({patent_number})',
        'fl': field_list, # field list
        'df': search_field, # default field to search
        'qf': search_fields, # multiple fields to search
        'facet': 'false',
        'mm': '100%', # minimum match
        'sort': f'{search_field} asc',
        'start': '0',
    }
    response = requests.post(base, json=data, headers=headers)

    # Add the patent details.
    data = response.json()
    docs = data['queryResults']['searchResponse']['response']['docs']
    doc = docs[0]
    doc = clean_dictionary(doc, function=camel_to_snake)

    # Optional: Get the attorney data.
    # doc['attorney'] = [clean_dictionary(x, function=camel_to_snake) for x in doc['attrny_addr']]
    # del doc['attrny_addr']
    
    # FIXME: Get the inventor data (from the text if not present here).
    try:
        doc['inventor_name'] = doc['inventor_name'].title()
        doc['inventors'] = [clean_dictionary(x, function=camel_to_snake) for x in doc['inventors']]
        inventor = doc['inventors'][0]
        doc['inventor_city'] = inventor['city'].replace(',', '').title()
        doc['inventor_state'] = inventor['geo_code']
        doc['inventor_country'] = inventor['country'].replace('(', '').replace(')', '')
    except KeyError:
        pass
    
    # Merge details with patent.
    patent = pd.concat([patent, pd.Series(doc)])

    # Get patent text by parsing the patent's webpage.
    response = requests.get(patent['patent_url'], headers=headers)
    soup = BeautifulSoup(response.content, features='html.parser')

    # Get the abstract.
    patent['abstract'] = soup.p.text.strip().replace('\n     ', ' ')

    # Get the applicant.
    tables = soup.findAll('table')
    values = [x.text.strip() for x in tables[3].findAll('td')]
    try:
        patent['applicant_name'] = values[2]
        patent['applicant_city'] = values[3]
        patent['applicant_state'] = values[4]
        patent['applicant_country'] = values[5]
    except IndexError:
        print('Error parsing applicant:', values)

    # Get the claims.
    # Test: Handle `It is claimed:`.
    claims = soup.text.split('claimed is:')[-1].split('claimed:')[0]
    claims = claims.split('Description  BACKGROUND OF THE INVENTION')[0]
    claims = claims.strip()
    claims = re.split('(\n\s\d.\s\s)', claims)
    claims = claims[::2]
    claims[0] = claims[0].replace('1.  ', '')
    patent['claims'] = [x.replace('\n', ' ').strip() for x in claims]

    # TODO: Download the PDF for the patent.
    # - Upload the PDF to Firebase Storage, saving the ref with the data.

    # Optional: Get plant details.

    # '\nSeeds\n '
    # Market Class: 
    # patent['parentage'] = soup.text.split('Parentage: ')[1].split(':')[0]
    # patent['classification'] = soup.text.split('Classification: ')[1].split(':')[0]

    # Optional: Get lab results?
    # soup.text.split('TABLE-US-')[1]

    # Optional: Get more patent details.
    # - citations_applicant_count
    # - full text (description)

    # Optional: Extract links to references.
    # tables[8].text

    # Bonus: Crop all images for each patent.
    # Upload the images to Firebase Storage for access online / through the API.

    # Return the augmented patent data.
    return patent


def get_strain_name(x):
    """Get a strain name in text surrounded by tildes."""
    try:
        return re.search('`(.*)`', x).group(1)
    except AttributeError:
        return ''


if __name__ == '__main__':

    #-------------------------------------------------------------------
    # 1. Find cannabis plant patents.
    # Future work: Download PDFs for the patents found.
    #-------------------------------------------------------------------

    # Search for cannabis patents.
    queries = [
        # 'cannabis',
        # 'cannabis plant',
        # 'cannabis cultivar',
        # 'cannabis variety',
        'hemp plant',
        # 'hemp cultivar',
        # 'hemp variety',
        # 'marijuana plant',
    ]
    limit = 1000
    for query in queries:

        # Search for patents by keyword(s).
        patents = search_patents(query, limit, term='TTL%2F')
        print('Found %i patents.' % len(patents))

        # Save the patents.
        key = kebab_case(query)
        datafile = f'../../.datasets/ai/plant-patents/{key}-patents.xlsx'
        to_excel_with_style(patents, datafile, sheet_name='Patents')

        # # Read the patents back in.
        # patents = pd.read_excel(datafile)

        # Isolate plant patents.
        # Note: There is probably a better way to identify plant patents.
        cultivars = patents.loc[
            (patents['patent_title'].str.contains('plant', case=False)) |
            (patents['patent_title'].str.contains('cultivar', case=False))
        ]
        print('Found %i cultivar patents.' % len(cultivars))

        # Get strain names.
        strain_names = cultivars['patent_title'].apply(get_strain_name)
        cultivars = cultivars.assign(strain_name=strain_names)

        # TODO: Remove duplicates: it appears that patents can be renewed.

        # Get details for each row.
        # Note: This could probably be done more efficiently.
        cultivar_data = []
        for _, cultivar in cultivars.iterrows():
            patent_details = get_patent_details(cultivar)
            cultivar_data.append(patent_details)
        cultivar_data = pd.concat(cultivar_data, axis=1)
        if isinstance(cultivar_data, pd.Series):
            cultivar_data = cultivar_data.to_frame()
        cultivar_data = cultivar_data.transpose()

        # Save the plant patent data.
        today = kebab_case(datetime.now().isoformat()[:16])
        datafile = f'../../.datasets/ai/plant-patents/plant-patents-{today}.xlsx'
        to_excel_with_style(cultivar_data, datafile, sheet_name='Patent Details')


    # # Lookup referenced cultivars:
    # # - Santhica 27
    # # - BLK03
    # # - AVI-1
    # patent = get_patent_details(pd.Series({
    #     'patent_number': 'PP34051',
    #     'patent_number_formatted': 'PP34,051',
    #     'patent_title': 'Cannabis plant named `AVI-1`',
    #     'patent_url': 'https://patft.uspto.gov/netacgi/nph-Parser?Sect1=PTO2&Sect2=HITOFF&u=%2Fnetahtml%2FPTO%2Fsearch-adv.htm&r=7&f=G&l=50&d=PTXT&p=1&S1=%22marijuana+plant%22&OS=%22marijuana+plant%22&RS=%22marijuana+plant%22',
    #     'strain_name': 'AVI-1',
        
    # }))
    # today = kebab_case(datetime.now().isoformat()[:16])
    # datafile = f'../../.datasets/ai/plant-patents/plant-patent-{today}.xlsx'
    # to_excel_with_style(patent.to_frame().T, datafile, sheet_name='Patent Details')


    #-------------------------------------------------------------------
    # 3. Organize all plant patent data points.
    # At this stage, a data guide can be written, with:
    # - Key
    # - Type
    # - Description
    # - Possible values
    # - Relation to other variables?
    #-------------------------------------------------------------------

    # Read programmatically collected plant patent data.
    datafile = '../../.datasets/ai/plant-patents/plant-patents.xlsx'
    details = pd.read_excel(datafile, sheet_name='Patent Details')

    # Read manually collected plant patent data.
    datafile = '../../.datasets/ai/plant-patents/plant-patents.xlsx'
    results = pd.read_excel(datafile, sheet_name='Patent Lab Results')


    # Count plant patents over time.
    details['date'] = pd.to_datetime(details['patent_issue_date'])
    yearly = details.groupby(pd.Grouper(key='date', freq='Y'))['patent_number'].count()
    yearly.value_counts()


    #-------------------------------------------------------------------
    # 4. Upload the patent data.
    #-------------------------------------------------------------------

    # Upload the data to Firebase Firestore for API access.
    # docs = 
    # 'public/data/plant_patents/{patent_number}



    #-------------------------------------------------------------------
    # Perform analysis of patent data.
    #-------------------------------------------------------------------

    # Plot the number of cannabis plant patents by year from the 1st to
    # the present.


    # Look at pertinent ratios.
    # - THCV to THC ratio.
    # - THCV to total cannabinoid ratio.
    # - THC / CBD ratio.
    # - CBD / CBC / THC ratio.
    # - All terpene ratios!!!


    #-------------------------------------------------------------------
    # Bonus: Visualize the patent data.
    #-------------------------------------------------------------------

    # Ridge plots for cultivars with terpene data.


    # Regression plots of ratios with all strains colored by strain.

