"""
Get Michigan Cannabis Data | Cannabis Data Science Meetup Group
Copyright (c) 2021 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 12/8/2021
Updated: 12/8/2021
License: MIT License <https://opensource.org/licenses/MIT>

Data Sources:
    

Resources:
    
    - Fed Fred API Keys
    <https://fred.stlouisfed.org/docs/api/api_key.html>
    
Objective:
    
    Retrieve Michigan cannabis data, locked in public PDFs,
    to save the data and calculate interesting statistics,
    such as retailers per 100,000 people and sales per retailer.
    
    You will need a Fed Fred API Key saved in a .env file
    as a FRED_API_KEY variable. A `data` and `figure` folders
    are also expected.

    You will also need to install various Python dependencies,
    including fredapi and pdfplumber.
    
    `pip install fredapi pdfplumber`

"""