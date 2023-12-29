# Descriptive Statistics

In a detailed exploration of Illinois and Massachusetts cannabis retail statistics, we utilize various data sources to unlock insights from public PDFs. The objective is to analyze data from licensed adult use cannabis dispensaries and monthly sales figures to calculate statistics, such as retailers per 100,000 people and sales per retailer. A Fed Fred API Key and `pdfplumber` are required.

The analysis begins with downloading and parsing retailer licensee data from Illinois, followed by similar steps for sales data. With this data, we can calculate the total number of retailers by month and visualize this against Illinois's population data to derive retailers per capita. We also calculate sales per retailer, creating visual representations of the trends.

We then expand the analysis to Massachusetts, using public cannabis data to estimate weekly averages, sales per retailer, and other performance metrics. Importantly, we estimate the relationship between dispensaries per capita and sales per dispensary, using an ordinary least-squares regression. We present our results in a beautiful visualization, emphasizing the relationship between dispensaries per capita and sales per dispensary.

This analysis, made possible from mining data from PDFs, offers valuable insights into the cannabis retail market's dynamics in Illinois and Massachusetts, reflecting the intricate balance between supply, demand, and regulatory impacts on market performance.
