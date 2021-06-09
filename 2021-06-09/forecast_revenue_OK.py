"""
Forecast Cannabis Revenue in Oklahoma | Cannlytics

Author: Keegan Skeate
Contact: <keegan@cannlytics.com>
Created: 5/27/2021
Updated: 5/30/2021
License: MIT License <https://opensource.org/licenses/MIT>
Data Sources:
    - [Medical Marijuana Excise Tax](https://oklahomastate.opengov.com/transparency#/33894/accountType=revenues&embed=n&breakdown=types&currentYearAmount=cumulative&currentYearPeriod=months&graph=bar&legendSort=desc&month=5&proration=false&saved_view=105742&selection=A49C34CEBF1D01A1738CB89828C9274D&projections=null&projectionType=null&highlighting=null&highlightingVariance=null&year=2021&selectedDataSetIndex=null&fiscal_start=earliest&fiscal_end=latest)
Resources:
    - [SQ788](https://www.sos.ok.gov/documents/questions/788.pdf)
    - [How to extract text from a PDF](https://stackoverflow.com/questions/34837707/how-to-extract-text-from-a-pdf-file/63518022#63518022)
    - [Timeseries forecasting with ARIMA models](https://github.com/SushmithaPulagam/TimeSeries_Auto-ARIMA)
"""

import matplotlib.pyplot as plt
import pandas as pd

from pmdarima.arima import auto_arima

# Surpress warnings.
import warnings
warnings.filterwarnings('ignore')

EXCLUDE = [
    '',
    'OMMA.ok.gov',
    'Page 518 of 518',
    'Oklahoma Medical Marijuana Authority',
    'Licensed Growers | May 26, 2021',
    'NAME',
    'ZIP',
    'PHONE',
    'CITY',
    'LICENSE No.',
    'EMAIL',
    'COUNTY',
]

EXLUDE_PARTS = [
    'Page ',
    'Licensed ',
    'LIST OF ',
    'As of '
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
}

MONTHS = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September': 9,
    'October': 10,
    'November': 11,
    'December' : 12,
}

TAX_RATE = 7

def parse_excise_tax(csv_data):
    """Parse excise tax from CSV to a Pandas DataFrame."""
    values = []
    data = csv_data.iloc[1].to_dict()
    for key, value in data.items():
        if 'Actual' in key:
            timestring = key.replace(' Actual', '')
            parts = timestring.replace(' ', '-').split('-')
            month_name = parts[0]
            month = MONTHS[month_name]
            if month < 6:
                year = '2021' # FIXME: Format year dynamically.
            else:
                year = '2020'
            values.append({
                'date': f'{year}-{month}',
                'excise_tax': int(value.strip().replace(',', '')),
            })
    return pd.DataFrame(values)
    


if __name__ == '__main__':
    
    #--------------------------------------------------------------------------
    # Read in the data.
    #--------------------------------------------------------------------------
    
    # Read licensees data.
    licensees = pd.read_excel('data/licensees_OK_2021-06-02.xlsx')
    
    # Read Oklahoma tax data.
    raw_excise_tax_data = pd.read_csv(
        'data/Oklahoma Data Snapshot 2021-06-09.csv',
        skiprows=4
    )
    
    #--------------------------------------------------------------------------
    # Calculate total revenue.
    #--------------------------------------------------------------------------
    
    # Format excise tax.    
    data = parse_excise_tax(raw_excise_tax_data)
    data = data.set_index(pd.to_datetime(data['date']))
    
    data['revenue'] = data['excise_tax'] * 100 / TAX_RATE
    data.revenue.plot()
    
    #--------------------------------------------------------------------------
    # Calculate the trend.
    #--------------------------------------------------------------------------

    # Add a time index.
    # data['t'] = range(0, len(data))

    # # Run a regression of total revenue on time, where t = 0, 1, ... T.
    # model = sm.formula.ols(formula='revenue ~ t', data=data)
    # regression = model.fit()
    # print(regression.summary())

    # # Plot the trend with total revenue.
    # data['trend'] = regression.predict()
    # data[['revenue', 'trend']].plot()

    # Calculate estimated revenue per dispensary per month.
    dispensaries = licensees.loc[licensees.license_type == 'dispensaries']
    number_of_dispensaries = len(dispensaries)
    data['revenue_per_dispensary'] = data['revenue'] / number_of_dispensaries
    
    #--------------------------------------------------------------------------
    # Forecast revenue for 2021
    #--------------------------------------------------------------------------
    
    # Define historic revenue.
    # Dropping the last observation as it's a duplicate
    historic_revenue = data['revenue'][:-1]
    
    # Fitthe best ARIMA model.
    arima_model = auto_arima(
        historic_revenue,
        start_p=0,
        d=0,
        start_q=0, 
        max_p=6,
        max_d=6,
        max_q=6,
        seasonal=False, 
        error_action='warn',
        trace = True,
        supress_warnings=True,
        stepwise = True,
    )
    
    # Summary of the model
    print(arima_model.summary())
    
    # Predict the next 7 months.
    horizon = 7
    forecast_index = pd.date_range(
        start='2021-05-01',
        end='2021-12-01',
        freq='M'
    )
    forecast = pd.DataFrame(
        arima_model.predict(n_periods=horizon),
        index=forecast_index
    )
    forecast.columns = ['forecast_revenue']
    
    # Plot the forecasts.
    plt.figure(figsize=(8, 5))
    plt.plot(historic_revenue, label='Historic')
    plt.plot(forecast['forecast_revenue'], label='Forecast')
    plt.legend(loc='Left corner')
    plt.show()
    
    # Calculate estimated total revenue in 2021.
    year_forecast = historic_revenue.loc[historic_revenue.index >= pd.to_datetime('2021-01-01')].sum() + forecast['forecast_revenue'].sum()
    year_forecast_millions = year_forecast / 1000000
    
    # TODO: Save the forecasts.
    forecast.to_excel('data/forecast_revenue_OK.xlsx')
    
