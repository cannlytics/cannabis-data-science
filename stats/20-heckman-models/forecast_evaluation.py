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
