"""
Box Jenkins ARIMA Forecasting Methodology
Copyright (c) 2017-2021 Keegan Skeate

Author: Keegan Skeate <keegan@cannlytics.com>
Created: March 2017
Updated: 10/27/2021
License: MIT License <https://opensource.org/licenses/MIT>
"""
# Standard imports:
from typing import Any, Optional

# External imports:
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA


def RMSE(predictions, actuals):
    """Calculate the root mean squared error of two series."""
    return np.sqrt(((predictions - actuals) ** 2).mean())


def arima_min_rmse_forecast(
        time_series,
        lag_order=6,
        hold_out_period=6,
        horizon=12,
        verbose=False,
    ):
    """Forecast a series with an ARIMA model selected by minimum RMSE
    for a given holdout period.

    Args:
        time_series (Series):
        lag_order (int): The maximum lag order for the ARIMA model.
        hold_out_period (int): The number of periods to hold out for training.
        horizon (int): The number of periods to forecast.
        verbose (bool): Wether or not to print out details.
    """

    # Initialize time series.
    X = time_series
    train, test = X[0:-hold_out_period], X[-hold_out_period:len(X)]
    p, q = np.arange(lag_order + 1), np.arange(lag_order + 1)
    
    # Define forecast variables.
    forecasts = list()
    lag_order_of_forecast = list()
    rmse_of_forecast = list()
    
    # Scan ARMA(p, q) models.
    for p in range(lag_order + 1):
        for q in range(lag_order + 1):
            order = np.array([p, q])
            lag_order_of_forecast.append(order)
            if verbose:
                print('Fitting ARIMA', order)

            # Rolling forecast.
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                try:
                    model = ARIMA(history, order=(p, 0, q))
                    model_fit = model.fit(disp=0)
                    output = model_fit.forecast()
                    yhat = output[0]
                    predictions.append(yhat)
                    actual = test[t]
                    history.append(actual)
                except:
                    pass
                if verbose:
                    print('y_hat: %f | y: %f | RMSE: %f' % (yhat, actual, rmse))
            if len(predictions)==0:
                forecasts.append(np.array([np.inf, np.inf, np.inf]))
                rmse_of_forecast.append(np.inf)
                if verbose:
                    print ('Not Stationary')
            else:
                forecasts.append(predictions)
                rmse = RMSE(predictions, test)
                rmse_of_forecast.append(rmse)
                if verbose:
                    print ('RMSE: %.4f' % rmse)

    # Identify the best model.
    min_rmse_model = rmse_of_forecast.index(min(rmse_of_forecast))
    best_lag_order = lag_order_of_forecast[min_rmse_model]
    if verbose:
        print('Best model: ARIMA', best_lag_order)

    # Forecast with the best model.
    order = (best_lag_order[0], 0, best_lag_order[1])
    model = ARIMA(X, order =order )
    regression = model.fit(disp=0)
    forecast = regression.forecast(horizon)
    return forecast[0]


def arima_min_bic_forecast(
        time_series: Any,
        lag_order: int,
        horizon: int,
        verbose: Optional[bool] = False,
    ):
    """Forecast a series with an ARIMA model selected by minimum BIC.

    Args:
        time_series (Series):
        lag_order (int): The maximum lag order for the ARIMA model.
        hold_out_period (int): The number of periods to hold out for training.
        horizon (int): The number of periods to forecast.
        verbose (bool): Wether or not to print out details.
    """
    X = time_series
    p, q = np.arange(lag_order+1) , np.arange(lag_order+1)
    lag_order_of_model = list()
    bic_of_model = list()

    # Scan ARMA(p, q) models.
    for p in range(lag_order + 1):
        for q in range(lag_order + 1):
            order = np.array([p, q])
            lag_order_of_model.append(order)
            try:
                model = ARIMA(X, order=(p, 0, q))
                model_fit = model.fit(disp=0)
                bic = model_fit.bic
            except:
                bic = np.inf
                if verbose:
                    print ('Not Stationary')
                pass
            bic_of_model.append(bic)
            if verbose:
                print ('Model: (%i, %i)' % (p, q), 'BIC: %.4f' % bic)

    # Identify the best model.
    min_bic_model = bic_of_model.index(min(bic_of_model))
    best_lag_order = lag_order_of_model[min_bic_model]
    if verbose:
        print('Best model:', best_lag_order)

    # Forecast with the best model.
    best_model = ARIMA(X, order = (best_lag_order[0], 0, best_lag_order[1]) )
    best_model_fit = best_model.fit(disp=0)
    best_output = best_model_fit.forecast(horizon)
    best_forecast = best_output[0]
    return pd.Series(best_forecast)
