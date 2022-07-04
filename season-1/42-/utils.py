"""
Utility Functions | Cannabis Data Science

Author: Keegan Skeate <keegan@cannlytics.com>
Created: 10/27/2021
Updated: 11/3/2021
License: MIT License <https://opensource.org/licenses/MIT>
"""
from pandas import Series, to_datetime


def end_of_period_timeseries(df, period='M'):
    """Convert a DataFrame from beginning-of-the-period to
    end-of-the-period timeseries.
    Args:
        df (DataFrame): The DataFrame to adjust timestamps.
        period (str): The period of the time series, monthly "M" by default.
    Returns:
        (DataFrame): The adjusted DataFrame, with end-of-the-month timestamps.
    """
    df.index = df.index.to_period(period).to_timestamp(period)
    return df



def forecast_arima(model, forecast_horizon, X=None):
    """Format an auto-ARIMA model forecast as a time series.
    Args:
        model (ARIMA): An pmdarima auto-ARIMA model.
        forecast_horizon (DatetimeIndex): A series of dates.
        X (DataFrame): Am optional DataFrame of exogenous variables.
    Returns:
        forecast (Series): The forecast series with forecast horizon index.
        conf (Array): A 2xN array of lower and upper confidence bounds.
    """
    periods = len(forecast_horizon)
    forecast, conf = model.predict(
        n_periods=periods,
        return_conf_int=True,
        X=X,
    )
    forecast = Series(forecast)
    forecast.index = forecast_horizon
    return forecast, conf


def format_millions(x, pos=None):
    """The two args are the value and tick position."""
    return '%1.0fM' % (x * 1e-6)


def format_thousands(x, pos=None):
    """The two args are the value and tick position."""
    return '%1.0fK' % (x * 1e-3)


def reverse_dataframe(df):
    """Reverse the ordering of a DataFrame.
    Args:
        df (DataFrame): A DataFrame to re-order.
    Returns:
        (DataFrame): The re-ordered DataFrame.
    """
    return df[::-1].reset_index(drop=True)


def set_training_period(series, date_start, date_end):
    """Helper function to restrict a series to the desired
    training time period.
    Args:
        series (Series): The series to clean.
    Returns
        (Series): The series restricted to the desired time period.
    """
    return series.loc[
        (series.index >= to_datetime(date_start)) & \
        (series.index < to_datetime(date_end))
    ]

