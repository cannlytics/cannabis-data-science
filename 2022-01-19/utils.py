"""
Utility Functions | Cannabis Data Science
Copyright (c) 2021-2022 Cannlytics

Author: Keegan Skeate <keegan@cannlytics.com>
Created: 10/27/2021
Updated: 1/18/2022
License: MIT License <https://opensource.org/licenses/MIT>
"""
# Standard imports.
from datetime import datetime
import re
from typing import Any, List, Optional, Tuple

# External imports.
from pandas import DataFrame, Series, to_datetime
from pandas.tseries.offsets import MonthEnd


def end_of_month(value: datetime) -> str:
    """Format a datetime as an ISO formatted date at the end of the month.
    Args:
        value (datetime): A datetime value to transform into an ISO date.
    Returns:
        (str): An ISO formatted date.
    """
    month = value.month
    if month < 10:
        month = f'0{month}'
    year = value.year
    day = value + MonthEnd(0)
    return f'{year}-{month}-{day.day}'


def end_of_year(value: datetime) -> str:
    """Format a datetime as an ISO formatted date at the end of the year.
    Args:
        value (datetime): A datetime value to transform into an ISO date.
    Returns:
        (str): An ISO formatted date.
    """
    return f'{value.year}-12-31'


def end_of_period_timeseries(data: DataFrame, period: Optional[str] = 'M') -> DataFrame:
    """Convert a DataFrame from beginning-of-the-period to
    end-of-the-period timeseries.
    Args:
        data (DataFrame): The DataFrame to adjust timestamps.
        period (str): The period of the time series, monthly "M" by default.
    Returns:
        (DataFrame): The adjusted DataFrame, with end-of-the-month timestamps.
    """
    data.index = data.index.to_period(period).to_timestamp(period)
    return data


def forecast_arima(
        model: Any,
        forecast_horizon: Any,
        exogenous: Optional[Any] = None,
) -> Tuple[Any]:
    """Format an auto-ARIMA model forecast as a time series.
    Args:
        model (ARIMA): An pmdarima auto-ARIMA model.
        forecast_horizon (DatetimeIndex): A series of dates.
        exogenous (DataFrame): Am optional DataFrame of exogenous variables.
    Returns:
        forecast (Series): The forecast series with forecast horizon index.
        conf (Array): A 2xN array of lower and upper confidence bounds.
    """
    periods = len(forecast_horizon)
    forecast, conf = model.predict(
        n_periods=periods,
        return_conf_int=True,
        X=exogenous,
    )
    forecast = Series(forecast)
    forecast.index = forecast_horizon
    return forecast, conf


def format_billions(value: float, pos: Optional[int] = None) -> str: #pylint: disable=unused-argument
    """The two args are the value and tick position."""
    return '%1.1fB' % (value * 1e-9)


def format_millions(value: float, pos: Optional[int] = None) -> str: #pylint: disable=unused-argument
    """The two args are the value and tick position."""
    return '%1.1fM' % (value * 1e-6)


def format_thousands(value: float, pos: Optional[int] = None) -> str: #pylint: disable=unused-argument
    """The two args are the value and tick position."""
    return '%1.0fK' % (value * 1e-3)


def get_blocks(files, size=65536):
    """Get a block of a file by the given size."""
    while True:
        block = files.read(size)
        if not block: break
        yield block


def get_number_of_lines(file_name, encoding='utf-16', errors='ignore'):
    """
    Read the number of lines in a large file.
    Credit: glglgl, SU3 <https://stackoverflow.com/a/9631635/5021266>
    License: CC BY-SA 3.0 <https://creativecommons.org/licenses/by-sa/3.0/>
    """
    with open(file_name, 'r', encoding=encoding, errors=errors) as f:
        count = sum(bl.count('\n') for bl in get_blocks(f))
        print('Number of rows:', count)
        return count


def reverse_dataframe(data: DataFrame) -> DataFrame:
    """Reverse the ordering of a DataFrame.
    Args:
        data (DataFrame): A DataFrame to re-order.
    Returns:
        (DataFrame): The re-ordered DataFrame.
    """
    return data[::-1].reset_index(drop=True)


def set_training_period(series: Series, date_start: str, date_end: str) -> Series:
    """Helper function to restrict a series to the desired
    training time period.
    Args:
        series (Series): The series to clean.
        date_start (str): An ISO date to mark the beginning of the training period.
        date_end (str): An ISO date to mark the end of the training period.
    Returns
        (Series): The series restricted to the desired time period.
    """
    return series.loc[
        (series.index >= to_datetime(date_start)) & \
        (series.index < to_datetime(date_end))
    ]


def sorted_nicely(unsorted_list: List[str]) -> List[str]:
    """Sort the given iterable in the way that humans expect.
    Credit: Mark Byers <https://stackoverflow.com/a/2669120/5021266>
    License: CC BY-SA 2.5 <https://creativecommons.org/licenses/by-sa/2.5/>
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(unsorted_list, key=alphanum_key)
