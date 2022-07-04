"""
Utility Functions | Cannabis Data Science

Author: Keegan Skeate <keegan@cannlytics.com>
Created: 10/27/2021
Updated: 10/27/2021
License: MIT License <https://opensource.org/licenses/MIT>
"""


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


def reverse_dataframe(df):
    """Reverse the ordering of a DataFrame.
    Args:
        df (DataFrame): A DataFrame to re-order.
    Returns:
        (DataFrame): The re-ordered DataFrame.
    """
    return df[::-1].reset_index(drop=True)