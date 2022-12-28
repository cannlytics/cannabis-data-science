"""
Utility Functions | Cannabis Data Science
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 10/27/2021
Updated: 12/11/2022
License: MIT License <https://opensource.org/licenses/MIT>
"""
import numpy as np


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


def format_millions(x, pos):
    """The two args are the value and tick position."""
    return '%1.0fM' % (x * 1e-6)


def reverse_dataframe(df):
    """Reverse the ordering of a DataFrame.
    Args:
        df (DataFrame): A DataFrame to re-order.
    Returns:
        (DataFrame): The re-ordered DataFrame.
    """
    return df[::-1].reset_index(drop=True)


def draw_brace(ax, x_span, yy, text):
    """Draws an annotated brace on the axes."""
    xmin, xmax = x_span
    x_span = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    ax_ymin, ax_ymax = ax.get_ylim()
    xax_span = ax_xmax - ax_xmin
    y_span = ax_ymax - ax_ymin
    resolution = int(x_span / xax_span * 100) * 2 + 1 # guaranteed uneven
    beta = 300.0 / xax_span # the higher this is, the smaller the radius
    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(resolution/2) + 1]
    y_half_brace = (1 / (1.0 + np.exp(-beta * (x_half - x_half[0])))
                    + 1 /(1.0 + np.exp(-beta * (x_half - x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = yy + (0.05 * y - 0.01) * y_span # adjust vertical position
    ax.autoscale(False)
    ax.plot(x, y, color='black', lw=1)
    ax.text((xmax + xmin) / 2., yy + .07 * y_span, text, ha='center', va='bottom')
