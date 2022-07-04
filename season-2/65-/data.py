"""
Data Utilities | Cannlytics

Authors: Keegan Skeate <keegan@cannlytics.com>
Created: 4/21/2022
Updated: 4/24/2022
License: <https://github.com/cannlytics/cannlytics/blob/main/LICENSE>
"""
# Internal imports.
from datetime import datetime
import re
from typing import Any, List, Optional

# External imports.
from dateutil import relativedelta
from fredapi import Fred
from pandas import merge, NaT, to_datetime
from pandas.tseries.offsets import MonthEnd


#-----------------------------------------------------------------------
# Fed FRED utilities.
#-----------------------------------------------------------------------

def get_state_population(
        api_key: str,
        state: str,
        district: Optional[str] = '',
        obs_start: Optional[Any] = None,
        obs_end: Optional[Any] = None,
        multiplier: Optional[float] = 1000.0,
) -> List[int]:
    """Get a given state's population from the Fed Fred API."""
    fred = Fred(api_key=api_key)
    population = fred.get_series(f'{state}POP{district}', obs_start, obs_end)
    try:
        population = [int(x * multiplier) for x in population.values]
        if len(population) == 1:
            return population[0]
    except ValueError:
        pass
    return population


def get_state_current_population(state, api_key=None):
    """Get a given state's latest population from the Fed Fred API,
    getting the number in 1000's and returning the absolute value.
    Args:
        state (str): The state abbreviation for the state to retrieve population
            data. The abbreviation can be upper or lower case.
        api_key (str): A Fed FRED API key. You can sign up for a free API key at
            http://research.stlouisfed.org/fred2/. You can also pass `None`
            and set the environment variable 'FRED_API_KEY' to the value of
            your API key.
    Returns:
        (dict): Returns a dictionary with population values and source.
    """
    fred = Fred(api_key=api_key)
    state_code = state.upper()
    population_source_code = f'{state_code}POP'
    population = fred.get_series(population_source_code)
    real_population = int(population.iloc[-1] * 1000)
    population_date = population.index[-1].isoformat()[:10]
    return {
        'population': real_population,
        'population_formatted': f'{real_population:,}',
        'population_source_code': population_source_code,
        'population_source': f'https://fred.stlouisfed.org/series/{population_source_code}',
        'population_at': population_date,
    }


#-----------------------------------------------------------------------
# Time utilities.
#-----------------------------------------------------------------------

def convert_month_year_to_date(x):
    """Convert a month, year series to datetime. E.g. `'April 2022'`."""
    try:
        return datetime.strptime(x.replace('.0', ''), '%B %Y')
    except:
        return NaT


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


def end_of_period_timeseries(data: Any, period: Optional[str] = 'M') -> Any:
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


def months_elapsed(start, end):
    """Calculate the months elapsed between two datetimes,
    returning 0 if a negative time span.
    """
    diff = relativedelta.relativedelta(end, start)
    time_span = diff.months + diff.years * 12
    return time_span if time_span > 0 else 0


def reverse_dataframe(data: Any) -> Any:
    """Reverse the ordering of a DataFrame.
    Args:
        data (DataFrame): A DataFrame to re-order.
    Returns:
        (DataFrame): The re-ordered DataFrame.
    """
    return data[::-1].reset_index(drop=True)


def set_training_period(series: Any, date_start: str, date_end: str) -> Any:
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


#-----------------------------------------------------------------------
# String utilities.
#-----------------------------------------------------------------------

def format_billions(value: float, pos: Optional[int] = None) -> str: #pylint: disable=unused-argument
    """The two args are the value and tick position."""
    return '%1.1fB' % (value * 1e-9)


def format_millions(value: float, pos: Optional[int] = None) -> str: #pylint: disable=unused-argument
    """The two args are the value and tick position."""
    return '%1.1fM' % (value * 1e-6)


def format_thousands(value: float, pos: Optional[int] = None) -> str: #pylint: disable=unused-argument
    """The two args are the value and tick position."""
    return '%1.0fK' % (value * 1e-3)


def sorted_nicely(unsorted_list: List[str]) -> List[str]:
    """Sort the given iterable in the way that humans expect.
    Credit: Mark Byers <https://stackoverflow.com/a/2669120/5021266>
    License: CC BY-SA 2.5 <https://creativecommons.org/licenses/by-sa/2.5/>
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alpha = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(unsorted_list, key=alpha)


def sentence_case(s):
    """
    Author: Zizouz212 https://stackoverflow.com/a/39969233/5021266
    License: CC BY-SA 3.0 https://creativecommons.org/licenses/by-sa/3.0/
    """
    return '. '.join(i.capitalize() for i in s.split('. ')).strip()


#-----------------------------------------------------------------------
# DataFrame utilities.
#-----------------------------------------------------------------------

def rmerge(left, right, **kwargs):
    """Perform a merge using pandas with optional removal of overlapping
    column names not associated with the join.

    Though I suspect this does not adhere to the spirit of pandas merge
    command, I find it useful because re-executing IPython notebook cells
    containing a merge command does not result in the replacement of existing
    columns if the name of the resulting DataFrame is the same as one of the
    two merged DataFrames, i.e. data = pa.merge(data,new_dataframe). I prefer
    this command over pandas df.combine_first() method because it has more
    flexible join options.

    The column removal is controlled by the 'replace' flag which is
    'left' (default) or 'right' to remove overlapping columns in either the
    left or right DataFrame. If 'replace' is set to None, the default
    pandas behavior will be used. All other parameters are the same
    as pandas merge command.

    Author: Michelle Gill
    Source: https://gist.github.com/mlgill/11334821

    Examples
    --------
    >>> left       >>> right
       a  b   c       a  c   d
    0  1  4   9    0  1  7  13
    1  2  5  10    1  2  8  14
    2  3  6  11    2  3  9  15
    3  4  7  12

    >>> rmerge(left,right,on='a')
       a  b  c   d
    0  1  4  7  13
    1  2  5  8  14
    2  3  6  9  15

    >>> rmerge(left,right,on='a',how='left')
       a  b   c   d
    0  1  4   7  13
    1  2  5   8  14
    2  3  6   9  15
    3  4  7 NaN NaN

    >>> rmerge(left,right,on='a',how='left',replace='right')
       a  b   c   d
    0  1  4   9  13
    1  2  5  10  14
    2  3  6  11  15
    3  4  7  12 NaN

    >>> rmerge(left,right,on='a',how='left',replace=None)
       a  b  c_x  c_y   d
    0  1  4    9    7  13
    1  2  5   10    8  14
    2  3  6   11    9  15
    3  4  7   12  NaN NaN
    """

    # Function to flatten lists from http://rosettacode.org/wiki/Flatten_a_list#Python
    def flatten(lst):
        return sum(([x] if not isinstance(x, list) else flatten(x) for x in lst), [])

    # Set default for removing overlapping columns in "left" to be true
    myargs = {'replace':'left'}
    myargs.update(kwargs)

    # Remove the replace key from the argument dict to be sent to
    # pandas merge command
    kwargs = {k:v for k, v in myargs.items() if k != 'replace'}

    if myargs['replace'] is not None:
        # Generate a list of overlapping column names not associated with the join
        skipcols = set(flatten([v for k, v in myargs.items() if k in ['on', 'left_on', 'right_on']]))
        leftcols = set(left.columns)
        rightcols = set(right.columns)
        dropcols = list((leftcols & rightcols).difference(skipcols))

        # Remove the overlapping column names from the appropriate DataFrame
        if myargs['replace'].lower() == 'left':
            left = left.copy().drop(dropcols, axis=1)
        elif myargs['replace'].lower() == 'right':
            right = right.copy().drop(dropcols, axis=1)

    return merge(left, right, **kwargs)
