
from typing import Optional, List

from cannlytics.data import create_hash
import pandas as pd


def anonymize(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        suffix: str = '_by',
    ) -> pd.DataFrame:
    """Anonymize a dataset by creating a hash for fields that end in "_by" or "_By."""
    if columns is None:
        suffix_pattern = f'.*{suffix}$'
        columns = df.filter(regex=suffix_pattern, axis=1, case=False).columns
    df.loc[:, columns] = df.loc[:, columns].astype(str).applymap(create_hash)
    return df



# Anonymize lab employee names.


# Calculate turnaround time by lab in CT.
