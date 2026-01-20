import pandas as pd


def date_range(start, end=None, length=None, freq=None) -> pd.DatetimeIndex:
    if end is None and length is None and freq is None:
        raise ValueError(
            "At least one of `end`, `length` or `freq` must have a non-null value"
        )
    return pd.date_range(start=start, end=end, periods=length, freq=freq)
