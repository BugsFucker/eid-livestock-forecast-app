import pandas as pd
from .config import EID_DATES


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['month'] = df['week_date'].dt.month
    df['dayofyear'] = df['week_date'].dt.dayofyear
    df['is_eid_year'] = df['year'].isin(EID_DATES.keys()).astype(int)

    # days_to_eid: difference between week_date and Eid date for that year (if known)
    def _days_to_eid(row):
        eid = EID_DATES.get(row['year'])
        if eid is None:
            return None
        return (eid - row['week_date'].date()).days

    df['days_to_eid'] = df.apply(_days_to_eid, axis=1)
    df['eid_season'] = df['days_to_eid'].between(-7, 21, inclusive='both')
    return df


def add_lag_features(df: pd.DataFrame, group_cols=None, target_col='price_numeric', lags=(1, 2, 4)) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ['memberStateCode', 'category']
    df = df.sort_values(group_cols + ['week_date']).copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = (
            df.groupby(group_cols)[target_col]
              .shift(lag)
        )
    return df


if __name__ == '__main__':
    from .data_prep import load_raw, clean_and_parse_dates, filter_reasonable_years
    df = load_raw()
    df = clean_and_parse_dates(df)
    df = filter_reasonable_years(df)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    print(df.head())
