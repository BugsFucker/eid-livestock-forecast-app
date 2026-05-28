import pandas as pd
from dateutil import parser
from .config import DATA_PATH


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def clean_and_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    # price like `€531.00` -> float per 100kg
    df = df.copy()
    df['price_numeric'] = (
        df['price']
        .astype(str)
        .str.replace('€','', regex=False)
        .str.replace(',','', regex=False)
        .astype(float)
    )
    # parse beginDate as representative week date
    df['beginDate'] = pd.to_datetime(df['beginDate'], format='%d/%m/%Y')
    df['endDate'] = pd.to_datetime(df['endDate'], format='%d/%m/%Y')
    df['week_date'] = df['beginDate'] + (df['endDate'] - df['beginDate'])/2
    df['year'] = df['week_date'].dt.year
    df['week'] = df['week_date'].dt.isocalendar().week.astype(int)
    return df


def filter_reasonable_years(df: pd.DataFrame, min_year: int = 2015) -> pd.DataFrame:
    return df.query('year >= @min_year').reset_index(drop=True)


if __name__ == '__main__':
    df = load_raw()
    df = clean_and_parse_dates(df)
    df = filter_reasonable_years(df)
    print(df.head())
