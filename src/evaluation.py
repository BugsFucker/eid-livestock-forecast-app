import pandas as pd
from .data_prep import load_raw, clean_and_parse_dates, filter_reasonable_years
from .features import add_calendar_features


def describe_dataset():
    df = load_raw()
    df = clean_and_parse_dates(df)
    df = filter_reasonable_years(df)
    df = add_calendar_features(df)
    print(df.describe(include='all'))


if __name__ == '__main__':
    describe_dataset()
