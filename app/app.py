import datetime as dt
from pathlib import Path

import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data_prep import load_raw, clean_and_parse_dates, filter_reasonable_years
from src.features import add_calendar_features, add_lag_features
from src.config import EID_DATES, MODEL_DIR


@st.cache_data
def load_dataset():
    df = load_raw()
    df = clean_and_parse_dates(df)
    df = filter_reasonable_years(df)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    # drop rows without lags
    df = df.dropna(subset=['price_numeric_lag_1', 'price_numeric_lag_2', 'price_numeric_lag_4'])
    return df


from src.modeling import train_model  # add near the top if not already imported

@st.cache_resource
def load_model():
    model_path = Path(MODEL_DIR) / 'rf_price_forecast.joblib'
    if not model_path.exists():
        # Train and save the model if it doesn't exist yet
        train_model()
    return joblib.load(model_path)

def main():
    st.set_page_config(page_title='Eid al-Adha Livestock Price Forecast', layout='wide')
    st.title('Eid al-Adha Livestock Price Forecast')

    df = load_dataset()
    model = load_model()

    st.sidebar.header('Filters')

    countries = sorted(df['memberStateCode'].unique())
    country = st.sidebar.selectbox('Country', countries)

    available_categories = sorted(
        df.loc[df['memberStateCode'] == country, 'category'].unique()
    )
    category = st.sidebar.selectbox('Category', available_categories)

    country_df = df[(df['memberStateCode'] == country) & (df['category'] == category)].copy()
    country_df = country_df.sort_values('week_date')

    st.subheader('Historical prices')
    st.line_chart(
        country_df.set_index('week_date')['price_numeric'],
        height=300,
    )

    st.subheader('Forecast next 8 weeks')
    # Use last available row as base and roll forward
    last_row = country_df.iloc[-1].copy()
    future_rows = []

    for i in range(1, 9):
        future = last_row.copy()
        future['week'] = int(last_row['week'] + i)
        future['week_date'] = last_row['week_date'] + pd.Timedelta(weeks=i)
        future['month'] = future['week_date'].month
        future['dayofyear'] = future['week_date'].timetuple().tm_yday
        eid = EID_DATES.get(future['year'])
        if eid is not None:
            future['days_to_eid'] = (eid - future['week_date'].date()).days
        else:
            future['days_to_eid'] = np.nan
        # keep lags from last_row
        features = future[['week', 'month', 'dayofyear', 'days_to_eid',
                           'price_numeric_lag_1', 'price_numeric_lag_2', 'price_numeric_lag_4',
                           'memberStateCode', 'category']]
        pred = model.predict(features.to_frame().T)[0]
        future['forecast_price'] = pred
        future_rows.append(future[['week_date', 'forecast_price']])

    future_df = pd.DataFrame(future_rows)
    future_df = future_df.set_index('week_date')
    st.line_chart(future_df['forecast_price'], height=300)

    st.markdown('---')
    st.caption('Data source: EU Agri-food Data Portal (sheep and goat meat prices).')


if __name__ == '__main__':
    main()
