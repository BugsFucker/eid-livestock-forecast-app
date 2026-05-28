import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from .config import MODEL_DIR
from .data_prep import load_raw, clean_and_parse_dates, filter_reasonable_years
from .features import add_calendar_features, add_lag_features


def build_dataset() -> pd.DataFrame:
    df = load_raw()
    df = clean_and_parse_dates(df)
    df = filter_reasonable_years(df, min_year=2015)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    # drop rows with missing lags
    df = df.dropna(subset=['price_numeric_lag_1', 'price_numeric_lag_2', 'price_numeric_lag_4'])
    return df


def get_feature_target(df: pd.DataFrame):
    target = df['price_numeric']
    num_features = ['week', 'month', 'dayofyear', 'days_to_eid',
                    'price_numeric_lag_1', 'price_numeric_lag_2', 'price_numeric_lag_4']
    cat_features = ['memberStateCode', 'category']

    X = df[num_features + cat_features]
    return X, target, num_features, cat_features


def train_model(random_state: int = 42):
    df = build_dataset()
    X, y, num_features, cat_features = get_feature_target(df)

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    cat_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features),
    ])

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = Path(MODEL_DIR) / 'rf_price_forecast.joblib'
    joblib.dump(pipe, model_path)
    print(f"Saved model to {model_path}")


if __name__ == '__main__':
    train_model()
