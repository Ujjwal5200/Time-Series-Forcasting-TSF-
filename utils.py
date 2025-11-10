import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import streamlit as st

def detect_time_series_type(df):
    """
    Detect if the time series is univariate, bivariate, or multivariate.
    Assumes first column is time, others are variables.
    """
    if len(df.columns) == 2:
        return "univariate"
    elif len(df.columns) == 3:
        return "bivariate"
    else:
        return "multivariate"

def preprocess_data(df, model_type):
    """
    Preprocess data based on model requirements.
    """
    steps = []
    # Assume first column is datetime
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index(df.columns[0])

    # Handle missing values
    if df.isnull().sum().sum() > 0:
        df = df.fillna(method='ffill').fillna(method='bfill')
        steps.append("Filled missing values using forward and backward fill.")

    # Stationarity for ARIMA
    if model_type in ['ARIMA', 'VAR']:
        for col in df.columns:
            if not is_stationary(df[col]):
                df[col] = df[col].diff().dropna()
                steps.append(f"Applied differencing to {col} for stationarity.")

    # Normalization for LSTM, RF
    if model_type in ['LSTM', 'Random Forest']:
        scaler = MinMaxScaler()
        # Only scale numeric columns, exclude datetime index
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        steps.append("Applied Min-Max normalization.")

    return df, steps

def is_stationary(series):
    """
    Check if series is stationary using ADF test.
    """
    result = adfuller(series.dropna())
    return result[1] < 0.05

def load_sample_data():
    """
    Load sample datasets.
    """
    # For simplicity, create dummy data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    uni = pd.DataFrame({'date': dates, 'value': np.random.randn(100).cumsum()})
    bi = pd.DataFrame({'date': dates, 'value1': np.random.randn(100).cumsum(), 'value2': np.random.randn(100).cumsum()})
    multi = pd.DataFrame({'date': dates, 'value1': np.random.randn(100).cumsum(), 'value2': np.random.randn(100).cumsum(), 'value3': np.random.randn(100).cumsum()})
    return {'univariate': uni, 'bivariate': bi, 'multivariate': multi}
