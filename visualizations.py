import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf as pacf_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def plot_line_chart(df):
    """
    Line chart for time series.
    """
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
    fig.update_layout(title="Line Chart")
    return fig

def plot_histogram(df):
    """
    Histograms for each column.
    """
    figs = []
    for col in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[col], name=col))
        fig.update_layout(title=f"Histogram of {col}")
        figs.append(fig)
    return figs

def plot_scatter(df):
    """
    Scatter plots for bivariate/multivariate.
    """
    if len(df.columns) >= 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.iloc[:, 0], y=df.iloc[:, 1], mode='markers'))
        fig.update_layout(title="Scatter Plot")
        return fig
    return None

def plot_autocorrelation(df):
    """
    Autocorrelation plot.
    """
    fig, ax = plt.subplots()
    plot_acf(df.iloc[:, 0], ax=ax)
    st.pyplot(fig)

def plot_heatmap(df):
    """
    Correlation heatmap for multivariate.
    """
    corr = df.corr()
    fig = px.imshow(corr, title="Correlation Heatmap")
    return fig

def plot_box(df):
    """
    Box plots for each column to detect outliers.
    """
    figs = []
    for col in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Box(y=df[col], name=col))
        fig.update_layout(title=f"Box Plot for {col}")
        figs.append(fig)
    return figs

def plot_seasonal_decompose(df, period=12):
    """
    Seasonal decomposition for univariate.
    """
    if len(df) > 2 * period and len(df.columns) > 0 and pd.api.types.is_numeric_dtype(df.iloc[:, 0]):  # Need enough data and numeric series
        try:
            decomposition = seasonal_decompose(df.iloc[:, 0], model='additive', period=period)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=decomposition.trend, name='Trend'))
            fig.add_trace(go.Scatter(x=df.index, y=decomposition.seasonal, name='Seasonal'))
            fig.add_trace(go.Scatter(x=df.index, y=decomposition.resid, name='Residual'))
            fig.update_layout(title=f"Seasonal Decomposition (Period={period})")
            return fig
        except Exception as e:
            st.warning(f"Seasonal decomposition failed: {e}")
            return None
    return None

def plot_pacf(df):
    """
    Partial Autocorrelation plot.
    """
    if len(df.columns) > 0 and pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
        fig, ax = plt.subplots()
        pacf_plot(df.iloc[:, 0], ax=ax)
        st.pyplot(fig)

def plot_stationarity(df):
    """
    Stationarity test results using ADF test.
    """
    results = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                result = adfuller(df[col].dropna())
                results[col] = {
                    'ADF Statistic': result[0],
                    'p-value': result[1],
                    'Critical Values': result[4],
                    'Stationary': result[1] < 0.05
                }
            except Exception as e:
                results[col] = {'Error': str(e)}
        else:
            results[col] = {'Error': 'Non-numeric column'}
    return results

def plot_data_summary(df):
    """
    Data summary: descriptive stats and dtypes.
    """
    summary = {
        'dtypes': df.dtypes.to_dict(),
        'describe': df.describe().to_dict()
    }
    return summary
