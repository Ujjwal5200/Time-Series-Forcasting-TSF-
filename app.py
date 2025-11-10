"""
Interactive Time Series Forecasting App

This Streamlit application provides an interactive interface for time series forecasting.
Users can upload CSV data or use sample datasets, visualize data with various plots,
select from multiple forecasting models (ARIMA, Prophet, Exponential Smoothing, VAR, Random Forest, LSTM),
configure model parameters, run forecasts, and view performance metrics.

Features:
- Data upload and automatic type detection (univariate, bivariate, multivariate)
- Multiple visualization options: Line Chart, Histogram, Box Plot, Scatter Plot, Autocorrelation, Heatmap, Seasonal Decomposition
- Model selection with detailed information and parameter tuning
- Forecasting with performance metrics (RMSE, MAE)
- Downloadable forecast reports

Author: [Your Name]
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils import detect_time_series_type, preprocess_data, load_sample_data
from visualizations import plot_line_chart, plot_histogram, plot_scatter, plot_autocorrelation, plot_heatmap, plot_box, plot_seasonal_decompose, plot_pacf, plot_stationarity, plot_data_summary
from models import forecast_arima, forecast_prophet, forecast_exponential_smoothing, forecast_var, forecast_rf, forecast_lstm, calculate_metrics, plot_forecast
import plotly.graph_objects as go

st.set_page_config(page_title="Time Series Forecasting App", layout="wide")

# Initialize session state variables to store data and model information across interactions
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_type' not in st.session_state:
    st.session_state.data_type = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'preprocessing_steps' not in st.session_state:
    st.session_state.preprocessing_steps = []

# Model info
model_info = {
    'ARIMA': {
        'description': 'AutoRegressive Integrated Moving Average',
        'when_to_use': 'For univariate time series with trends and seasonality.',
        'pros': 'Flexible, interpretable.',
        'cons': 'Requires stationarity, parameter tuning.',
        'preprocessing': 'Differencing for stationarity.'
    },
    'Prophet': {
        'description': 'Facebook\'s forecasting tool',
        'when_to_use': 'For univariate with holidays and changepoints.',
        'pros': 'Handles seasonality, outliers.',
        'cons': 'Less flexible for multivariate.',
        'preprocessing': 'None special.'
    },
    'Exponential Smoothing': {
        'description': 'Weighted averages for forecasting',
        'when_to_use': 'For univariate with trend and seasonality.',
        'pros': 'Simple, fast.',
        'cons': 'Assumes additive/multiplicative model.',
        'preprocessing': 'None.'
    },
    'VAR': {
        'description': 'Vector AutoRegression',
        'when_to_use': 'For multivariate time series.',
        'pros': 'Captures interdependencies.',
        'cons': 'Requires stationarity.',
        'preprocessing': 'Differencing.'
    },
    'Random Forest': {
        'description': 'Ensemble of decision trees',
        'when_to_use': 'For multivariate, non-linear relationships.',
        'pros': 'Handles non-linearity, robust.',
        'cons': 'Black box, may overfit.',
        'preprocessing': 'Normalization.'
    },
    'LSTM': {
        'description': 'Long Short-Term Memory neural network',
        'when_to_use': 'For complex patterns in time series, univariate or multivariate.',
        'pros': 'Captures long-term dependencies, flexible.',
        'cons': 'Requires more data, computationally intensive.',
        'preprocessing': 'Normalization.'
    }
}

st.title("Interactive Time Series Forecasting App")

# Create tabs for different sections of the app
tab1, tab2, tab3, tab4 = st.tabs(["Upload & Detect", "Visualize", "Model Selection", "Forecast & Results"])

# Tab 1: Data Upload and Type Detection
with tab1:
    st.header("Upload CSV or Use Sample Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    use_sample = st.selectbox("Or use sample data", [None, 'univariate', 'bivariate', 'multivariate'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df
        st.session_state.data_type = detect_time_series_type(df)
        st.success(f"Data uploaded. Detected type: {st.session_state.data_type}")
        st.dataframe(df.head())
    elif use_sample:
        samples = load_sample_data()
        st.session_state.data = samples[use_sample]
        st.session_state.data_type = use_sample
        st.success(f"Sample {use_sample} data loaded.")
        st.dataframe(st.session_state.data.head())

# Tab 2: Data Visualization
with tab2:
    if st.session_state.data is not None:
        st.header("Data Visualization")
        # Preprocess data for visualization: convert first column to datetime and set as index
        df_viz = st.session_state.data.copy()
        try:
            df_viz.iloc[:, 0] = pd.to_datetime(df_viz.iloc[:, 0], errors='coerce')
            df_viz = df_viz.set_index(df_viz.columns[0])
            # Check if index is datetime and there are numeric columns
            if not pd.api.types.is_datetime64_any_dtype(df_viz.index) or df_viz.select_dtypes(include=[np.number]).empty:
                st.error("Data must have a valid datetime column and at least one numeric column for visualization.")
                df_viz = None
        except Exception as e:
            st.error(f"Error preprocessing data for visualization: {e}")
            df_viz = None
        
        if df_viz is not None:
            viz_options = ['Line Chart', 'Histogram', 'Box Plot', 'Scatter Plot', 'Autocorrelation', 'PACF', 'Stationarity Test', 'Heatmap', 'Seasonal Decomposition', 'Data Summary']
            viz_option = st.selectbox("Choose visualization", viz_options)
            
            if viz_option == 'Line Chart':
                fig = plot_line_chart(df_viz)
                st.plotly_chart(fig)
            elif viz_option == 'Histogram':
                figs = plot_histogram(df_viz)
                for fig in figs:
                    st.plotly_chart(fig)
            elif viz_option == 'Scatter Plot' and st.session_state.data_type in ['bivariate', 'multivariate']:
                fig = plot_scatter(df_viz)
                if fig:
                    st.plotly_chart(fig)
            elif viz_option == 'Autocorrelation':
                plot_autocorrelation(df_viz)
            elif viz_option == 'PACF':
                plot_pacf(df_viz)
            elif viz_option == 'Stationarity Test':
                results = plot_stationarity(df_viz)
                for col, res in results.items():
                    st.write(f"Column: {col}")
                    if 'Error' in res:
                        st.write(f"Error: {res['Error']}")
                    else:
                        st.write(f"ADF Statistic: {res['ADF Statistic']:.4f}")
                        st.write(f"p-value: {res['p-value']:.4f}")
                        st.write(f"Critical Values: {res['Critical Values']}")
                        st.write(f"Stationary: {'Yes' if res['Stationary'] else 'No'}")
                    st.write("---")
            elif viz_option == 'Heatmap' and st.session_state.data_type == 'multivariate':
                fig = plot_heatmap(df_viz)
                st.plotly_chart(fig)
            elif viz_option == 'Box Plot':
                figs = plot_box(df_viz)
                for fig in figs:
                    st.plotly_chart(fig)
            elif viz_option == 'Seasonal Decomposition':
                period = st.slider("Select period for seasonal decomposition", min_value=2, max_value=60, value=12)
                fig = plot_seasonal_decompose(df_viz, period=period)
                if fig:
                    st.plotly_chart(fig)
                else:
                    st.warning("Not enough data or invalid data type for seasonal decomposition.")
            elif viz_option == 'Data Summary':
                summary = plot_data_summary(df_viz)
                st.write("Data Types:")
                for col, dtype in summary['dtypes'].items():
                    st.write(f"{col}: {dtype}")
                st.write("Descriptive Statistics:")
                for stat, values in summary['describe'].items():
                    st.write(f"**{stat}**")
                    for col, val in values.items():
                        st.write(f"{col}: {val}")
                    st.write("---")
        else:
            st.warning("Data preprocessing failed. Please check your data format.")
    else:
        st.warning("Please upload data first.")

# Tab 3: Model Selection and Parameter Configuration
with tab3:
    if st.session_state.data is not None:
        st.header("Model Selection")
        # Select appropriate models based on data type
        if st.session_state.data_type == 'univariate':
            models = ['ARIMA', 'Prophet', 'Exponential Smoothing', 'LSTM']
        elif st.session_state.data_type == 'bivariate':
            models = ['VAR', 'Random Forest', 'LSTM']  # Treat as multivariate
        else:
            models = ['VAR', 'Random Forest', 'LSTM']
        
        selected_model = st.selectbox("Select Model", models)
        
        # Display model information in an expandable section
        with st.expander("Model Information"):
            info = model_info[selected_model]
            st.write(f"**Description:** {info['description']}")
            st.write(f"**When to use:** {info['when_to_use']}")
            st.write(f"**Pros:** {info['pros']}")
            st.write(f"**Cons:** {info['cons']}")
            st.write(f"**Preprocessing:** {info['preprocessing']}")
        
        # Configure model parameters based on selected model
        params = {}
        if selected_model == 'ARIMA':
            params['p'] = st.slider("p (AR order)", 0, 5, 1)
            params['d'] = st.slider("d (Differencing)", 0, 2, 1)
            params['q'] = st.slider("q (MA order)", 0, 5, 1)
            params['steps'] = st.number_input("Forecast steps", 1, 50, 10)
        elif selected_model == 'Prophet':
            params['periods'] = st.number_input("Forecast periods", 1, 50, 10)
        elif selected_model == 'Exponential Smoothing':
            params['trend'] = st.selectbox("Trend", [None, 'add', 'mul'])
            params['seasonal'] = st.selectbox("Seasonal", [None, 'add', 'mul'])
            params['steps'] = st.number_input("Forecast steps", 1, 50, 10)
        elif selected_model == 'VAR':
            params['lags'] = st.slider("Lags", 1, 10, 1)
            params['steps'] = st.number_input("Forecast steps", 1, 50, 10)
        elif selected_model == 'Random Forest':
            params['lags'] = st.slider("Lags", 1, 10, 1)
            params['n_estimators'] = st.slider("N Estimators", 10, 200, 100)
        elif selected_model == 'LSTM':
            params['look_back'] = st.slider("Look Back", 5, 50, 10)
            params['units'] = st.slider("Units", 10, 200, 50)
            params['epochs'] = st.slider("Epochs", 10, 100, 20)
            params['steps'] = st.number_input("Forecast steps", 1, 50, 10)

        # Store selected model and parameters in session state
        st.session_state.selected_model = selected_model
        st.session_state.params = params
    else:
        st.warning("Please upload data first.")

with tab4:
    if st.session_state.data is not None and 'selected_model' in st.session_state:
        st.header("Forecast & Results")
        if st.button("Run Forecast"):
            # Preprocess
            preprocessed, steps = preprocess_data(st.session_state.data.copy(), st.session_state.selected_model)
            st.session_state.preprocessed_data = preprocessed
            st.session_state.preprocessing_steps = steps
            
            st.subheader("Preprocessing Steps")
            for step in steps:
                st.write(f"- {step}")
            
            # Forecast
            if st.session_state.selected_model == 'ARIMA':
                forecast, model = forecast_arima(preprocessed.iloc[:, 0], st.session_state.params)
            elif st.session_state.selected_model == 'Prophet':
                forecast, model = forecast_prophet(preprocessed.iloc[:, 0], st.session_state.params)
            elif st.session_state.selected_model == 'Exponential Smoothing':
                forecast, model = forecast_exponential_smoothing(preprocessed.iloc[:, 0], st.session_state.params)
            elif st.session_state.selected_model == 'VAR':
                forecast, model = forecast_var(preprocessed, st.session_state.params)
            elif st.session_state.selected_model == 'Random Forest':
                forecast, model = forecast_rf(preprocessed, st.session_state.params)
            elif st.session_state.selected_model == 'LSTM':
                if st.session_state.data_type == 'univariate':
                    forecast, model = forecast_lstm(preprocessed.iloc[:, 0], st.session_state.params)
                else:
                    forecast, model = forecast_lstm(preprocessed, st.session_state.params)
            
            # Metrics (dummy, since no actual future data)
            # For demo, use last part as test
            actual = preprocessed.iloc[-len(forecast):] if len(forecast) <= len(preprocessed) else preprocessed
            actual_flat = actual.values.flatten()
            forecast_flat = forecast.values.flatten()
            min_len = min(len(actual_flat), len(forecast_flat))
            rmse, mae = calculate_metrics(actual_flat[:min_len], forecast_flat[:min_len])
            
            st.subheader("Performance Metrics")
            st.write(f"RMSE: {rmse:.2f}")
            st.write(f"MAE: {mae:.2f}")
            
            # Plot
            fig = plot_forecast(actual.iloc[:, 0] if st.session_state.data_type == 'univariate' else actual, forecast, "Actual vs Predicted")
            st.plotly_chart(fig)
            
            # Download
            st.subheader("Download Report")
            report = f"Model: {st.session_state.selected_model}\nParams: {st.session_state.params}\nPreprocessing: {steps}\nRMSE: {rmse}\nMAE: {mae}"
            st.download_button("Download Report", report, "report.txt")
            # For plots, could save fig as image, but for simplicity, skip
    else:
        st.warning("Please select a model first.")
