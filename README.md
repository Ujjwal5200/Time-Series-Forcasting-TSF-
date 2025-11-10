# Interactive Time Series Forecasting App

## Overview

The **Interactive Time Series Forecasting App** is a user-friendly web application built with Streamlit that empowers users to perform advanced time series forecasting without requiring deep expertise in machine learning or data science. Whether you're a business analyst, researcher, student, or hobbyist, this app simplifies the process of analyzing historical data and predicting future trends.

### What It Does
This app allows users to:
- Upload their own CSV datasets or use built-in sample data.
- Automatically detect the type of time series (univariate, bivariate, or multivariate).
- Visualize data through interactive charts and statistical plots to gain insights.
- Select from multiple state-of-the-art forecasting models and tune their parameters.
- Generate forecasts and evaluate performance with metrics like RMSE and MAE.
- Download forecast reports for further analysis.

### Value It Provides
- **Accessibility**: Democratizes time series forecasting by providing an intuitive interface that handles complex preprocessing, model training, and evaluation behind the scenes.
- **Efficiency**: Saves time by automating data preparation, model selection, and visualization, allowing users to focus on interpreting results rather than coding.
- **Versatility**: Supports various data types and models, making it suitable for diverse applications like sales forecasting, stock price prediction, weather modeling, and more.
- **Educational**: Serves as a learning tool for understanding time series concepts, model differences, and the impact of parameters on forecasts.
- **Practical**: Enables data-driven decision-making with downloadable reports and interactive visualizations.

### Why It's Important
Time series forecasting is crucial in today's data-driven world for predicting future events based on historical patterns. From optimizing inventory in retail to anticipating market trends in finance, accurate forecasts drive better business outcomes. However, traditional forecasting requires specialized skills in statistics and programming, limiting its use. This app bridges that gap by making powerful forecasting tools accessible to everyone, fostering innovation and informed decision-making across industries.

### Relevance Today
In an era of big data and AI, the ability to forecast trends is more relevant than ever. With increasing data availability from IoT devices, social media, and sensors, organizations need tools to extract actionable insights quickly. This app addresses the growing demand for user-friendly analytics platforms, especially as remote work and digital transformation accelerate the need for self-service data tools. It's particularly valuable for small businesses, startups, and educational institutions that lack dedicated data science teams.

## Features

- **Data Upload & Type Detection**: Upload CSV files or use sample datasets. Automatically identifies univariate, bivariate, or multivariate time series.
- **Interactive Visualizations**: Explore data with line charts, histograms, box plots, scatter plots, autocorrelation, PACF, stationarity tests, heatmaps, seasonal decomposition, and data summaries.
- **Model Selection & Tuning**: Choose from 6 forecasting models with detailed descriptions, pros/cons, and customizable parameters.
- **Forecasting & Evaluation**: Run forecasts, view performance metrics (RMSE, MAE), and visualize actual vs. predicted values.
- **Preprocessing Automation**: Handles missing values, stationarity checks, differencing, and normalization automatically based on the selected model.
- **Downloadable Reports**: Export forecast results and model details for offline use.
- **Session State Management**: Maintains data and selections across app interactions for a seamless experience.


   ```



## Usage

1. **Upload Data**: In the "Upload & Detect" tab, upload a CSV file (first column should be dates) or select a sample dataset.
2. **Visualize**: Explore your data in the "Visualize" tab using various plots to understand patterns, seasonality, and stationarity.
3. **Select Model**: In the "Model Selection" tab, choose a forecasting model based on your data type and adjust parameters as needed.
4. **Forecast**: Click "Run Forecast" in the "Forecast & Results" tab to generate predictions, view metrics, and download reports.

### Data Format
- CSV files with the first column as datetime (e.g., YYYY-MM-DD).
- Subsequent columns as numeric time series variables.
- Supported types: Univariate (1 variable), Bivariate (2 variables), Multivariate (3+ variables).

## Supported Models

| Model | Type | Best For | Key Parameters |
|-------|------|----------|----------------|
| **ARIMA** | Univariate | Trends & Seasonality | p (AR order), d (Differencing), q (MA order) |
| **Prophet** | Univariate | Holidays & Changepoints | Forecast periods |
| **Exponential Smoothing** | Univariate | Trend & Seasonality | Trend type, Seasonal type |
| **VAR** | Multivariate | Interdependent Variables | Lags |
| **Random Forest** | Multivariate | Non-linear Relationships | Lags, Number of Estimators |
| **LSTM** | Uni/Multivariate | Complex Patterns | Look-back window, Units, Epochs |

Each model includes detailed information on when to use it, advantages, disadvantages, and preprocessing requirements.

## Visualizations

- **Line Chart**: Time series trends over time.
- **Histogram**: Distribution of values.
- **Box Plot**: Outlier detection.
- **Scatter Plot**: Relationships between variables (bivariate/multivariate).
- **Autocorrelation & PACF**: Lag dependencies.
- **Stationarity Test**: ADF test results for each variable.
- **Heatmap**: Correlation matrix (multivariate).
- **Seasonal Decomposition**: Trend, seasonal, and residual components.
- **Data Summary**: Descriptive statistics and data types.

## How It Was Built

This app was developed using Python and leverages several powerful libraries:

- **Streamlit**: For the interactive web interface.
- **Pandas & NumPy**: Data manipulation and numerical operations.
- **Plotly & Matplotlib**: Interactive and static visualizations.
- **Statsmodels**: Statistical models like ARIMA, VAR, and Exponential Smoothing.
- **Prophet**: Facebook's forecasting library for univariate series.
- **Scikit-learn**: Random Forest implementation and metrics.
- **TensorFlow/Keras**: LSTM neural network for deep learning-based forecasting.

The architecture is modular, with separate files for models (`models.py`), utilities (`utils.py`), and visualizations (`visualizations.py`). Error handling ensures robustness, and preprocessing is automated based on model requirements. The app uses Streamlit's session state to maintain user interactions seamlessly.

## Contributing

Contributions are welcome! If you'd like to improve the app, add new models, or enhance visualizations:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Make your changes and test thoroughly.
4. Submit a pull request with a clear description.

Please ensure code follows PEP 8 standards and includes appropriate error handling.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Empower your forecasting journey with the Interactive Time Series Forecasting App – where data meets insights effortlessly.*
