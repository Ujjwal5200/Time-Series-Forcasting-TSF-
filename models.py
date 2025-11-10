import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from statsmodels.tsa.api import VAR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def forecast_arima(data, params):
    """
    ARIMA forecasting for univariate.
    """
    try:
        if not isinstance(data, (pd.Series, np.ndarray)) or len(data) < 10:
            raise ValueError("Data must be a pandas Series or numpy array with at least 10 observations.")
        if not pd.api.types.is_numeric_dtype(data):
            raise ValueError("Data must be numeric.")
        if not all(k in params for k in ['p', 'd', 'q', 'steps']):
            raise ValueError("Missing required parameters: p, d, q, steps.")
        if params['steps'] <= 0 or params['steps'] > 100:
            raise ValueError("Steps must be between 1 and 100.")
        model = ARIMA(data, order=(params['p'], params['d'], params['q']))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=params['steps'])
        return forecast, model_fit
    except Exception as e:
        raise RuntimeError(f"ARIMA forecasting failed: {e}")

def forecast_prophet(data, params):
    """
    Prophet forecasting for univariate.
    """
    try:
        if not isinstance(data, pd.Series) or len(data) < 10:
            raise ValueError("Data must be a pandas Series with at least 10 observations.")
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            raise ValueError("Data index must be datetime.")
        if not pd.api.types.is_numeric_dtype(data):
            raise ValueError("Data must be numeric.")
        if 'periods' not in params or params['periods'] <= 0 or params['periods'] > 100:
            raise ValueError("Periods must be between 1 and 100.")
        df = data.reset_index()
        df.columns = ['ds', 'y']
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=params['periods'])
        forecast = model.predict(future)
        return forecast['yhat'], model
    except Exception as e:
        raise RuntimeError(f"Prophet forecasting failed: {e}")

def forecast_exponential_smoothing(data, params):
    """
    Exponential Smoothing for univariate.
    """
    try:
        if not isinstance(data, (pd.Series, np.ndarray)) or len(data) < 10:
            raise ValueError("Data must be a pandas Series or numpy array with at least 10 observations.")
        if not pd.api.types.is_numeric_dtype(data):
            raise ValueError("Data must be numeric.")
        if 'steps' not in params or params['steps'] <= 0 or params['steps'] > 100:
            raise ValueError("Steps must be between 1 and 100.")
        model = ExponentialSmoothing(data, trend=params['trend'], seasonal=params['seasonal'])
        model_fit = model.fit()
        forecast = model_fit.forecast(params['steps'])
        return forecast, model_fit
    except Exception as e:
        raise RuntimeError(f"Exponential Smoothing forecasting failed: {e}")

def forecast_var(data, params):
    """
    VAR for multivariate.
    """
    try:
        if not isinstance(data, pd.DataFrame) or data.shape[1] < 2:
            raise ValueError("Data must be a pandas DataFrame with at least 2 columns.")
        if len(data) < 20:
            raise ValueError("Data must have at least 20 observations.")
        if not all(pd.api.types.is_numeric_dtype(data[col]) for col in data.columns):
            raise ValueError("All columns must be numeric.")
        if not all(k in params for k in ['lags', 'steps']):
            raise ValueError("Missing required parameters: lags, steps.")
        if params['lags'] <= 0 or params['lags'] >= len(data) // 2:
            raise ValueError("Lags must be positive and less than half the data length.")
        if params['steps'] <= 0 or params['steps'] > 100:
            raise ValueError("Steps must be between 1 and 100.")
        model = VAR(data)
        model_fit = model.fit(maxlags=params['lags'])
        forecast = model_fit.forecast(data.values[-params['lags']:], steps=params['steps'])
        forecast_df = pd.DataFrame(forecast, columns=data.columns)
        return forecast_df, model_fit
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"VAR forecasting failed due to numerical instability (e.g., singular matrix). Try using a different model, reducing lags, or ensuring data is stationary and has sufficient observations.")
    except Exception as e:
        raise RuntimeError(f"VAR forecasting failed: {e}")

def forecast_rf(data, params):
    """
    Random Forest for multivariate.
    """
    try:
        if not isinstance(data, pd.DataFrame) or data.shape[1] < 2:
            raise ValueError("Data must be a pandas DataFrame with at least 2 columns.")
        if len(data) < 20:
            raise ValueError("Data must have at least 20 observations.")
        if not all(pd.api.types.is_numeric_dtype(data[col]) for col in data.columns):
            raise ValueError("All columns must be numeric.")
        if not all(k in params for k in ['lags', 'n_estimators']):
            raise ValueError("Missing required parameters: lags, n_estimators.")
        if params['lags'] <= 0 or params['lags'] >= len(data) // 2:
            raise ValueError("Lags must be positive and less than half the data length.")
        if params['n_estimators'] <= 0 or params['n_estimators'] > 500:
            raise ValueError("N Estimators must be between 1 and 500.")
        # Simple implementation: predict next value based on lagged features
        df = data.copy()
        for col in df.columns:
            for lag in range(1, params['lags']+1):
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        df = df.dropna()
        if df.empty:
            raise ValueError("Not enough data after lagging.")
        X = df.drop(df.columns[:len(data.columns)], axis=1)
        y = df[data.columns]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestRegressor(n_estimators=params['n_estimators'])
        model.fit(X_train, y_train)
        forecast = model.predict(X_test)
        return pd.DataFrame(forecast, columns=data.columns), model
    except Exception as e:
        raise RuntimeError(f"Random Forest forecasting failed: {e}")

def calculate_metrics(actual, predicted):
    """
    Calculate RMSE and MAE.
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    return rmse, mae

def forecast_lstm(data, params):
    """
    LSTM forecasting for univariate or multivariate.
    """
    try:
        if isinstance(data, pd.DataFrame):
            # Ensure only numeric columns
            data = data.select_dtypes(include=[np.number])
            if data.empty:
                raise ValueError("No numeric columns in data.")
            columns = data.columns
            data_values = data.values.astype(np.float32)
        elif isinstance(data, pd.Series):
            if not pd.api.types.is_numeric_dtype(data):
                raise ValueError("Data must be numeric.")
            columns = data.name
            data_values = data.values.reshape(-1, 1).astype(np.float32)
        else:
            data_values = np.asarray(data, dtype=np.float32)
            columns = None

        if len(data_values) < 20:
            raise ValueError("Data must have at least 20 observations.")
        if not all(k in params for k in ['look_back', 'units', 'epochs', 'steps']):
            raise ValueError("Missing required parameters: look_back, units, epochs, steps.")
        look_back = params['look_back']
        steps = params['steps']
        if look_back <= 0 or look_back >= len(data_values) // 2:
            raise ValueError("Look back must be positive and less than half the data length.")
        if params['units'] <= 0 or params['units'] > 500:
            raise ValueError("Units must be between 1 and 500.")
        if params['epochs'] <= 0 or params['epochs'] > 200:
            raise ValueError("Epochs must be between 1 and 200.")
        if steps <= 0 or steps > 100:
            raise ValueError("Steps must be between 1 and 100.")

        # Prepare data for LSTM
        X, y = [], []
        for i in range(len(data_values) - look_back - steps + 1):
            X.append(data_values[i:i+look_back])
            y.append(data_values[i+look_back:i+look_back+steps].flatten())

        if not X:
            raise ValueError("Not enough data for LSTM training with given parameters.")

        X = np.array(X)
        y = np.array(y)

        # Build model
        model = Sequential()
        model.add(LSTM(params['units'], input_shape=(look_back, data_values.shape[1])))
        model.add(Dense(steps * data_values.shape[1]))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, y, epochs=params['epochs'], batch_size=1, verbose=0)

        # Forecast
        last = data_values[-look_back:]
        forecast = model.predict(last.reshape(1, look_back, data_values.shape[1]), verbose=0)
        forecast = forecast.reshape(steps, data_values.shape[1])

        if data_values.shape[1] == 1:
            return pd.Series(forecast.flatten()), model
        else:
            return pd.DataFrame(forecast, columns=columns), model
    except Exception as e:
        raise RuntimeError(f"LSTM forecasting failed: {e}")

def plot_forecast(actual, forecast, title):
    """
    Plot actual vs predicted.
    """
    fig = go.Figure()
    if isinstance(actual, pd.Series):
        fig.add_trace(go.Scatter(x=actual.index, y=actual.values, mode='lines', name='Actual'))
        # Assume daily frequency for forecast index
        forecast_index = pd.date_range(start=actual.index[-1], periods=len(forecast)+1, freq='D')[1:]
        fig.add_trace(go.Scatter(x=forecast_index, y=forecast.values, mode='lines', name='Forecast'))
        fig.add_vline(x=actual.index[-1], line_dash="dash", line_color="red")
    else:  # DataFrame for multivariate
        for col in actual.columns:
            fig.add_trace(go.Scatter(x=actual.index, y=actual[col], mode='lines', name=f'Actual {col}'))
            forecast_index = pd.date_range(start=actual.index[-1], periods=len(forecast)+1, freq='D')[1:]
            fig.add_trace(go.Scatter(x=forecast_index, y=forecast[col], mode='lines', name=f'Forecast {col}'))
        fig.add_vline(x=actual.index[-1], line_dash="dash", line_color="red")
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Value")
    return fig
