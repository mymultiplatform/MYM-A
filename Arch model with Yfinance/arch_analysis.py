import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import yfinance as yf
from datetime import datetime, timedelta
import sys
import io
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.graphics.tsaplots import plot_pacf as statsmodels_plot_pacf

sys.setrecursionlimit(10000)  # Increase the limit to a higher value

def fetch_yfinance_data(symbol, start_date, end_date):
    """Fetch data from yfinance for a specific date range"""
    print(f"Attempting to fetch {symbol} data from {start_date} to {end_date}.")
    data = yf.download(symbol, start=start_date, end=end_date)
    if data is None or len(data) == 0:
        print(f"Failed to fetch data for {symbol}")
        return None
    print(f"Successfully fetched {len(data)} bars of data.")
    return data

def calculate_returns(prices):
    """Calculate log returns"""
    return np.log(prices / prices.shift(1)).dropna()

def plot_data(series, title):
    """Plot time series data"""
    plt.figure(figsize=(10,4))
    plt.plot(series)
    plt.title(title, fontsize=20)
    plt.show()

def plot_pacf(series):
    """Plot PACF of squared returns"""
    plt.figure(figsize=(10, 4))
    statsmodels_plot_pacf(np.array(series)**2, lags=40)
    plt.title("PACF of Squared Returns")
    plt.show()

def fit_arch_model(returns, p=1):
    """Fit ARCH(p) model"""
    model = arch_model(returns, vol='ARCH', p=p)
    return model.fit(disp='off')

def forecast_volatility(model_results, horizon=1):
    """Forecast volatility"""
    forecast = model_results.forecast(horizon=horizon)
    return np.sqrt(forecast.variance.values[-1, :])

def perform_arch_analysis(returns, test_size=100):
    # Convert returns to pandas Series if it's not already
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    # Split data into train and test sets
    train = returns[:-test_size]
    test = returns[-test_size:]

    # Fit ARCH(1) model on train data only
    model = arch_model(train, vol='ARCH', p=1)
    model_fit = model.fit(disp='off')

    # Make predictions on test data
    forecasts = model_fit.forecast(horizon=len(test))
    forecast_variance = np.sqrt(forecasts.variance.values[-1, :])

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(test.index, test, label='Actual Returns')
    plt.plot(test.index, forecast_variance, label='Predicted Volatility')
    plt.title('ARCH(1) Volatility Forecast')
    plt.legend()
    plt.show()

    print(model_fit.summary())

    return model_fit, forecasts

def tune_arch_parameters(returns, p_range=(1, 5), num_splits=5):
    """Tune ARCH model parameters using cross-validation"""
    best_aic = np.inf
    best_p = None
    tscv = TimeSeriesSplit(n_splits=num_splits)

    for p in range(p_range[0], p_range[1] + 1):
        aic_scores = []
        for train_index, val_index in tscv.split(returns):
            train, val = returns.iloc[train_index], returns.iloc[val_index]
            model = arch_model(train, vol='ARCH', p=p)
            results = model.fit(disp='off')
            aic_scores.append(results.aic)
        avg_aic = np.mean(aic_scores)
        if avg_aic < best_aic:
            best_aic = avg_aic
            best_p = p

    print(f"Best ARCH parameter: p={best_p}")
    return best_p

def perform_rolling_volatility_forecast(returns, test_size=100, p=1):
    """Perform rolling volatility forecast"""
    rolling_predictions = []
    for i in range(test_size):
        train = returns[:-(test_size-i)]
        model = arch_model(train, vol='ARCH', p=p)
        model_fit = model.fit(disp='off')
        pred = model_fit.forecast(horizon=1)
        rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
    
    return rolling_predictions

def arch_analysis(data):
    """Main function to run ARCH analysis with rolling 5-day forecast"""
    # Calculate returns for all data
    returns = calculate_returns(data['Close'])
    
    # Split into train and test
    train_end = datetime(2023, 12, 31)
    train_returns = returns[returns.index <= train_end]
    test_returns = returns[returns.index > train_end]
    
    # Tune ARCH parameters using training data
    best_p = tune_arch_parameters(train_returns)
    
    # Perform rolling forecast
    forecast_horizon = 5  # Forecast 5 steps ahead
    rolling_forecasts = []
    actual_next_5_days = []
    
    for i in range(0, len(test_returns) - forecast_horizon, forecast_horizon):
        train = returns[:len(train_returns) + i]
        model = arch_model(train, vol='ARCH', p=best_p)
        model_fit = model.fit(disp='off')
        forecast = model_fit.forecast(horizon=forecast_horizon)
        rolling_forecasts.extend(np.sqrt(forecast.variance.values[-1, :]))
        actual_next_5_days.extend(test_returns[i:i+forecast_horizon])
    
    # Trim forecasts and actuals to match
    min_len = min(len(rolling_forecasts), len(actual_next_5_days))
    rolling_forecasts = rolling_forecasts[:min_len]
    actual_next_5_days = actual_next_5_days[:min_len]
    
    # Create date range for x-axis
    forecast_dates = test_returns.index[:min_len]
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_dates, np.abs(actual_next_5_days), label='Actual Volatility', alpha=0.5)
    plt.plot(forecast_dates, rolling_forecasts, label='Predicted Volatility', color='red')
    plt.title('ARCH Volatility Forecast (Rolling 5-day)')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.show()
    
    # Calculate error metrics
    mse = np.mean((np.array(rolling_forecasts) - np.abs(actual_next_5_days))**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(np.array(rolling_forecasts) - np.abs(actual_next_5_days)))
    
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    
    return model_fit, rolling_forecasts, actual_next_5_days, forecast_dates, returns