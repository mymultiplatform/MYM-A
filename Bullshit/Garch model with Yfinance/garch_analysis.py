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

def fit_garch_model(returns, p=2, q=2):
    """Fit GARCH(p,q) model"""
    model = arch_model(returns, p=p, q=q)
    return model.fit(disp='off')

def forecast_volatility(model_results, horizon=1):
    """Forecast volatility"""
    forecast = model_results.forecast(horizon=horizon)
    return np.sqrt(forecast.variance.values[-1, :])

def perform_garch_analysis(returns, test_size=100):
    # Convert returns to pandas Series if it's not already
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    # Split data into train and test sets
    train = returns[:-test_size]
    test = returns[-test_size:]

    # Fit GARCH(1,1) model on train data only
    model = arch_model(train, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp='off')

    # Make predictions on test data
    forecasts = model_fit.forecast(horizon=len(test))
    forecast_variance = np.sqrt(forecasts.variance.values[-1, :])

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(test.index, test, label='Actual Returns')
    plt.plot(test.index, forecast_variance, label='Predicted Volatility')
    plt.title('GARCH(1,1) Volatility Forecast')
    plt.legend()
    plt.show()

    print(model_fit.summary())

    return model_fit, forecasts

def tune_garch_parameters(returns, p_range=(1, 3), q_range=(1, 3), num_splits=5):
    """Tune GARCH model parameters using cross-validation"""
    best_aic = np.inf
    best_params = None
    tscv = TimeSeriesSplit(n_splits=num_splits)

    for p in range(p_range[0], p_range[1] + 1):
        for q in range(q_range[0], q_range[1] + 1):
            aic_scores = []
            for train_index, val_index in tscv.split(returns):
                train, val = returns.iloc[train_index], returns.iloc[val_index]
                model = arch_model(train, vol='Garch', p=p, q=q)
                results = model.fit(disp='off')
                aic_scores.append(results.aic)
            avg_aic = np.mean(aic_scores)
            if avg_aic < best_aic:
                best_aic = avg_aic
                best_params = (p, q)

    print(f"Best GARCH parameters: p={best_params[0]}, q={best_params[1]}")
    return best_params

def perform_rolling_volatility_forecast(returns, test_size=100, p=2, q=2):
    """Perform rolling volatility forecast"""
    rolling_predictions = []
    for i in range(test_size):
        train = returns[:-(test_size-i)]
        model = arch_model(train, p=p, q=q)
        model_fit = model.fit(disp='off')
        pred = model_fit.forecast(horizon=1)
        rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
    
    return rolling_predictions

def garch_analysis(data):
    """Main function to run GARCH analysis with rolling 5-day forecast"""
    # Calculate returns for all data
    returns = calculate_returns(data['Close'])
    
    # Split into train and test
    train_end = datetime(2023, 12, 31)
    train_returns = returns[returns.index <= train_end]
    test_returns = returns[returns.index > train_end]
    
    # Tune GARCH parameters using training data
    best_p, best_q = tune_garch_parameters(train_returns)
    
    # Perform rolling forecast
    forecast_horizon = 5  # Forecast 5 steps ahead
    rolling_forecasts = []
    actual_next_5_days = []
    
    for i in range(0, len(test_returns) - forecast_horizon, forecast_horizon):
        train = returns[:len(train_returns) + i]
        model = arch_model(train, p=best_p, q=best_q)
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
    plt.title('GARCH Volatility Forecast (Rolling 5-day)')
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