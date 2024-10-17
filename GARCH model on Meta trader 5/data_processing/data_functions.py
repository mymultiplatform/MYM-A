import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import time
from metatrader.mt5_functions import get_mt5_timeframe, get_mt5_symbol_info
import sys
sys.setrecursionlimit(10000)  # Increase the limit to a higher value

def fetch_mt5_data(symbol, timeframe, start_date, end_date):
    """Fetch data from MetaTrader 5 for a specific date range"""
    print(f"Attempting to fetch {symbol} data from {start_date} to {end_date} on {timeframe} timeframe.")
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        print(f"Failed to fetch data for {symbol}")
        return None
    print(f"Successfully fetched {len(rates)} bars of data.")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

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
    plot_pacf(np.array(series)**2)
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
    # Split data into train and test sets
    train, test = returns[:-test_size], returns[-test_size:]

    # Fit GARCH(1,1) model
    model = arch_model(train, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp='off')

    # Make predictions
    forecasts = model_fit.forecast(horizon=test_size)
    forecast_variance = np.sqrt(forecasts.variance.values[-1, :])

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(test, label='Actual Returns')
    plt.plot(forecast_variance, label='Predicted Volatility')
    plt.title('GARCH(1,1) Volatility Forecast')
    plt.legend()
    plt.show()

    print(model_fit.summary())

    return model_fit, forecasts

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

def garch_analysis(symbol, timeframe_str):
    """Main function to run GARCH analysis"""
    timeframe = get_mt5_timeframe(timeframe_str)
    
    # Define date range for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Fetch data for the last 30 days
    df = fetch_mt5_data(symbol, timeframe, start_date, end_date)
    if df is None:
        print("Failed to fetch data. Exiting analysis.")
        return None
    
    # Calculate returns
    returns = calculate_returns(df['close'])
    
    # Plot data
    plot_data(df['close'], f'Last 30 Days Data: {symbol} ({timeframe_str})')
    plot_data(returns, f'Last 30 Days Returns: {symbol} ({timeframe_str})')
    
    # Plot PACF
    plot_pacf(returns)
    
    # Fit GARCH model on the data
    model_fit = fit_garch_model(returns)
    print(model_fit.summary())
    
    # Predict volatility for the next period
    prediction = forecast_volatility(model_fit)[0]
    
    print(f"Predicted volatility for the next period: {prediction}")
    
    # Perform rolling volatility forecast
    test_size = 100
    rolling_predictions = perform_rolling_volatility_forecast(returns, test_size)
    
    # Plot rolling volatility forecast
    plt.figure(figsize=(10,4))
    true, = plt.plot(returns[-test_size:])
    preds, = plt.plot(rolling_predictions)
    plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
    plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)
    plt.show()
    
    # Calculate standard deviation of returns
    std_dev = returns.std()
    
    return prediction, datetime.now(), rolling_predictions, std_dev

def start_garch_analysis():
    symbol = "USDCHF"
    timeframe = mt5.TIMEFRAME_M15
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()

    # Fetch data from MT5
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    
    if rates is not None and len(rates) > 0:
        # Convert rates to numpy array
        rates_array = np.array(rates)
        
        # Extract close prices
        close_prices = rates_array['close']
        
        # Calculate returns
        returns = 100 * np.log(close_prices[1:] / close_prices[:-1])
        
        # Perform GARCH analysis
        model_fit, forecasts = perform_garch_analysis(returns)
        
        # Perform rolling volatility forecast
        test_size = 100
        rolling_predictions = perform_rolling_volatility_forecast(returns, test_size)
        
        # Plot rolling volatility forecast
        plt.figure(figsize=(10,4))
        true, = plt.plot(returns[-test_size:])
        preds, = plt.plot(rolling_predictions)
        plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
        plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)
        plt.show()
        
        # Calculate and display standard deviation
        std_dev = np.std(returns)
        print(f"Standard deviation of returns: {std_dev}")
        
        print("GARCH analysis and rolling volatility forecast completed successfully.")