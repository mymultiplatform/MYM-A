import MetaTrader5 as mt5
from tkinter import messagebox
import threading
from data_processing.data_functions import fetch_historical_data, preprocess_data, train_lstm_model, predict_future, determine_trend
from trading.trading_functions import place_trade
import time
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

# Performance Tracker class as defined previously
class PerformanceTracker:
    def __init__(self):
        self.total_profit = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.trade_history = []

    def update(self, trade_type, entry_price, exit_price, entry_date, exit_date):
        profit = exit_price - entry_price if trade_type == "BUY" else entry_price - exit_price
        self.total_profit += profit
        self.total_trades += 1
        if profit > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        self.trade_history.append({
            "type": trade_type,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "profit": profit
        })

    def calculate_win_rate(self):
        return (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0

    def calculate_average_profit(self):
        return self.total_profit / self.total_trades if self.total_trades > 0 else 0

    def calculate_max_drawdown(self):
        cumulative_profits = np.cumsum([trade["profit"] for trade in self.trade_history])
        max_drawdown = 0
        peak = cumulative_profits[0]
        for profit in cumulative_profits:
            if profit > peak:
                peak = profit
            drawdown = peak - profit
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown

    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        profits = [trade["profit"] for trade in self.trade_history]
        returns = np.array(profits) / [trade["entry_price"] for trade in self.trade_history]
        excess_returns = returns - risk_free_rate / 252  # Assuming 252 trading days in a year
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns) if len(returns) > 0 else 0

    def generate_report(self):
        print("\n" + "="*50)
        print("Performance Report:")
        print(f"Total Profit/Loss: ${self.total_profit:.2f}")
        print(f"Total Trades: {self.total_trades}")
        print(f"Winning Trades: {self.winning_trades}")
        print(f"Losing Trades: {self.losing_trades}")
        print(f"Win Rate: {self.calculate_win_rate():.2f}%")
        print(f"Average Profit per Trade: ${self.calculate_average_profit():.2f}")
        print(f"Maximum Drawdown: ${self.calculate_max_drawdown():.2f}")
        print(f"Sharpe Ratio: {self.calculate_sharpe_ratio():.2f}")
        print("="*50 + "\n")

# Initialize the PerformanceTracker
performance_tracker = PerformanceTracker()

def connect_to_mt5(login, password, server):
    if not mt5.initialize():
        messagebox.showerror("Error", "initialize() failed")
        mt5.shutdown()
        return

    authorized = mt5.login(login=int(login), password=password, server=server)
    if authorized:
        print("Connected to MetaTrader 5")
        start_backtesting()
        start_automation()
    else:
        messagebox.showerror("Error", "Failed to connect to MetaTrader 5")

def start_backtesting():
    symbol = "BTCUSD"
    timeframe = mt5.TIMEFRAME_D1
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    threading.Thread(target=run_backtest, args=(symbol, timeframe, start_date, end_date), daemon=True).start()

def run_backtest(symbol, timeframe, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        print("Failed to fetch historical data")
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    print(f"Fetched {len(df)} data points for backtesting")
    
    if len(df) < 60:  # Ensure we have at least 60 data points
        print("Not enough data for backtesting")
        return

    # Preprocess all data at once
    scaled_data, scaler = preprocess_data(df[['close']])
    if scaled_data is None or scaler is None:
        print("Failed to preprocess data")
        return
    
    print(f"Scaled data shape: {scaled_data.shape}")

    # Train the model once with all available data
    model = train_lstm_model(scaled_data)
    if model is None:
        print("Failed to train model")
        return

    # Use a sliding window approach for backtesting
    window_size = 60  # Use the same window size as in create_train_data
    for i in range(window_size, len(df)):
        current_data = scaled_data[i-window_size:i]
        
        future_days = 1
        predicted_prices = predict_future(model, current_data, future_days)
        predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
        
        actual_price = df.iloc[i]['close']
        predicted_price = predicted_prices[0][0]
        
        if predicted_price > actual_price:
            trend = "Bull"
        elif predicted_price < actual_price:
            trend = "Bear"
        else:
            trend = "Neutral"
        
        # Simulate trade based on prediction
        if trend in ["Bull", "Bear"]:
            entry_price = df.iloc[i-1]['close']
            exit_price = actual_price
            trade_type = "BUY" if trend == "Bull" else "SELL"
            
            performance_tracker.update(trade_type, entry_price, exit_price, df.iloc[i-1]['time'], df.iloc[i]['time'])

    # Print backtest results
    performance_tracker.generate_report()

def start_automation():
    threading.Thread(target=automation_loop, daemon=True).start()

def automation_loop():
    symbol = "BTCUSD"
    timeframe = mt5.TIMEFRAME_D1
    start_date = "2024-01-01"
    end_date = "2024-10-10"
    
    # Fetch historical data
    try:
        data = fetch_historical_data(mt5, symbol, timeframe, start_date, end_date)
    except Exception as e:
        print(f"Failed to fetch historical data: {e}")
        return
    
    print(f"Fetched {len(data)} data points for live trading")
    print(f"Data columns: {data.columns}")
    print(f"Data head:\n{data.head()}")
    
    if len(data) < 60:  # Ensure we have at least 60 data points
        print("Not enough data for live trading")
        return
    
    # Split data into training and testing sets
    train_data = data[data['time'] < '2024-01-01']
    test_data = data[data['time'] >= '2024-01-01']


    
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    
    # Preprocess and train model on training data
    scaled_train_data, scaler = preprocess_data(train_data[['close']])
    if scaled_train_data is None or scaler is None:
        print("Failed to preprocess training data")
        return
    
    print(f"Scaled training data shape: {scaled_train_data.shape}")
    
    model = train_lstm_model(scaled_train_data)
    if model is None:
        print("Failed to train initial model")
        return
    
    # Simulate trading on test data
    for i in range(len(test_data)):
        current_date = test_data.iloc[i]['time']
        historical_data = data[data['time'] < current_date]
        
        scaled_historical_data, _ = preprocess_data(historical_data[['close']])
        if scaled_historical_data is None:
            print(f"Failed to preprocess historical data at index {i}")
            continue
        
        future_days = 1
        predicted_prices = predict_future(model, scaled_historical_data[-60:], future_days)
        predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
        
        actual_price = test_data.iloc[i]['close']
        predicted_price = predicted_prices[0][0]
        
        if predicted_price > actual_price:
            trend = "Bull"
        elif predicted_price < actual_price:
            trend = "Bear"
        else:
            trend = "Neutral"
        
        if trend == "Bull":
            entry_price = test_data.iloc[i-1]['close'] if i > 0 else test_data.iloc[i]['close']
            exit_price = actual_price
            performance_tracker.update("BUY", entry_price, exit_price, historical_data.iloc[-1]['time'], current_date)

        elif trend == "Bear":
            entry_price = test_data.iloc[i-1]['close'] if i > 0 else test_data.iloc[i]['close']
            exit_price = actual_price
            performance_tracker.update("SELL", entry_price, exit_price, historical_data.iloc[-1]['time'], current_date)
        
        # Retrain model periodically (e.g., every 30 days)
        if i % 30 == 0 and i > 0:
            updated_train_data = data[data['time'] < current_date]
            scaled_updated_train_data, scaler = preprocess_data(updated_train_data[['close']])
            if scaled_updated_train_data is not None and scaler is not None:
                model = train_lstm_model(scaled_updated_train_data)
                if model is None:
                    print(f"Failed to retrain model at index {i}")
        
        time.sleep(1)  # Simulate daily trading
    
    # Print final results of live trading
    performance_tracker.generate_report()
    print("Live trading simulation completed for 2023")
