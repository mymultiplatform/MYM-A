import MetaTrader5 as mt5
from tkinter import messagebox
import threading
from data_processing.data_functions import fetch_historical_data, preprocess_data, create_train_data, train_lstm_model, predict_future, determine_trend
from trading.trading_functions import place_trade, generate_performance_report
import pandas as pd
import time
from datetime import datetime
import numpy as np
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
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2021, 12, 31)

    threading.Thread(target=run_backtest, args=(symbol, timeframe, start_date, end_date), daemon=True).start()

def run_backtest(symbol, timeframe, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        print("Failed to fetch historical data")
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"Fetched {len(df)} data points for backtesting")

    if len(df) < 60:
        print("Not enough data for backtesting")
        return

    window_size = 365
    step_size = 30

    for i in range(window_size, len(df), step_size):
        train_data = df.iloc[max(0, i-window_size):i]
        test_data = df.iloc[i:i+step_size]

        scaled_train_data, scaler = preprocess_data(train_data[['close']])
        if scaled_train_data is None or scaler is None:
            print(f"Failed to preprocess training data at index {i}")
            continue

        X_train, y_train = create_train_data(scaled_train_data, 60)
        if X_train is None or y_train is None:
            print(f"Failed to create training data at index {i}")
            continue

        model = train_lstm_model(X_train, y_train)
        if model is None:
            print(f"Failed to train model at index {i}")
            continue

        for j, row in test_data.iterrows():
            historical_data = df.iloc[:j+1]
            scaled_historical_data = scaler.transform(historical_data[['close']].values.reshape(-1, 1))

            future_days = 1
            predicted_prices = predict_future(model, scaled_historical_data[-60:], scaler, future_days)
            predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

            actual_price = row['close']
            predicted_price = predicted_prices[0][0]

            trend = determine_trend([actual_price, predicted_price])

            if trend == "Bull":
                place_trade("BUY", row['time'], actual_price)
            elif trend == "Bear":
                place_trade("SELL", row['time'], actual_price)

    generate_performance_report()

def start_automation():
    threading.Thread(target=automation_loop, daemon=True).start()

def automation_loop():
    symbol = "BTCUSD"
    timeframe = mt5.TIMEFRAME_D1
    start_date = "2024-01-01"
    end_date = "2024-10-10"

    try:
        data = fetch_historical_data(mt5, symbol, timeframe, start_date, end_date)
    except Exception as e:
        print(f"Failed to fetch historical data: {e}")
        return

    print(f"Fetched {len(data)} data points for live trading")

    if len(data) < 60:
        print("Not enough data for live trading")
        return

    window_size = 365
    step_size = 30

    for i in range(window_size, len(data), step_size):
        train_data = data.iloc[max(0, i-window_size):i]
        test_data = data.iloc[i:min(i+step_size, len(data))]

        scaled_train_data, scaler = preprocess_data(train_data[['close']])
        if scaled_train_data is None or scaler is None:
            print(f"Failed to preprocess training data at index {i}")
            continue

        X_train, y_train = create_train_data(scaled_train_data, 60)
        if X_train is None or y_train is None:
            print(f"Failed to create training data at index {i}")
            continue

        model = train_lstm_model(X_train, y_train)
        if model is None:
            print(f"Failed to train model at index {i}")
            continue

        for j, row in test_data.iterrows():
            historical_data = data.iloc[:j+1]
            scaled_historical_data = scaler.transform(historical_data[['close']].values.reshape(-1, 1))

            future_days = 1
            predicted_prices = predict_future(model, scaled_historical_data[-60:], scaler, future_days)
            predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

            actual_price = row['close']
            predicted_price = predicted_prices[0][0]

            trend = determine_trend([actual_price, predicted_price])

            if trend == "Bull":
                place_trade("BUY", row['time'], actual_price)
            elif trend == "Bear":
                place_trade("SELL", row['time'], actual_price)

            time.sleep(1)

    generate_performance_report()
    print("Live trading simulation completed")
