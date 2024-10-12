import MetaTrader5 as mt5
from tkinter import messagebox
import threading
from data_processing.data_functions import fetch_historical_data, preprocess_data, create_train_data, train_lstm_model, predict_future, determine_trend
from trading.trading_functions import place_trade, generate_performance_report, backtest_strategy, export_trades_to_excel
import pandas as pd
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
        run_trading_process()
    else:
        messagebox.showerror("Error", "Failed to connect to MetaTrader 5")

def run_trading_process():
    symbol = "BTCUSD"
    timeframe = mt5.TIMEFRAME_D1
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2024, 1, 1)

    # Fetch all historical data
    all_data = fetch_historical_data(mt5, symbol, timeframe, start_date, end_date)
    if all_data is None or len(all_data) == 0:
        print("Failed to fetch historical data")
        return

    # Define parameters
    initial_train_size = 365 * 2  # 2 years
    test_size = 365  # 1 year
    retrain_frequency = 30  # Retrain every 30 days

    # Initial training
    initial_train_data = all_data[:initial_train_size]
    model, scaler = train_model(initial_train_data)
    if model is None or scaler is None:
        print("Failed to train initial model")
        return

    # Rolling window backtesting and live simulation
    for i in range(initial_train_size, len(all_data) - test_size, retrain_frequency):
        # Define the current window
        train_data = all_data[i-initial_train_size:i]
        test_data = all_data[i:i+test_size]

        # Retrain the model
        model, scaler = train_model(train_data)
        if model is None or scaler is None:
            print(f"Failed to retrain model at date {all_data.iloc[i]['time']}")
            continue

        # Run backtesting on the test data
        print(f"Running backtesting from {test_data.iloc[0]['time']} to {test_data.iloc[-1]['time']}")
        backtest_strategy(test_data, model, scaler, "Backtest")

        # Generate and export backtest report
        generate_performance_report(test_data.iloc[-1]['time'], "Backtest")
        export_trades_to_excel(f"backtest_trades_{test_data.iloc[-1]['time'].strftime('%Y%m%d')}.xlsx")

    # Final live simulation
    live_sim_data = all_data[-test_size:]
    print(f"Running live simulation from {live_sim_data.iloc[0]['time']} to {live_sim_data.iloc[-1]['time']}")
    run_live_simulation(live_sim_data, model, scaler)

    # Generate and export live simulation report
    generate_performance_report(live_sim_data.iloc[-1]['time'], "Live Simulation")
    export_trades_to_excel("live_sim_trades.xlsx")


def train_model(train_data):
    if len(train_data) < 60:
        print("Not enough data to train the model")
        return None, None

    scaled_train_data, scaler = preprocess_data(train_data[['close']])
    if scaled_train_data is None or scaler is None:
        return None, None

    X_train, y_train = create_train_data(scaled_train_data, 60)
    if X_train is None or y_train is None:
        return None, None

    model = train_lstm_model(X_train, y_train)
    return model, scaler


def run_live_simulation(live_sim_data, initial_model, initial_scaler):
    print("Starting live trading simulation...")
    window_size = 365
    retrain_frequency = 30
    model, scaler = initial_model, initial_scaler
    for i in range(0, len(live_sim_data), retrain_frequency):
        current_data = live_sim_data.iloc[:i]
        if i > 0:
            model, scaler = train_model(current_data.tail(window_size))

        test_data = live_sim_data.iloc[i:min(i+retrain_frequency, len(live_sim_data))]

        for _, row in test_data.iterrows():
            historical_data = current_data[current_data['time'] < row['time']]
            # ... (rest of the function)
            if len(historical_data) < 60:
                print(f"Not enough historical data for date {row['time']}. Skipping...")
                continue

            scaled_historical_data = scaler.transform(historical_data[['close']].values.reshape(-1, 1))
            
            predicted_prices = predict_future(model, scaled_historical_data[-60:], scaler, 1)
            predicted_price = predicted_prices[0][0]
            
            trend = determine_trend([row['close'], predicted_price])
            
            if trend == "Bull":
                place_trade("BUY", row['time'], row['close'], "Live Simulation")
            elif trend == "Bear":
                place_trade("SELL", row['time'], row['close'], "Live Simulation")

            print(f"Date: {row['time']}, Actual: {row['close']:.2f}, Predicted: {predicted_price:.2f}, Trend: {trend}")

    print("Live trading simulation completed")
