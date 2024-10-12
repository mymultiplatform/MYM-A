import MetaTrader5 as mt5
from tkinter import messagebox
import threading
from data_processing.data_functions import fetch_historical_data, preprocess_data, train_lstm_model, predict_future, determine_trend
from trading.trading_functions import place_trade
import time
import numpy as np



def connect_to_mt5(login, password, server):
    if not mt5.initialize():
        messagebox.showerror("Error", "initialize() failed")
        mt5.shutdown()
        return

    authorized = mt5.login(login=int(login), password=password, server=server)
    if authorized:
        print("Connected to MetaTrader 5")
        start_automation()
    else:
        messagebox.showerror("Error", "Failed to connect to MetaTrader 5")

def start_automation():
    def automation_loop():
        symbol = "BTCUSD"
        timeframe = mt5.TIMEFRAME_D1
        start_date = "2023-09-01"
        end_date = "2023-12-31"
        
        # Fetch historical data
        data = fetch_historical_data(mt5, symbol, timeframe, start_date, end_date)
        
        # Split data into training and testing sets
        train_data = data[data['time'] < '2023-09-01']
        test_data = data[data['time'] >= '2023-12-31']
        
        # Preprocess and train model on training data
        scaled_train_data, scaler = preprocess_data(train_data)
        model = train_lstm_model(scaled_train_data)
        
        # Simulate trading on test data
        for i in range(len(test_data)):
            current_date = test_data.iloc[i]['time']
            historical_data = data[data['time'] < current_date]
            
            scaled_historical_data, _ = preprocess_data(historical_data)
            
            future_days = 30
            predicted_prices = predict_future(model, scaled_historical_data, future_days)
            predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
            
            trend = determine_trend(predicted_prices)
            
            if trend == "Bull":
                place_trade(mt5.ORDER_TYPE_BUY, current_date)
            elif trend == "Bear":
                place_trade(mt5.ORDER_TYPE_SELL, current_date)
            
            # Retrain model periodically (e.g., every 30 days)
            if i % 30 == 0:
                updated_train_data = data[data['time'] < current_date]
                scaled_updated_train_data, scaler = preprocess_data(updated_train_data)
                model = train_lstm_model(scaled_updated_train_data)
            
            time.sleep(1)  # Simulate daily trading
        
        print("Backtesting completed for 2023")

    threading.Thread(target=automation_loop, daemon=True).start()
