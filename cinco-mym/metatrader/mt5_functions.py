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
        while True:
            symbol = "BTCUSD"
            timeframe = mt5.TIMEFRAME_D1
            days = 600
            data = fetch_historical_data(mt5, symbol, timeframe, days)
            scaled_data, scaler = preprocess_data(data)
            model = train_lstm_model(scaled_data)
            future_days = 60
            predicted_prices = predict_future(model, scaled_data, future_days)
            predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
            trend = determine_trend(predicted_prices)
            if trend == "Bull":
                place_trade(mt5.ORDER_TYPE_BUY)
            elif trend == "Bear":
                place_trade(mt5.ORDER_TYPE_SELL)
            time.sleep(3600)  # Run the prediction and trading every hour

    threading.Thread(target=automation_loop, daemon=True).start()
