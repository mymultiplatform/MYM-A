#  / \__
 (    @\____
 /         O
/   (_____/
/_____/ U
  / \__
 (    @\____
 /         O
/   (_____/
/_____/ U
  / \__
 (    @\____
 /         O
/   (_____/
/_____/ U







import tkinter as tk
from tkinter import messagebox
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import threading
import time
import random

def main():
    global root, display_var, click_button, message_label, connect_button
    root = tk.Tk()
    root.title("MYM-A MODO CHEZ")
    root.geometry("600x400")

    # Create frames
    dice_frame = tk.Frame(root, width=300, height=400)
    dice_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    login_frame = tk.Frame(root, width=300, height=400)
    login_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Dice UI
    title_label = tk.Label(dice_frame, text="3 FACE DICE", font=("Helvetica", 16))
    title_label.pack(pady=10)

    display_var = tk.StringVar()
    display_label = tk.Label(dice_frame, textvariable=display_var, font=("Helvetica", 24), width=5, height=2, relief="solid")
    display_label.pack(pady=20)

    click_button = tk.Button(dice_frame, text="Click", font=("Helvetica", 14), command=on_button_click)
    click_button.pack(pady=10)

    message_label = tk.Label(dice_frame, text="", font=("Helvetica", 14), fg="green")
    message_label.pack(pady=10)

    # Login UI
    login_title = tk.Label(login_frame, text="🏧MYM-A", font=("Helvetica", 20))
    login_title.pack(pady=20)

    login_label = tk.Label(login_frame, text="Login:", font=("Helvetica", 14))
    login_label.pack(pady=5)
    login_entry = tk.Entry(login_frame, font=("Helvetica", 14))
    login_entry.pack(pady=5)
    login_entry.insert(0, "312128713")

    password_label = tk.Label(login_frame, text="Password:", font=("Helvetica", 14))
    password_label.pack(pady=5)
    password_entry = tk.Entry(login_frame, show="*", font=("Helvetica", 14))
    password_entry.pack(pady=5)
    password_entry.insert(0, "Sexo247420@")

    server_label = tk.Label(login_frame, text="Server:", font=("Helvetica", 14))
    server_label.pack(pady=5)
    server_entry = tk.Entry(login_frame, font=("Helvetica", 14))
    server_entry.pack(pady=5)
    server_entry.insert(0, "XMGlobal-MT5 7")

    connect_button = tk.Button(login_frame, text="Connect", font=("Helvetica", 14),
                               command=lambda: connect_to_mt5(login_entry.get(), password_entry.get(), server_entry.get()))
    connect_button.pack(pady=20)

    # Start the automatic dice rolling in a separate thread
    threading.Thread(target=auto_roll, daemon=True).start()

    root.mainloop()

def roll_dice():
    def loading_animation():
        loading_text = ["", ".", "..", "..."]
        for _ in range(3):
            for text in loading_text:
                display_var.set(text)
                time.sleep(0.5)

    loading_animation()
    dice_result = random.randint(1, 3)
    display_var.set(dice_result)

    if dice_result == 3:
        message_label.config(text="Click Connect")
        root.after(0, connect_button.invoke)
    else:
        root.after(500, enable_button)

def reset():
    message_label.config(text="")
    display_var.set("")
    enable_button()

def on_button_click():
    threading.Thread(target=roll_dice).start()
    click_button.config(state="disabled")

def enable_button():
    click_button.config(state="normal")

def auto_roll():
    while True:
        time.sleep(10)
        if click_button["state"] == "normal":
            root.after(0, on_button_click)

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
            data = fetch_historical_data(symbol, timeframe, days)
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
            time.sleep(3999600)  # Run the prediction and trading every hour

    threading.Thread(target=automation_loop, daemon=True).start()

def fetch_historical_data(symbol, timeframe, days):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, days)
    if rates is None or len(rates) == 0:
        raise Exception("Failed to retrieve rates")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df[['time', 'close']]

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))
    return scaled_data, scaler

def train_lstm_model(scaled_data):
    time_step = 60
    X_train, y_train = create_train_data(scaled_data, time_step)

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, batch_size=32, epochs=50)
    return model

def create_train_data(scaled_data, time_step):
    X_train, y_train = [], []
    for i in range(time_step, len(scaled_data)):
        X_train.append(scaled_data[i-time_step:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train

def predict_future(model, data, future_days):
    predictions = []
    time_step = 60
    input_seq = data[-time_step:]

    for _ in range(future_days):
        input_seq = input_seq.reshape((1, input_seq.shape[0], 1))
        predicted_price = model.predict(input_seq)[0]
        predictions.append(predicted_price)
        input_seq = np.append(input_seq[:, 1:], predicted_price)
    return predictions

def determine_trend(predicted_prices):
    start_price = predicted_prices[0]
    end_price = predicted_prices[-1]
    if end_price > start_price:
        return "Bull"
    elif end_price < start_price:
        return "Bear"
    else:
        return "Neutral"

def place_trade(order_type):
    symbol = "BTCUSD"
    lot_size = 0.1
    price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "deviation": 10,
        "magic": 234000,
        "comment": "Automated trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print("Trade successfully placed")
        monitor_trade(result.order)
    else:
        messagebox.showerror("Trade Error", f"Failed to place trade: {result.retcode}")

def monitor_trade(order_id):
    while True:
        position = mt5.positions_get(ticket=order_id)
        if position:
            position = position[0]
            profit = position.profit
            if profit >= 0.01:
                close_trade(order_id, position.type)
                break
        time.sleep(5)

def close_trade(order_id, order_type):
    symbol = "BTCUSD"
    price = mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.1,
        "type": mt5.ORDER_TYPE_SELL if order_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "position": order_id,
        "price": price,
        "deviation": 10,
        "magic": 234000,
        "comment": "Automated trade close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print("Trade successfully closed")
        root.after(0, on_button_click)  # Restart the dice rolling
    else:
        messagebox.showerror("Trade Close Error", f"Failed to close trade: {result.retcode}")

if __name__ == "__main__":
    main()
