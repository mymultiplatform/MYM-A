import tkinter as tk
from tkinter import messagebox
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from hyperopt import hp, tpe, fmin, Trials
from hyperopt.pyll.base import scope
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
            
            # Train models
            lstm_model = train_lstm_model(scaled_data)
            gb_model = train_gb_model(scaled_data)
            hyperopt_model = train_hyperopt_model(scaled_data)
            arima_model = train_arima_model(data['close'])
            svr_model = train_svr_model(scaled_data)
            rf_model = train_rf_model(scaled_data)
            mlp_model = train_mlp_model(scaled_data)
            transformer_model = train_transformer_model(scaled_data)
            gru_model = train_gru_model(scaled_data)

            # Predict future prices
            future_days = 60
            lstm_predictions = predict_future_lstm(lstm_model, scaled_data, future_days)
            gb_predictions = predict_future_gb(gb_model, scaled_data, future_days)
            hyperopt_predictions = predict_future_hyperopt(hyperopt_model, scaled_data, future_days)
            arima_predictions = predict_future_arima(arima_model, len(data), future_days)
            svr_predictions = predict_future_svr(svr_model, scaled_data, future_days)
            rf_predictions = predict_future_rf(rf_model, scaled_data, future_days)
            mlp_predictions = predict_future_mlp(mlp_model, scaled_data, future_days)
            transformer_predictions = predict_future_transformer(transformer_model, scaled_data, future_days)
            gru_predictions = predict_future_gru(gru_model, scaled_data, future_days)

            # Combine predictions
            combined_predictions = (np.array(lstm_predictions) + 
                                    np.array(gb_predictions) + 
                                    np.array(hyperopt_predictions) + 
                                    np.array(arima_predictions) + 
                                    np.array(svr_predictions) + 
                                    np.array(rf_predictions) + 
                                    np.array(mlp_predictions) + 
                                    np.array(transformer_predictions) + 
                                    np.array(gru_predictions)) / 9
            combined_predictions = scaler.inverse_transform(np.array(combined_predictions).reshape(-1, 1))

            # Determine trend
            trend = determine_trend(combined_predictions)
            if trend == "Bull":
                place_trade(mt5.ORDER_TYPE_BUY)
            elif trend == "Bear":
                place_trade(mt5.ORDER_TYPE_SELL)
            time.sleep(3600)  # Run the prediction and trading every hour

    threading.Thread(target=automation_loop, daemon=True).start()

def fetch_historical_data(symbol, timeframe, days):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, days)
    if rates is None or len(rates) == 0:
        raise Exception("Failed to retrieve rates")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['open', 'high', 'low', 'close', 'tick_volume']])
    return scaled_data, scaler

def train_lstm_model(scaled_data):
    time_step = 60
    X_train, y_train = create_train_data(scaled_data, time_step)

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, batch_size=32, epochs=50)
    return model

def train_gb_model(scaled_data):
    time_step = 60
    X_train, y_train = create_train_data(scaled_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], -1)

    def objective(params):
        model = GradientBoostingRegressor(**params)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2)
        model.fit(X_train_split, y_train_split)
        y_pred = model.predict(X_val_split)
        return mean_squared_error(y_val_split, y_pred)

    space = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 10)),
        'learning_rate': hp.loguniform('learning_rate', -4, 0),
        'max_depth': scope.int(hp.quniform('max_depth', 1, 9, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1))
    }

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
    model = GradientBoostingRegressor(**best)
    model.fit(X_train, y_train)
    return model

def train_hyperopt_model(scaled_data):
    time_step = 60
    X_train, y_train = create_train_data(scaled_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], -1)

    def objective(params):
        model = GradientBoostingRegressor(**params)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2)
        model.fit(X_train_split, y_train_split)
        y_pred = model.predict(X_val_split)
        return mean_squared_error(y_val_split, y_pred)

    space = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 10)),
        'learning_rate': hp.loguniform('learning_rate', -4, 0),
        'max_depth': scope.int(hp.quniform('max_depth', 1, 9, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1))
    }

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
    model = GradientBoostingRegressor(**best)
    model.fit(X_train, y_train)
    return model

def train_arima_model(data):
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit

def train_svr_model(scaled_data):
    time_step = 60
    X_train, y_train = create_train_data(scaled_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], -1)
    
    def objective(params):
        model = SVR(**params)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2)
        model.fit(X_train_split, y_train_split)
        y_pred = model.predict(X_val_split)
        return mean_squared_error(y_val_split, y_pred)

    space = {
        'C': hp.loguniform('C', -1, 3),
        'epsilon': hp.loguniform('epsilon', -3, 0),
        'gamma': hp.loguniform('gamma', -3, 0)
    }

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
    model = SVR(**best)
    model.fit(X_train, y_train)
    return model

def train_rf_model(scaled_data):
    time_step = 60
    X_train, y_train = create_train_data(scaled_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], -1)

    def objective(params):
        model = RandomForestRegressor(**params)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2)
        model.fit(X_train_split, y_train_split)
        y_pred = model.predict(X_val_split)
        return mean_squared_error(y_val_split, y_pred)

    space = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 10)),
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1
  'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1))
    }

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
    model = RandomForestRegressor(**best)
    model.fit(X_train, y_train)
    return model

def train_mlp_model(scaled_data):
    time_step = 60
    X_train, y_train = create_train_data(scaled_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], -1)

    def objective(params):
        model = MLPRegressor(**params)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2)
        model.fit(X_train_split, y_train_split)
        y_pred = model.predict(X_val_split)
        return mean_squared_error(y_val_split, y_pred)

    space = {
        'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50)]),
        'alpha': hp.loguniform('alpha', -5, 0),
        'learning_rate_init': hp.loguniform('learning_rate_init', -5, -1),
        'max_iter': scope.int(hp.quniform('max_iter', 100, 1000, 100))
    }

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
    model = MLPRegressor(**best)
    model.fit(X_train, y_train)
    return model

class TransformerModel(nn.Module):
    def __init__(self, feature_size=5, num_layers=1, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=feature_size, nhead=1, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(feature_size, 1)

    def forward(self, src, tgt):
        output = self.transformer(src, tgt)
        return self.linear(output)

def train_transformer_model(scaled_data):
    time_step = 60
    X_train, y_train = create_train_data(scaled_data, time_step)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    model = TransformerModel(feature_size=X_train.shape[2])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):  # Number of epochs
        model.train()
        optimizer.zero_grad()
        output = model(X_train, X_train)
        loss = criterion(output.squeeze(), y_train)
        loss.backward()
        optimizer.step()

    return model

def train_gru_model(scaled_data):
    time_step = 60
    X_train, y_train = create_train_data(scaled_data, time_step)

    model = Sequential()
    model.add(GRU(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(GRU(100, return_sequences=False))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, batch_size=32, epochs=50)
    return model

def create_train_data(scaled_data, time_step):
    X_train, y_train = [], []
    for i in range(time_step, len(scaled_data)):
        X_train.append(scaled_data[i-time_step:i])
        y_train.append(scaled_data[i, 3])  # Using close price as target
    X_train, y_train = np.array(X_train), np.array(y_train)
    return X_train, y_train

def predict_future_lstm(model, data, future_days):
    predictions = []
    time_step = 60
    input_seq = data[-time_step:]

    for _ in range(future_days):
        input_seq = input_seq.reshape((1, input_seq.shape[0], input_seq.shape[1]))
        predicted_price = model.predict(input_seq)[0]
        predictions.append(predicted_price)
        new_row = np.array([predicted_price] * input_seq.shape[2]).reshape(1, -1)
        input_seq = np.vstack((input_seq[0, 1:], new_row))
    return predictions

def predict_future_gb(model, data, future_days):
    predictions = []
    time_step = 60
    input_seq = data[-time_step:].reshape(1, -1)

    for _ in range(future_days):
        predicted_price = model.predict(input_seq)[0]
        predictions.append(predicted_price)
        new_row = np.array([predicted_price] * data.shape[1]).reshape(1, -1)
        input_seq = np.hstack((input_seq[:, data.shape[1]:], new_row))
    return predictions

def predict_future_hyperopt(model, data, future_days):
    return predict_future_gb(model, data, future_days)

def predict_future_arima(model, start_index, future_days):
    forecast = model.forecast(steps=future_days)
    return forecast

def predict_future_svr(model, data, future_days):
    predictions = []
    time_step = 60
    input_seq = data[-time_step:].reshape(1, -1)

    for _ in range(future_days):
        predicted_price = model.predict(input_seq)[0]
        predictions.append(predicted_price)
        new_row = np.array([predicted_price] * data.shape[1]).reshape(1, -1)
        input_seq = np.hstack((input_seq[:, data.shape[1]:], new_row))
    return predictions

def predict_future_rf(model, data, future_days):
    return predict_future_gb(model, data, future_days)

def predict_future_mlp(model, data, future_days):
    return predict_future_gb(model, data, future_days)

def predict_future_transformer(model, data, future_days):
    model.eval()
    predictions = []
    time_step = 60
    input_seq = torch.tensor(data[-time_step:], dtype=torch.float32).unsqueeze(0)

    for _ in range(future_days):
        with torch.no_grad():
            predicted_price = model(input_seq, input_seq)[:, -1, :].item()
        predictions.append(predicted_price)
        new_row = torch.tensor([[predicted_price] * data.shape[1]], dtype=torch.float32)
        input_seq = torch.cat((input_seq[:, 1:], new_row), dim=1)
    return predictions

def predict_future_gru(model, data, future_days):
    return predict_future_lstm(model, data, future_days)

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
