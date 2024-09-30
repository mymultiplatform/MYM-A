import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tkinter as tk
from datetime import timedelta
import threading
import time
import random

# Function to roll dice and determine delay
def roll_dice():
    dice_result = random.randint(1, 3)
    return dice_result

def main():
    # Initialize and connect to MT5 to fetch historical data
    if not mt5.initialize():
        raise Exception("MetaTrader5 initialization failed")

    login = 312128713
    password = "Sexo247420@"
    server = "XMGlobal-MT5 7"

    if mt5.login(login=login, password=password, server=server):
        print("Connected to MetaTrader 5 for data retrieval")
        perform_backtest()
    else:
        raise Exception("Failed to connect to MetaTrader 5")

def perform_backtest():
    symbol = "BTCUSD"
    timeframe = mt5.TIMEFRAME_H4  # 4-hour time frame for the simulation
    days = 600
    data = fetch_historical_data(symbol, timeframe, days)
    scaled_data, scaler = preprocess_data(data)

    # Split data into training and validation sets (80% train, 20% validation)
    split_index = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:split_index]
    validation_data = scaled_data[split_index:]

    model = train_lstm_model(train_data, validation_data)

    time_step = 60
    balance = 10000  # Starting balance with $10000 capital
    leverage = 5  # 5:1 leverage
    lot_size = 0.1  # Lot size (adjust as per strategy)
    trade_log = []

    trade_interval_hours = 4
    dice_interval_hours = 5  # Delay if dice roll is 1 or 2

    # Sequentially take trades based on every 'time_step' interval
    trade_amount = balance * leverage  # Amount bet per trade based on leverage
    i = time_step
    first_trade = True

    while i < len(data) - 1:
        input_seq = scaled_data[i - time_step:i].reshape((1, time_step, 1))
        predicted_price = model.predict(input_seq)[0][0]
        predicted_price = scaler.inverse_transform([[predicted_price]])[0][0]

        current_price = data['close'].iloc[i]

        # Determine trade type (Buy or Sell)
        if predicted_price > current_price:
            trade_type = 'Buy'
        else:
            trade_type = 'Sell'

        # Log the trade entry
        open_trade = {'time': data['time'].iloc[i], 'type': trade_type, 'entry_price': current_price}

        # Set exit condition without using future data
        trade_open = True
        for j in range(i + 1, len(data)):
            next_price = data['close'].iloc[j]

            # Calculate price movement for liquidation check (simulating 1 pip = 1 unit)
            price_movement = abs(next_price - open_trade['entry_price'])

            # Simulate checking price at each time step
            if trade_open:
                # Check for liquidation
                if price_movement >= 1:  # Liquidation if price moves 1 pip against the trade
                    balance = 0  # Account gets liquidated
                    print(f"Account Liquidated at {data['time'].iloc[j]} with price movement of {price_movement} pips")
                    return  # Stop the backtest if liquidated

                if trade_type == 'Buy':
                    # Check if profit is at least 1% of the trade amount
                    if next_price >= open_trade['entry_price'] * 1.01:
                        profit = trade_amount * 0.01 * leverage  # Leverage is considered here
                        balance += profit
                        trade_log.append({**open_trade, 'exit_price': next_price, 'exit_time': data['time'].iloc[j]})
                        trade_open = False  # Close the trade
                elif trade_type == 'Sell':
                    # Check if profit is at least 1% of the trade amount
                    if next_price <= open_trade['entry_price'] * 0.99:
                        profit = trade_amount * 0.01 * leverage  # Leverage is considered here
                        balance += profit
                        trade_log.append({**open_trade, 'exit_price': next_price, 'exit_time': data['time'].iloc[j]})
                        trade_open = False  # Close the trade

            if not trade_open:
                break  # Exit loop once the trade is closed

        # Simulate the waiting period (4 hours or additional dice roll hours)
        if first_trade:
            # Wait 4 hours for the first trade
            i += trade_interval_hours * time_step
            first_trade = False
        else:
            # Roll the dice for subsequent trades
            dice_roll = roll_dice()
            if dice_roll == 3:
                # Normal 4-hour delay
                i += trade_interval_hours * time_step
            else:
                # 5-hour delay if dice is 1 or 2
                i += dice_interval_hours * time_step

    # Plotting the results
    plt.figure(figsize=(9, 9))
    plt.plot(data['time'], data['close'], color='black', label='Close Price')

    for trade in trade_log:
        entry_color = 'green' if trade['type'] == 'Buy' else 'red'
        exit_color = 'blue'

        # Plot the entry point
        plt.scatter(trade['time'], trade['entry_price'], color=entry_color, label=f"{trade['type']} Entry")

        # Plot the exit point
        plt.scatter(trade['exit_time'], trade['exit_price'], color=exit_color, label=f"{trade['type']} Exit")

        # Draw a dotted line between entry and exit for all trades
        plt.plot([trade['time'], trade['exit_time']],
                 [trade['entry_price'], trade['exit_price']],
                 linestyle='dotted', color='gray')

        # Annotate the exit point
        plt.annotate(f"exit", (trade['exit_time'], trade['exit_price']), textcoords="offset points", xytext=(0, 10),
                     ha='center')

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'BTCUSD Price Prediction with LSTM and 10:1 Leverage (Balance: {balance:.2f})')
    plt.show()

def fetch_historical_data(symbol, timeframe, days):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, days)
    if rates is None or len(rates) == 0:
        raise Exception("Failed to retrieve rates")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df[['time', 'close', 'open', 'high', 'low']]

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))
    return scaled_data, scaler

def train_lstm_model(train_data, validation_data):
    time_step = 60
    X_train, y_train = create_train_data(train_data, time_step)
    X_val, y_val = create_train_data(validation_data, time_step)

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))  # Add dropout to prevent overfitting
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Implement Early Stopping based on validation loss
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stopping])

    return model

def create_train_data(scaled_data, time_step):
    X_train, y_train = [], []
    for i in range(time_step, len(scaled_data)):
        X_train.append(scaled_data[i - time_step:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train

def show_ui():
    root = tk.Tk()
    root.title("LSTM Backtesting Visualization")
    root.geometry("300x200")

    start_button = tk.Button(root, text="Start Backtest", command=main, font=("Helvetica", 14))
    start_button.pack(pady=50)

    root.mainloop()

if __name__ == "__main__":
    show_ui()
