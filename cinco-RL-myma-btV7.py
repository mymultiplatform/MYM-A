import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tkinter as tk
import random

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
    days = 4000  # Extended to get data back to 2018
    data = fetch_historical_data(symbol, timeframe, days)
    scaled_data, scaler = preprocess_data(data)

    # Split data into training and validation sets (80% train, 20% validation)
    split_index = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:split_index]
    validation_data = scaled_data[split_index:]

    model = train_lstm_model(train_data, validation_data)

    time_step = 60
    balance = 1000  # Starting balance with $1000 capital
    leverage = 5  # Leverage is 5:1
    stop_loss_pct = 0.02  # Stop-loss at 2% of the trade
    trade_log = []
    rewards = []  # Track rewards for trades

    num_episodes = 250
    num_trades = 10  # Randomly sample 10 trades

    # Initialize cumulative balance list
    cumulative_balance = []

    # Setup animation plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'BTCUSD Price Prediction with LSTM and 5:1 Leverage')
    line, = ax.plot([], [], color='black', label='Close Price')
    trade_markers = []

    def update_plot(trade_data, episode_num):
        ax.clear()
        ax.plot(data['time'], data['close'], color='black', label='Close Price')
        for trade in trade_data:
            entry_color = 'green' if trade['type'] == 'Buy' else 'red'
            exit_color = 'blue'

            # Plot the entry point
            ax.scatter(trade['time'], trade['entry_price'], color=entry_color, label=f"{trade['type']} Entry", marker='^', s=100)

            # Plot the exit point
            ax.scatter(trade['exit_time'], trade['exit_price'], color=exit_color, label=f"{trade['type']} Exit", marker='v', s=100)

            # Draw a dotted line between entry and exit for all trades
            ax.plot([trade['time'], trade['exit_time']],
                     [trade['entry_price'], trade['exit_price']],
                     linestyle='dotted', color='gray')

            # Annotate the exit point
            exit_annotation = "Stopped" if trade.get('stop_loss', False) else "Exit"
            ax.annotate(f"Trade {trade['trade_number']}: {exit_annotation}", (trade['exit_time'], trade['exit_price']),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='black')

        ax.legend()
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'BTCUSD Price Prediction - Episode {episode_num} (Balance: {balance:.2f})')
        ax.grid()

    def run_episode(episode_num):
        nonlocal balance  # Use the outer balance variable
        trade_log = []
        rewards = []

        # Randomly sample 10 trades between 2018 and 2024
        trade_positions = random.sample(range(time_step, len(data) - 1), num_trades)

        for trade_index, i in enumerate(trade_positions):
            input_seq = scaled_data[i - time_step:i].reshape((1, time_step, 1))
            predicted_price = model.predict(input_seq)[0][0]
            predicted_price = scaler.inverse_transform([[predicted_price]])[0][0]

            current_price = data['close'].iloc[i]

            # Determine trade type (Buy or Sell)
            trade_type = 'Buy' if predicted_price > current_price else 'Sell'

            # Log the trade entry
            open_trade = {'time': data['time'].iloc[i], 'type': trade_type, 'entry_price': current_price, 'trade_number': trade_index + 1}

            # Set exit condition without using future data
            trade_open = True
            stop_loss_price = (open_trade['entry_price'] * (1 - stop_loss_pct)) if trade_type == 'Buy' else (open_trade['entry_price'] * (1 + stop_loss_pct))
            for j in range(i + 1, len(data)):
                next_price = data['close'].iloc[j]

                # If balance is zero or below, stop trading
                if balance <= 0:
                    balance = 0
                    print(f"Account Liquidated at {data['time'].iloc[j]}")
                    break

                if trade_open:
                    # Stop-Loss mechanism
                    if (trade_type == 'Buy' and next_price <= stop_loss_price) or (trade_type == 'Sell' and next_price >= stop_loss_price):
                        loss = (balance * leverage) * stop_loss_pct
                        balance -= loss
                        balance = max(0, balance)  # Prevent balance from going negative
                        trade_log.append({**open_trade, 'exit_price': next_price, 'exit_time': data['time'].iloc[j], 'liquidated': False, 'stop_loss': True})
                        trade_open = False
                        rewards.append(-1)  # Punish for stop-loss
                        print(f"Trade {trade_index + 1} stopped out at {data['time'].iloc[j]} with stop-loss at {next_price}")
                        break

                    # If trade hits profit target (2% of trade amount)
                    if trade_type == 'Buy':
                        if next_price >= open_trade['entry_price'] * 1.02:
                            profit = (balance * leverage) * 0.02  # 2% profit
                            balance += profit
                            trade_log.append({**open_trade, 'exit_price': next_price, 'exit_time': data['time'].iloc[j], 'liquidated': False, 'stop_loss': False})
                            rewards.append(1)  # Reward for profit
                            trade_open = False
                    elif trade_type == 'Sell':
                        if next_price <= open_trade['entry_price'] * 0.98:
                            profit = (balance * leverage) * 0.02  # 2% profit
                            balance += profit
                            trade_log.append({**open_trade, 'exit_price': next_price, 'exit_time': data['time'].iloc[j], 'liquidated': False, 'stop_loss': False})
                            rewards.append(1)  # Reward for profit
                            trade_open = False

                if not trade_open:
                    break  # Exit loop once the trade is closed

            # After every trade, adjust the model slightly based on reward or punishment
            adjust_model_with_rewards(model, rewards, input_seq)

        cumulative_balance.append(balance)  # Update cumulative balance after each episode

        # Update the plot with the latest trade data
        update_plot(trade_log, episode_num)

    # Animate through the episodes
    for episode_num in range(num_episodes):
        run_episode(episode_num + 1)

    # After all episodes, plot the final results
    plot_final_results(cumulative_balance, num_episodes)

def adjust_model_with_rewards(model, rewards, input_seq):
    """ 
    Adjust the model weights slightly based on the trade outcome.
    Positive reward increases confidence in the decision, negative reward penalizes it.
    """
    if len(rewards) == 0:
        return  # Skip adjustment if there are no rewards yet

    last_reward = rewards[-1]
    if last_reward > 0:
        # Reward: reinforce the model's decision by training it more with the same input
        model.fit(input_seq, np.array([[1]]), epochs=1, verbose=0)
    else:
        # Punish: penalize by training on opposite decision
        model.fit(input_seq, np.array([[0]]), epochs=1, verbose=0)

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

def plot_final_results(cumulative_balance, num_episodes):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_episodes + 1), cumulative_balance, marker='o')
    plt.title("Final Cumulative Balance Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Balance ($)")
    plt.grid()
    plt.show()

def show_ui():
    root = tk.Tk()
    root.title("LSTM Backtesting Visualization")
    root.geometry("300x200")

    start_button = tk.Button(root, text="Start Backtest", command=main, font=("Helvetica", 14))
    start_button.pack(pady=50)

    root.mainloop()

if __name__ == "__main__":
    show_ui()
