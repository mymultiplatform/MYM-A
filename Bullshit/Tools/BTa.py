import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

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

    # Setup parameters for backtesting
    num_episodes = 250
    time_step = 60
    balance = 1000  # Starting balance with $1000 capital
    leverage = 5  # Leverage is 5:1
    stop_loss_pct = 0.02  # Stop-loss at 2% of the trade

    # Train the LSTM model
    model = train_lstm_model(scaled_data[:int(len(scaled_data) * 0.8)], scaled_data[int(len(scaled_data) * 0.8):])

    cumulative_balance = []  # Initialize cumulative balance list
    trade_log = []  # Log to store trade details for plotting

    # Loop through 250 days, making a trade at 12 PM each day
    for episode_num in range(num_episodes):
        print(f"Starting Episode {episode_num + 1}")
        trade_data = run_episode(episode_num + 1, model, data, scaled_data, scaler, balance, time_step, stop_loss_pct, leverage)
        cumulative_balance.append(balance)  # Update cumulative balance after each episode
        trade_log.append(trade_data)  # Store trade data for plotting

    # After all episodes, plot the final results
    plot_final_results(cumulative_balance, num_episodes)
    animate_trades(trade_log, cumulative_balance, num_episodes)

def run_episode(episode_num, model, data, scaled_data, scaler, balance, time_step, stop_loss_pct, leverage):
    trade_index = episode_num + 1  # Start at 1 for 250 episodes

    if trade_index < len(data):
        entry_time = data['time'].iloc[trade_index].replace(hour=12, minute=0, second=0)
        if entry_time in data['time'].values:
            index = data[data['time'] == entry_time].index[0]

            # Prepare input sequence for LSTM
            input_seq = scaled_data[index - time_step:index].reshape((1, time_step, 1))
            predicted_price = model.predict(input_seq)[0][0]
            predicted_price = scaler.inverse_transform([[predicted_price]])[0][0]

            current_price = data['close'].iloc[index]
            trade_type = 'Buy' if predicted_price > current_price else 'Sell'
            open_trade = {
                'time': entry_time,
                'type': trade_type,
                'entry_price': current_price,
                'trade_number': episode_num + 1
            }

            trade_open = True
            stop_loss_price = (open_trade['entry_price'] * (1 - stop_loss_pct)) if trade_type == 'Buy' else (open_trade['entry_price'] * (1 + stop_loss_pct))

            for j in range(index + 1, len(data)):
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
                        print(f"Trade {episode_num + 1} stopped out at {data['time'].iloc[j]} with stop-loss at {next_price}")
                        trade_open = False
                        break

                    # If trade hits profit target (2% of trade amount)
                    if trade_type == 'Buy' and next_price >= open_trade['entry_price'] * 1.02:
                        profit = (balance * leverage) * 0.02  # 2% profit
                        balance += profit
                        trade_open = False
                    elif trade_type == 'Sell' and next_price <= open_trade['entry_price'] * 0.98:
                        profit = (balance * leverage) * 0.02  # 2% profit
                        balance += profit
                        trade_open = False

                if not trade_open:
                    break  # Exit loop once the trade is closed

            return {
                **open_trade,
                'exit_price': next_price,
                'exit_time': data['time'].iloc[j],
                'final_balance': balance
            }

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

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))  # Add dropout to prevent overfitting
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=50, batch_size=32, callbacks=[early_stopping])

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

def animate_trades(trade_log, cumulative_balance, num_episodes):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'BTCUSD Price Prediction with Trades')
    lines, = ax.plot([], [], lw=2)
    scatter_entries = ax.scatter([], [], color='green', label='Entry', marker='o')
    scatter_exits = ax.scatter([], [], color='blue', label='Exit', marker='v')

    def update(frame):
        ax.clear()  # Clear the axis for each frame
        ax.plot(range(len(cumulative_balance)), cumulative_balance, marker='o', label='Balance')
        
        # Draw trades
        if frame < len(trade_log):
            current_trade = trade_log[frame]
            ax.scatter(current_trade['time'], current_trade['entry_price'], color='green', label='Entry', marker='o', s=100)
            ax.scatter(current_trade['exit_time'], current_trade['exit_price'], color='blue', label='Exit', marker='v', s=100)

        ax.legend()
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Balance ($)')
        ax.set_xlim(0, num_episodes)
        ax.set_ylim(min(cumulative_balance), max(cumulative_balance))

    ani = FuncAnimation(fig, update, frames=num_episodes, repeat=False, interval=100)
    plt.show()

if __name__ == "__main__":
    main()
