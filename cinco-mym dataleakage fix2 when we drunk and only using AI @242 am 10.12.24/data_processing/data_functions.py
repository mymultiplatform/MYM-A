import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback
from tqdm import tqdm
import MetaTrader5 as mt5


def fetch_historical_data(mt5, symbol, timeframe, start_date, end_date):
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    print(f"Attempting to fetch data for {symbol} from {start_date} to {end_date}")
    
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        print(f"Failed to retrieve rates for {symbol} from {start_date} to {end_date}")
        raise Exception("Failed to retrieve rates")
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"Successfully fetched {len(df)} data points")
    return df[['time', 'close']]

def preprocess_data(data, window_size=5, scaler=None, is_training=True):
    if len(data) < window_size:
        print(f"Not enough data for preprocessing. Required: {window_size}, Available: {len(data)}")
        return None, None

    if scaler is None and is_training:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
    elif scaler is not None:
        scaled_data = scaler.transform(data)
    else:
        print("Error: Scaler is None for non-training data")
        return None, None

    return scaled_data, scaler

def run_trading_process():
    from trading.trading_functions import backtest_strategy, generate_performance_report, export_trades_to_excel
    symbol = "Nvidia"
    timeframe = mt5.TIMEFRAME_D1
    start_date = datetime(2010, 1, 1)
    train_end_date = datetime(2022, 1, 1)
    backtest_end_date = datetime(2023, 1, 1)
    window_size = 5  # Consistent window size

    # Fetch all historical data
    all_data = fetch_historical_data(mt5, symbol, timeframe, start_date, backtest_end_date)
    if all_data is None or len(all_data) == 0:
        print("Failed to fetch historical data")
        return

    # Split data into training and backtesting periods
    train_data = all_data[all_data['time'] < train_end_date]
    backtest_data = all_data[(all_data['time'] >= train_end_date) & (all_data['time'] < backtest_end_date)]

    # Scale the training data
    scaled_train_data, scaler = preprocess_data(train_data[['close']], window_size, is_training=True)
    if scaler is None:
        print("Failed to scale training data")
        return

    # Train the model on 2010-2021 data
    model = train_model(scaled_train_data, window_size)
    if model is None:
        print("Failed to train model")
        return

    # Prepare and scale backtest data using the same scaler
    scaled_backtest_data, _ = preprocess_data(backtest_data[['close']], window_size, scaler, is_training=False)

    # Run backtesting strategy on 2022-2023 data
    print(f"Running backtesting from {backtest_data.iloc[0]['time']} to {backtest_data.iloc[-1]['time']}")
    backtest_results = backtest_strategy(scaled_backtest_data, model, scaler, window_size, backtest_data['time'])

    # Generate and export reports
    generate_performance_report(backtest_data.iloc[-1]['time'], "Backtest")
    export_trades_to_excel("trades_report.xlsx")






def create_train_data(scaled_data, window_size):
    if scaled_data is None or len(scaled_data) == 0:
        print("No data to create training set")
        return None, None
    
    X_train, y_train = [], []
    for i in range(window_size, len(scaled_data)):
        X_train.append(scaled_data[i-window_size:i, 0])
        y_train.append(scaled_data[i, 0])
    
    if len(X_train) == 0 or len(y_train) == 0:
        print("Not enough data to create training set")
        return None, None
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train

def train_lstm_model(X_train, y_train, window_size):
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    
    with strategy.scope():
        model = Sequential([
            LSTM(200, return_sequences=True, input_shape=(window_size, 1)),
            LSTM(200, return_sequences=True),
            LSTM(100, return_sequences=False),
            Dense(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
    
    
    epochs = 1000
    batch_size = 128
    
    print("Starting model training...")
    with tqdm(total=epochs, unit="epoch") as pbar:
        history = model.fit(
            X_train, y_train, 
            batch_size=batch_size, 
            epochs=epochs, 
            validation_split=0.2,
            callbacks=[
                TqdmCallback(verbose=0)
            ],
            verbose=0
        )
        pbar.update(len(history.history['loss']))
    
    print("Model training completed.")
    return model

def predict_future(model, data, scaler, future_days, window_size):
    input_seq = data[-window_size:]
    predictions = []
    for _ in range(future_days):
        input_seq_scaled = scaler.transform(input_seq.reshape(-1, 1))
        input_seq_reshaped = input_seq_scaled.reshape((1, window_size, 1))
        predicted_price = model.predict(input_seq_reshaped)[0]
        predictions.append(predicted_price)
        input_seq = np.append(input_seq[1:], scaler.inverse_transform(predicted_price.reshape(-1, 1))[0])
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))


def determine_trend(predicted_prices):
    start_price = predicted_prices[0]
    end_price = predicted_prices[-1]
    if end_price > start_price:
        return "Bull"
    elif end_price < start_price:
        return "Bear"
    else:
        return "Neutral"

def split_data(data, train_ratio=0.7, val_ratio=0.15):
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    return train_data, val_data, test_data

def train_model(train_data, window_size=30):
    if len(train_data) < window_size:
        print("Not enough data to train the model")
        return None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_data[['close']].values)

    X_train, y_train = create_train_data(scaled_data, window_size)
    if X_train is None or y_train is None:
        return None, None

    model = train_lstm_model(X_train, y_train, window_size)
    return model, scaler

def determine_trend(current_price, predicted_price):
    if predicted_price > current_price:
        return "Bull"
    elif predicted_price < current_price:
        return "Bear"
    else:
        return "Neutral"
