import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import tensorflow as tf
from tqdm.keras import TqdmCallback
from tqdm import tqdm
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import time


def fetch_historical_data(mt5, symbol, timeframe, start_date, end_date):
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    print(f"Attempting to fetch data for {symbol} from {start_date} to {end_date}")
    
    # Check if the symbol is available
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")
        return None

    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to get symbol info for {symbol}")
        return None

    print(f"Symbol info: {symbol_info._asdict()}")

    # Attempt to fetch rates
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    
    if rates is None or len(rates) == 0:
        print(f"Failed to retrieve rates for {symbol} from {start_date} to {end_date}")
        
        # Check if we're connected to the server
        if not mt5.terminal_info().connected:
            print("Not connected to MetaTrader 5. Please check your connection.")
        
        # Check available symbols
        symbols = mt5.symbols_get()
        print(f"Available symbols: {[s.name for s in symbols]}")
        
        # Check if the timeframe is valid
        timeframes = [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M30,
                      mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_D1, mt5.TIMEFRAME_W1, mt5.TIMEFRAME_MN1]
        if timeframe not in timeframes:
            print(f"Invalid timeframe: {timeframe}")
        
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"Successfully fetched {len(df)} data points")
    return df[['time', 'close']]

def preprocess_data(data, window_size, scaler=None, is_training=True):
    if len(data) < window_size:
        print(f"Not enough data for preprocessing. Required: {window_size}, Available: {len(data)}")
        return None, None

    if scaler is None and is_training:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    elif scaler is not None:
        scaled_data = scaler.transform(data.reshape(-1, 1))
    else:
        print("Error: Scaler is None for non-training data")
        return None, None

    return scaled_data, scaler

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
    
    epochs = 8000
    batch_size = 128
    
    # Add early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    
    print("Starting model training...")
    with tqdm(total=epochs, unit="epoch") as pbar:
        history = model.fit(
            X_train, y_train, 
            batch_size=batch_size, 
            epochs=epochs, 
            validation_split=0.2,
            callbacks=[
                TqdmCallback(verbose=0),
                early_stopping
            ],
            verbose=0
        )
        pbar.update(len(history.history['loss']))
    
    print("Model training completed.")
    return model

def train_model(train_data, window_size):
    if len(train_data) < window_size:
        print("Not enough data to train the model")
        return None

    X_train, y_train = create_train_data(train_data, window_size)
    if X_train is None or y_train is None:
        return None

    model = Sequential([
        LSTM(200, return_sequences=True, input_shape=(window_size, 1)),
        LSTM(200, return_sequences=True),
        LSTM(100, return_sequences=False),
        Dense(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, batch_size=128, epochs=100, validation_split=0.2, verbose=1)
    return model

def predict_future(model, train_data, scaler, future_periods, window_size):
    input_seq = train_data[-window_size:]
    predictions = []
    for _ in range(future_periods):
        input_seq_scaled = scaler.transform(input_seq.reshape(-1, 1))
        input_seq_reshaped = input_seq_scaled.reshape((1, window_size, 1))
        predicted_price = model.predict(input_seq_reshaped)[0]
        predictions.append(predicted_price)
        input_seq = np.append(input_seq[1:], scaler.inverse_transform(predicted_price.reshape(-1, 1))[0])
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

def run_trading_process():
    # Define the symbol and timeframe
    symbol = "BTCUSD"
    timeframe = mt5.TIMEFRAME_H4  # 4-hour timeframe
    start_date = datetime(2012, 4,18)
    train_end_date = datetime(2021, 12, 31, 20, 0)  # Last 4-hour candle of 2021
    window_size = 30  # 10 days

    # Check MT5 connection
    if not mt5.terminal_info().connected:
        print("Not connected to MetaTrader 5. Please check your connection.")
        return

    # Set up relearning every 30 days
    relearn_interval = timedelta(days=30)
    next_relearn_date = train_end_date + relearn_interval

    while True:
        # Fetch all historical 4-hour data from 2010 to the current train_end_date
        all_data = fetch_historical_data(mt5, symbol, timeframe, start_date, train_end_date)
        if all_data is None or len(all_data) == 0:
            print("Failed to fetch historical data. Aborting process.")
            return

        # Verify data
        print(f"Training data from {all_data.iloc[0]['time']} to {all_data.iloc[-1]['time']}")
        print(f"Total data points: {len(all_data)}")

        if len(all_data) < window_size:
            print(f"Not enough data points. Required: {window_size}, Available: {len(all_data)}")
            return

        # Scale the training data
        scaled_train_data, scaler = preprocess_data(all_data['close'].values, window_size, is_training=True)
        if scaler is None:
            print("Failed to scale training data")
            return

        # Train the model on historical data
        model = train_model(scaled_train_data, window_size)
        if model is None:
            print("Failed to train model")
            return

        # Get the last close price from the training data
        last_close = all_data.iloc[-1]['close']
        last_date = all_data.iloc[-1]['time']

        # Fetch actual price data for the prediction period (next 30 days or until next_relearn_date)
        prediction_start_date = last_date + timedelta(hours=4)
        prediction_end_date = min(next_relearn_date, prediction_start_date + timedelta(days=30))
        actual_data = fetch_historical_data(mt5, symbol, timeframe, prediction_start_date, prediction_end_date)

        if actual_data is None or len(actual_data) == 0:
            print("Failed to fetch actual data for the prediction period. Aborting process.")
            return

        # Generate predictions for the same dates as actual data
        predictions = predict_future(model, all_data['close'].values, scaler, len(actual_data), window_size)

        # Extract model equation
        weights = model.get_weights()
        bias = weights[-1][0]
        last_dense_weights = weights[-2]
        equation = f"y = {bias:.4f}"
        for i, w in enumerate(last_dense_weights):
            equation += f" + {w[0]:.4f} * x{i+1}"

        # Print predictions
        print("\n4-Hour Predictions:")
        for i, (pred_datetime, pred_close) in enumerate(zip(actual_data['time'], predictions)):
            print(f"Date: {pred_datetime}, Predicted Close: ${pred_close[0]:.2f}")

        # Prepare data for Excel export
        pred_df = pd.DataFrame({
            'DateTime': actual_data['time'],
            'Predicted_Close': predictions.flatten(),
            'Actual_Close': actual_data['close'],
            'Model_Equation': [equation] * len(actual_data)
        })
        pred_df.to_excel(f"4hour_predictions_report_{train_end_date.strftime('%Y%m%d')}.xlsx", index=False)
        print(f"4-hour predictions exported to 4hour_predictions_report_{train_end_date.strftime('%Y%m%d')}.xlsx")

        # Plotting
        plt.figure(figsize=(16, 8))
        plt.plot(actual_data['time'], predictions, color='red', linewidth=2, label='Prediction')
        plt.plot(actual_data['time'], actual_data['close'], color='green', linewidth=2, label='Actual Price')
        plt.xlabel('Date and Time')
        plt.ylabel('Price')
        plt.title(f'{symbol} 4-Hour Price Prediction vs Actual (Starting from {prediction_start_date})')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'4hour_price_prediction_plot_{train_end_date.strftime("%Y%m%d")}.png')
        plt.close()
        print(f"4-hour price prediction plot saved as 4hour_price_prediction_plot_{train_end_date.strftime('%Y%m%d')}.png")

        print(f"Next relearning scheduled for: {next_relearn_date}")

        # Wait until it's time to relearn
        while datetime.now() < next_relearn_date:
            time.sleep(3600)  # Sleep for 1 hour

        # Update the training data and prepare for the next iteration
        train_end_date = next_relearn_date
        next_relearn_date += relearn_interval