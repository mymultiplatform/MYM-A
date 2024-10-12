## data functions.py 

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime
import os
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

def preprocess_data(data, window_size=60, scaler=None, is_training=True):
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

def create_train_data(scaled_data, time_step):
    if scaled_data is None or len(scaled_data) == 0:
        print("No data to create training set")
        return None, None
    
    X_train, y_train = [], []
    for i in range(time_step, len(scaled_data)):
        X_train.append(scaled_data[i-time_step:i, 0])
        y_train.append(scaled_data[i, 0])
    
    if len(X_train) == 0 or len(y_train) == 0:
        print("Not enough data to create training set")
        return None, None
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback
from tqdm import tqdm

# Configure TensorFlow to use GPU
# Set TensorFlow logging level to show all messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

print("TensorFlow version:", tf.__version__)

# Configure TensorFlow to use GPU
physical_devices = tf.config.list_physical_devices('GPU')
print("Physical devices:", physical_devices)

if len(physical_devices) > 0:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("GPU is available and configured for use.")
        print(f"Found {len(physical_devices)} GPU(s):")
        for i, device in enumerate(physical_devices):
            print(f"  GPU {i}: {device}")
    except RuntimeError as e:
        print("Error configuring GPU:", e)
else:
    print("No GPU detected. Training will proceed on CPU.")

def train_lstm_model(X_train, y_train):
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    
    with strategy.scope():
        model = Sequential([
            LSTM(200, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(200, return_sequences=True),
            LSTM(100, return_sequences=False),
            Dense(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    
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
                early_stopping,
                TqdmCallback(verbose=0)
            ],
            verbose=0
        )
        pbar.update(len(history.history['loss']))
    
    print("Model training completed.")
    return model

# Verify GPU usage
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow is built with CUDA:", tf.test.is_built_with_cuda())

# Test GPU computation
if len(physical_devices) > 0:
    print("Testing GPU computation...")
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
    print("GPU computation test result:", c.numpy())
else:
    print("Skipping GPU computation test as no GPU is available.")
def predict_future(model, data, scaler, future_days):
    input_seq = data[-1:]
    predictions = []
    for _ in range(future_days):
        input_seq_scaled = scaler.transform(input_seq.reshape(-1, 1))
        input_seq_reshaped = input_seq_scaled.reshape((1, input_seq_scaled.shape[0], 1))
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

def train_model(train_data):
    if len(train_data) < 60:
        print("Not enough data to train the model")
        return None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_data[['close']].values)

    X_train, y_train = create_train_data(scaled_data, 30)
    if X_train is None or y_train is None:
        return None, None

    model = train_lstm_model(X_train, y_train)
    return model, scaler

def predict_future(model, data, scaler, future_days):
    input_seq = data[-5:].reshape(-30, 1)
    input_seq_scaled = scaler.transform(input_seq)
    input_seq_reshaped = input_seq_scaled.reshape((1, input_seq_scaled.shape[0], 1))
    predicted_scaled = model.predict(input_seq_reshaped)[0]
    return scaler.inverse_transform(predicted_scaled.reshape(-1, 1))

def determine_trend(current_price, predicted_price):
    if predicted_price > current_price:
        return "Bull"
    elif predicted_price < current_price:
        return "Bear"
    else:
        return "Neutral"