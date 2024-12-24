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
import numpy as np
from datetime import datetime, timedelta

def fetch_historical_data(mt5, symbol, timeframe, start_date, end_date):
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        raise Exception("Failed to retrieve rates")
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df[['time', 'close']]

def preprocess_data(data):
    if len(data) == 0:
        print("No data to preprocess")
        return None, None
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))
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
    if X_train.shape[0] == 0 or X_train.shape[1] == 0:
        print("X_train has invalid shape:", X_train.shape)
        return None, None
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train



def train_lstm_model(scaled_data):
    time_step = 60
    X_train, y_train = create_train_data(scaled_data, time_step)
    
    if X_train is None or y_train is None:
        print("Unable to train model due to insufficient data")
        return None

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0)
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
