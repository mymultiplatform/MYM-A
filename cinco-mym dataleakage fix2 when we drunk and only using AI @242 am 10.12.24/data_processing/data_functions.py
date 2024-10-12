import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime

def fetch_historical_data(mt5, symbol, timeframe, start_date, end_date):
    # Convert date strings to datetime objects if they're not already
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    print(f"Attempting to fetch data for {symbol} from {start_date} to {end_date}")
    
    # Fetch historical data
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        print(f"Failed to retrieve rates for {symbol} from {start_date} to {end_date}")
        raise Exception("Failed to retrieve rates")
    
    # Create DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"Successfully fetched {len(df)} data points")
    return df[['time', 'close']]


def preprocess_data(data, window_size=60, scaler=None):
    if len(data) < window_size:
        print(f"Not enough data for preprocessing. Required: {window_size}, Available: {len(data)}")
        return None, None

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
    else:
        scaled_data = scaler.transform(data)

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

def train_lstm_model(X_train, y_train):
    # Define LSTM model
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0)
    return model

def predict_future(model, data, scaler, future_days):
    input_seq = data[-60:] # Assuming 60 is the time_step
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
