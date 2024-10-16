import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime, timedelta
from tqdm import tqdm
import MetaTrader5 as mt5
import matplotlib.pyplot as plt

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 50)
        self.fc2 = nn.Linear(50, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        return out

def fetch_historical_data(mt5, symbol, timeframe, start_date, end_date):
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    print(f"Attempting to fetch data for {symbol} from {start_date} to {end_date}")
    
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")
        return None

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to get symbol info for {symbol}")
        return None

    print(f"Symbol info: {symbol_info._asdict()}")

    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    
    if rates is None or len(rates) == 0:
        print(f"Failed to retrieve rates for {symbol} from {start_date} to {end_date}")
        
        if not mt5.terminal_info().connected:
            print("Not connected to MetaTrader 5. Please check your connection.")
        
        symbols = mt5.symbols_get()
        print(f"Available symbols: {[s.name for s in symbols]}")
        
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    
    dataset = TensorDataset(X_train, y_train)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    model = LSTMModel(input_size=1, hidden_size=200, num_layers=3, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    epochs = 1000
    patience = 400
    best_val_loss = float('inf')
    no_improve_epoch = 0
    
    print("Starting model training...")
    for epoch in tqdm(range(epochs), unit="epoch"):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y.unsqueeze(1)).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epoch = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve_epoch += 1
            if no_improve_epoch >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
    
    model.load_state_dict(torch.load('best_model.pth'))
    print("Model training completed.")
    return model

def predict_future(model, train_data, scaler, future_periods, window_size, mu, sigma, dt):
    device = next(model.parameters()).device
    model.eval()
    input_seq = train_data[-window_size:]
    predictions = []
    last_price = scaler.inverse_transform(input_seq[-1].reshape(-1, 1))[0][0]
    
    for i in range(future_periods):
        input_seq_scaled = scaler.transform(input_seq.reshape(-1, 1))
        input_seq_tensor = torch.FloatTensor(input_seq_scaled).unsqueeze(0).to(device)
        with torch.no_grad():
            predicted_price = model(input_seq_tensor).item()
        
        # Check if it's a trading period (assuming 24/5 market)
        is_trading_period = i % (6 * 5) < (6 * 5 - 6)  # Exclude weekends
        
        if is_trading_period:
            # Apply Geometric Brownian Motion for trading periods
            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * np.random.normal(0, 1)
            price_multiplier = np.exp(drift + diffusion)
            stochastic_price = last_price * price_multiplier
        else:
            # For non-trading periods, keep the price unchanged
            stochastic_price = last_price

        predictions.append(stochastic_price)
        
        input_seq = np.append(input_seq[1:], stochastic_price)
        last_price = stochastic_price

    return np.array(predictions).reshape(-1, 1)

def run_trading_process():
    symbol = "USDCHF"
    timeframe = mt5.TIMEFRAME_H4
    start_date = datetime(2010, 1, 4)
    train_end_date = datetime(2021, 12, 31, 20, 0)
    window_size = 1

    if not mt5.terminal_info().connected:
        print("Not connected to MetaTrader 5. Please check your connection.")
        return

    all_data = fetch_historical_data(mt5, symbol, timeframe, start_date, train_end_date)
    if all_data is None or len(all_data) == 0:
        print("Failed to fetch historical data. Aborting process.")
        return

    print(f"Training data from {all_data.iloc[0]['time']} to {all_data.iloc[-1]['time']}")
    print(f"Total data points: {len(all_data)}")

    if len(all_data) < window_size:
        print(f"Not enough data points. Required: {window_size}, Available: {len(all_data)}")
        return

    scaled_train_data, scaler = preprocess_data(all_data['close'].values, window_size, is_training=True)
    if scaler is None:
        print("Failed to scale training data")
        return

    X_train, y_train = create_train_data(scaled_train_data, window_size)
    if X_train is None or y_train is None:
        print("Failed to create training data")
        return

    model = train_lstm_model(X_train, y_train, window_size)
    if model is None:
        print("Failed to train model")
        return

    last_close = all_data.iloc[-1]['close']
    last_date = all_data.iloc[-1]['time']

    prediction_start_date = last_date + timedelta(hours=4)
    prediction_end_date = prediction_start_date + timedelta(days=1016)
    actual_data = fetch_historical_data(mt5, symbol, timeframe, prediction_start_date, prediction_end_date)

    if actual_data is None or len(actual_data) == 0:
        print("Failed to fetch actual data for the prediction period. Aborting process.")
        return

 # Adjust these parameters for forex prediction
    mu = 0  # No drift
    sigma = 0.0001  # Very low volatility
    dt = 1 / (252 * 6)  # Time step for 4-hour intervals, considering 252 trading days per year

    predictions = predict_future(model, all_data['close'].values, scaler, len(actual_data), window_size, mu, sigma, dt)

    # Extract model equation (this part is different for PyTorch)
    last_layer = list(model.children())[-1]
    weights = last_layer.weight.data.cpu().numpy()
    bias = last_layer.bias.data.cpu().numpy()
    equation = f"y = {bias[0]:.4f}"
    for i, w in enumerate(weights[0]):
        equation += f" + {w:.4f} * x{i+1}"

    print("\n4-Hour Predictions:")
    for i, (pred_datetime, pred_close) in enumerate(zip(actual_data['time'], predictions)):
        print(f"Date: {pred_datetime}, Predicted Close: ${pred_close[0]:.2f}")

    pred_df = pd.DataFrame({
        'DateTime': actual_data['time'],
        'Predicted_Close': predictions.flatten(),
        'Actual_Close': actual_data['close'],
        'Model_Equation': [equation] * len(actual_data)
    })
    pred_df.to_excel("4hour_predictions_report.xlsx", index=False)
    print("4-hour predictions exported to 4hour_predictions_report.xlsx")

    plt.figure(figsize=(16, 8))
    plt.plot(actual_data['time'], predictions, color='red', linewidth=2, label='Prediction')
    plt.plot(actual_data['time'], actual_data['close'], color='green', linewidth=2, label='Actual Price')
    plt.xlabel('Date and Time')
    plt.ylabel('Price')
    plt.title(f'{symbol} 4-Hour Price Prediction vs Actual (Starting from {prediction_start_date})')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('4hour_price_prediction_plot.png')
    plt.close()
    print("4-hour price prediction plot saved as 4hour_price_prediction_plot.png")