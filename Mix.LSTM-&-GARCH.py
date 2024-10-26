import tkinter as tk
from tkinter import messagebox
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import threading
from arch import arch_model

def main():
    # Create the main window
    root = tk.Tk()
    root.title("MT5MYM-A")
    root.geometry("1200x800")

    # Create a frame for the login UI
    login_frame = tk.Frame(root, width=400, height=600, bg="lightgrey")
    login_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Title for the login UI
    login_title = tk.Label(login_frame, text="🏧MYM-A", font=("Helvetica", 20), bg="lightgrey")
    login_title.pack(pady=20)

    # Login fields
    login_label = tk.Label(login_frame, text="Login:", font=("Helvetica", 14), bg="lightgrey")
    login_label.pack(pady=5)

    login_entry = tk.Entry(login_frame, font=("Helvetica", 14))
    login_entry.pack(pady=5)
    login_entry.insert(0, "")  # Leave empty for user input

    password_label = tk.Label(login_frame, text="Password:", font=("Helvetica", 14), bg="lightgrey")
    password_label.pack(pady=5)

    password_entry = tk.Entry(login_frame, show="*", font=("Helvetica", 14))
    password_entry.pack(pady=5)
    password_entry.insert(0, "")  # Leave empty for user input

    server_label = tk.Label(login_frame, text="Server:", font=("Helvetica", 14), bg="lightgrey")
    server_label.pack(pady=5)

    server_entry = tk.Entry(login_frame, font=("Helvetica", 14))
    server_entry.pack(pady=5)
    server_entry.insert(0, "")  # Leave empty for user input

    # Connect button
    connect_button = tk.Button(
        login_frame,
        text="Connect",
        font=("Helvetica", 14),
        command=lambda: threading.Thread(target=connect_to_mt5, args=(
            login_entry.get(), password_entry.get(), server_entry.get(), root
        )).start()
    )
    connect_button.pack(pady=20)

    # Create a frame for the main content
    main_frame = tk.Frame(root, width=800, height=600)
    main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Create a label with the text "BTC/USD"
    label = tk.Label(main_frame, text="BTC/USD", font=("Helvetica", 24))
    label.pack(expand=True)

    # Create a label with the text "Loading..." at the bottom
    loading_label = tk.Label(main_frame, text="Loading...", font=("Helvetica", 14))
    loading_label.pack(side=tk.BOTTOM, pady=10)

    # Start the Tkinter event loop
    root.mainloop()

def connect_to_mt5(login, password, server, root):
    # Initialize MetaTrader 5
    if not mt5.initialize():
        messagebox.showerror("Error", "initialize() failed")
        return

    # Log in to the MetaTrader 5 account
    authorized = mt5.login(login=int(login), password=password, server=server)
    if authorized:
        print("Connected to MetaTrader 5")
        display_account_info(root)
        fetch_and_display_chart(root)
        mt5.shutdown()  # Shutdown after fetching data
    else:
        messagebox.showerror("Error", "Failed to connect to MetaTrader 5")
        mt5.shutdown()

def display_account_info(root):
    # Get account info
    account_info = mt5.account_info()
    if account_info is None:
        messagebox.showerror("Error", "Failed to get account info")
        return

    # Create a new window for account information
    info_window = tk.Toplevel(root)
    info_window.title("Account Information")
    info_window.geometry("400x300")

    # Display account information
    info_labels = [
        f"Account ID: {account_info.login}",
        f"Balance: {account_info.balance}",
        f"Equity: {account_info.equity}",
        f"Margin: {account_info.margin}",
        f"Free Margin: {account_info.margin_free}",
        f"Leverage: {account_info.leverage}",
    ]

    for info in info_labels:
        label = tk.Label(info_window, text=info, font=("Helvetica", 14))
        label.pack(pady=5)

def fetch_and_display_chart(root):
    symbol = "BTCUSD"
    timeframe = mt5.TIMEFRAME_D1

    # Dates for historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    prediction_start_date = end_date
    future_days = 40  # Predict 40 days into the future
    prediction_end_date = end_date + timedelta(days=future_days)

    data = fetch_historical_data(symbol, timeframe, start_date, end_date)
    if data is None or data.empty:
        messagebox.showerror("Error", "Failed to fetch historical data")
        return

    scaled_data, scaler = preprocess_data(data['close'].values)
    if scaled_data is None:
        messagebox.showerror("Error", "Failed to preprocess data")
        return

    # Train the LSTM model
    lstm_model = train_lstm_model(scaled_data)
    if lstm_model is None:
        messagebox.showerror("Error", "Failed to train LSTM model")
        return

    # Predict the future prices using LSTM
    lstm_predictions = predict_future_prices(lstm_model, scaled_data, scaler, future_days)
    lstm_predictions = lstm_predictions.flatten()

    # Prepare data for GARCH model
    historical_prices = pd.Series(data['close'].values, index=data['time'])
    returns = calculate_returns(historical_prices)

    # Fit GARCH model
    garch_model_fit = fit_garch_model(returns)
    if garch_model_fit is None:
        messagebox.showerror("Error", "Failed to fit GARCH model")
        return

    # Forecast future volatility using GARCH
    garch_volatility = forecast_volatility(garch_model_fit, future_days)

    # Combine LSTM predictions with GARCH volatility
    combined_predictions = combine_predictions(lstm_predictions, garch_volatility)

    # Prepare dates for the predictions
    prediction_dates = pd.date_range(start=prediction_start_date + timedelta(days=1), periods=future_days)

    # Create DataFrame for predictions
    predicted_df = pd.DataFrame({'time': prediction_dates, 'close': combined_predictions.flatten()})

    # Plot and display the chart with combined predictions
    plot_predictions(data, predicted_df, root)

def fetch_historical_data(symbol, timeframe, start_date, end_date):
    utc_from = start_date
    utc_to = end_date
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
    if rates is None or len(rates) == 0:
        print("Failed to retrieve rates")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df[['time', 'close']]

def preprocess_data(data):
    if len(data) < 61:  # Minimum data length required (window_size + 1)
        print("Not enough data to preprocess")
        return None, None
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = data.reshape(-1, 1)
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def train_lstm_model(scaled_data):
    window_size = 60
    X_train, y_train = create_train_data(scaled_data, window_size)
    if X_train.size == 0 or y_train.size == 0:
        print("Not enough training data")
        return None

    # Debugging: Print shapes
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)

    model = LSTMModel(input_size=1, hidden_size=100, num_layers=2, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.view(-1), y_train)
        loss.backward()
        optimizer.step()

        # Debugging: Print loss
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    return model

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def create_train_data(scaled_data, window_size):
    X_train, y_train = [], []
    if len(scaled_data) < window_size:
        return np.array(X_train), np.array(y_train)
    for i in range(window_size, len(scaled_data)):
        X_train.append(scaled_data[i - window_size:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train

def predict_future_prices(model, data, scaler, future_days):
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    window_size = 60
    input_seq = data[-window_size:]

    input_seq = torch.FloatTensor(input_seq).to(device)
    input_seq = input_seq.unsqueeze(0)  # Shape: (1, 60, 1)

    for _ in range(future_days):
        with torch.no_grad():
            pred = model(input_seq)  # Shape: (1, 1)
        predictions.append(pred.item())
        new_input = pred.unsqueeze(2)  # Corrected shape to (1, 1, 1)

        # Concatenate along the sequence dimension
        input_seq = torch.cat((input_seq[:, 1:, :], new_input), dim=1)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

def calculate_returns(prices):
    returns = np.log(prices / prices.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return returns

def fit_garch_model(returns):
    try:
        model = arch_model(returns, vol='Garch', p=1, q=1)
        model_fit = model.fit(disp='off')
        return model_fit
    except Exception as e:
        print(f"GARCH model fitting failed: {e}")
        return None

def forecast_volatility(model_fit, horizon):
    forecast = model_fit.forecast(horizon=horizon)
    volatility = np.sqrt(forecast.variance.values[-1, :])
    return volatility

def combine_predictions(lstm_preds, garch_volatility):
    # Ensure the lengths match
    min_length = min(len(lstm_preds), len(garch_volatility))
    lstm_preds = lstm_preds[:min_length]
    garch_volatility = garch_volatility[:min_length]

    # Adjust shapes
    lstm_preds = lstm_preds.reshape(-1, 1)
    garch_volatility = garch_volatility.reshape(-1, 1)

    # Adjust LSTM predictions using GARCH volatility
    combined_preds = lstm_preds * (1 + garch_volatility)
    return combined_preds

def plot_predictions(data, predicted_df, root):
    # Clear previous frames if any (ensure this is thread-safe)
    def update_gui():
        for widget in root.winfo_children():
            widget.destroy()

        # Recreate frames
        chart_frame = tk.Frame(root)
        chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['time'], data['close'], label="Historical BTC/USD")
        ax.plot(predicted_df['time'], predicted_df['close'], 'r--', label="Predicted BTC/USD")

        # Peaks in historical data
        peaks, _ = find_peaks(data['close'])
        ax.plot(data['time'].iloc[peaks], data['close'].iloc[peaks], "ro", markersize=5)

        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.set_title("BTC/USD Daily Chart with Combined LSTM and GARCH Predictions")
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Schedule the GUI update in the main thread
    root.after(0, update_gui)

if __name__ == "__main__":
    main()
