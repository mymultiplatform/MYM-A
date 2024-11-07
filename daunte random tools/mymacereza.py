
import tkinter as tk
from tkinter import messagebox
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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
    login_entry.insert(0, "5027341082")  # Pre-fill with your login

    password_label = tk.Label(login_frame, text="Password:", font=("Helvetica", 14), bg="lightgrey")
    password_label.pack(pady=5)

    password_entry = tk.Entry(login_frame, show="*", font=("Helvetica", 14))
    password_entry.pack(pady=5)
    password_entry.insert(0, "IxS@Po1b")  # Pre-fill with your password

    server_label = tk.Label(login_frame, text="Server:", font=("Helvetica", 14), bg="lightgrey")
    server_label.pack(pady=5)

    server_entry = tk.Entry(login_frame, font=("Helvetica", 14))
    server_entry.pack(pady=5)
    server_entry.insert(0, "MetaQuotes-Demo")  # Pre-fill with your server

    # Connect button
    connect_button = tk.Button(login_frame, text="Connect", font=("Helvetica", 14),
                               command=lambda: connect_to_mt5(login_entry.get(), password_entry.get(), server_entry.get(), root))
    connect_button.pack(pady=20)

    # Create a frame for the main content
    main_frame = tk.Frame(root, width=800, height=600)
    main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Create a label with the text "USD/MX"
    label = tk.Label(main_frame, text="USD/MXN", font=("Helvetica", 24))
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
        mt5.shutdown()
        return

    # Log in to the MetaTrader 5 account
    authorized = mt5.login(login=int(login), password=password, server=server)
    if authorized:
        print("Connected to MetaTrader 5")
        display_account_info(root)
        fetch_and_display_chart(root)
    else:
        messagebox.showerror("Error", "Failed to connect to MetaTrader 5")

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
        f"Leverage: {account_info.leverage}"
    ]

    for info in info_labels:
        label = tk.Label(info_window, text=info, font=("Helvetica", 14))
        label.pack(pady=5)

def fetch_and_display_chart(root):
    symbol = "USDMXN"
    timeframe = mt5.TIMEFRAME_D1
    days = 600  # Fetch 600 days of data

    data = fetch_historical_data(symbol, timeframe, days)
    scaled_data, scaler = preprocess_data(data)

    # Train the LSTM model
    model = train_lstm_model(scaled_data)

    # Predict the future prices
    future_days = 60
    predicted_prices = predict_future(model, scaled_data, future_days)
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

    # Append predicted prices to the original data
    predicted_dates = pd.date_range(start=data['time'].iloc[-1], periods=future_days + 1, closed='right')
    predicted_df = pd.DataFrame({'time': predicted_dates, 'close': predicted_prices.flatten()})

    # Plot and display the chart with peaks and predictions
    plot_predictions(data, predicted_df, root)

def fetch_historical_data(symbol, timeframe, days):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, days)
    if rates is None or len(rates) == 0:
        raise Exception("Failed to retrieve rates")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df[['time', 'close']]

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))
    return scaled_data, scaler

def train_lstm_model(scaled_data):
    time_step = 60
    X_train, y_train = create_train_data(scaled_data, time_step)

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, batch_size=32, epochs=50)  # Increased epochs for better training
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

def plot_predictions(data, predicted_df, root):
    combined_df = pd.concat([data, predicted_df])

    fig, ax = plt.subplots()
    ax.plot(combined_df['time'], combined_df['close'], label="USD/MXN")
    ax.plot(predicted_df['time'], predicted_df['close'], 'r--', label="Predicted USD/MXN")

    peaks, _ = find_peaks(data['close'])
    ax.plot(data['time'].iloc[peaks], data['close'].iloc[peaks], "ro", markersize=5)  # Red circles at peaks

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("USD/MXN Daily Chart with Predictions")
    ax.legend()

    chart_frame = tk.Frame(root)
    chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    main()
