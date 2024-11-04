import tkinter as tk
from tkinter import messagebox
import MetaTrader5 as mt5
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import threading
import time
from discord_webhook import DiscordWebhook
import io
import sys
import logging

# Use a non-interactive backend to suppress GUI warnings
matplotlib.use('Agg')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to send the image to Discord
def send_to_discord(image_data):
    webhook_url = 'https://discord.com/api/webhooks/1298744394324639804/eY5xzTrrXoBaYfOsCgdx1UC8Lf-Tvy1uVnSc40JwOHHS4YTnUz1sUBpuvgTdHJZNmatd'
    webhook = DiscordWebhook(url=webhook_url, content="Here is the latest candlestick chart with the largest candle marked (BTC).")

    # Attach the image as a file
    webhook.add_file(file=image_data, filename='candlestick_chart.png')

    response = webhook.execute()
    if response.status_code == 200:
        logging.info("Image sent to Discord successfully!")
    else:
        logging.error(f"Failed to send image. Status code: {response.status_code}")

# Connect to MetaTrader 5
def connect_to_mt5(login, password, server):
    if not mt5.initialize():
        messagebox.showerror("Error", "initialize() failed")
        mt5.shutdown()
        return

    authorized = mt5.login(login=int(login), password=password, server=server)
    if authorized:
        logging.info("Connected to MetaTrader 5")
        start_monitoring()
    else:
        messagebox.showerror("Error", "Failed to connect to MetaTrader 5")
        mt5.shutdown()

# Monitor and plot every 80 minutes
def start_monitoring():
    def monitor_loop():
        while True:
            symbol = "BTCUSD"
            timeframe = mt5.TIMEFRAME_M30
            plot_latest_candles(symbol, timeframe)
            time.sleep(80 * 60)  # Refresh every 80 minutes

    threading.Thread(target=monitor_loop, daemon=True).start()

# Fetch and plot the latest 30-minute candlesticks
def plot_latest_candles(symbol, timeframe):
    # Fetch enough candles to ensure proper spacing; adjust as needed
    num_candles = 80
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
    if rates is None or len(rates) == 0:
        logging.error("Failed to retrieve rates")
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Convert datetime to matplotlib's internal date format
    df['mpl_time'] = mdates.date2num(df['time'])

    df['body_size'] = abs(df['close'] - df['open'])  # Size of candlestick body

    # Find the largest candlestick body
    largest_candle_idx = df['body_size'].idxmax()
    largest_candle = df.iloc[largest_candle_idx]

    # Plot the candlesticks and mark the largest one
    img_data = plot_candlesticks(df, largest_candle, num_candles)

    # Send the image to Discord
    send_to_discord(img_data)

# Plot the candlestick chart and mark the largest candle
def plot_candlesticks(df, largest_candle, num_candles):
    fig, ax = plt.subplots(figsize=(12, 8))

    width = 0.02  # Adjusted width for better spacing

    for idx, row in df.iterrows():
        color = 'green' if row['close'] > row['open'] else 'red'
        # Real body
        ax.bar(row['mpl_time'], row['close'] - row['open'], width=width,
               bottom=min(row['open'], row['close']),
               color=color, edgecolor='black', linewidth=0.5)

        # Upper and lower shadows
        ax.vlines(row['mpl_time'], row['low'], row['high'], color='black', linewidth=0.5)

    # Mark the largest candle with a yellow star
    ax.plot(largest_candle['mpl_time'], largest_candle['close'], marker='*', color='yellow',
            markersize=15, label='Largest Candle')

    # Formatting the x-axis with date labels
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')

    # Adding grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    ax.set_title(f"{num_candles}-Candlestick Chart for {df['time'].iloc[-1].strftime('%Y-%m-%d %H:%M')}", fontsize=16)
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Price", fontsize=14)
    ax.legend()

    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels

    # Save the plot to a buffer instead of showing it
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)  # Move to the beginning of the buffer

    # Close the plot to free memory
    plt.close(fig)

    return buf.getvalue()  # Return the image data in bytes

# Tkinter UI setup
def main():
    global root, connect_button
    root = tk.Tk()
    root.title("MYM-A Synthetic Signal Alerts")
    root.geometry("400x450")  # Increased height for better layout

    # Login UI
    login_frame = tk.Frame(root)
    login_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    login_title = tk.Label(login_frame, text="🏧 MYM-A", font=("Helvetica", 24))
    login_title.pack(pady=20)

    # Login Entry
    login_label = tk.Label(login_frame, text="Login:", font=("Helvetica", 14))
    login_label.pack(anchor='w', pady=(10, 0))
    login_entry = tk.Entry(login_frame, font=("Helvetica", 14))
    login_entry.pack(fill=tk.X, pady=5)
    login_entry.insert(0, "312128713")

    # Password Entry
    password_label = tk.Label(login_frame, text="Password:", font=("Helvetica", 14))
    password_label.pack(anchor='w', pady=(10, 0))
    password_entry = tk.Entry(login_frame, show="*", font=("Helvetica", 14))
    password_entry.pack(fill=tk.X, pady=5)
    password_entry.insert(0, "Sexo247420@")

    # Server Entry
    server_label = tk.Label(login_frame, text="Server:", font=("Helvetica", 14))
    server_label.pack(anchor='w', pady=(10, 0))
    server_entry = tk.Entry(login_frame, font=("Helvetica", 14))
    server_entry.pack(fill=tk.X, pady=5)
    server_entry.insert(0, "XMGlobal-MT5 7")

    # Connect Button
    connect_button = tk.Button(login_frame, text="Connect", font=("Helvetica", 14),
                               command=lambda: connect_to_mt5(login_entry.get(), password_entry.get(), server_entry.get()))
    connect_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("An unexpected error occurred:")
        sys.exit(1)
