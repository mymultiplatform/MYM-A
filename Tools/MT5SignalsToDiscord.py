import tkinter as tk
from tkinter import messagebox
import MetaTrader5 as mt5
import pandas as pd
import matplotlib.pyplot as plt
import threading
import time
from discord_webhook import DiscordWebhook
import io

# Function to send the image to Discord
def send_to_discord(image_data):
    webhook_url = 'https://discord.com/api/webhooks/1298744394324639804/eY5xzTrrXoBaYfOsCgdx1UC8Lf-Tvy1uVnSc40JwOHHS4YTnUz1sUBpuvgTdHJZNmatd'
    webhook = DiscordWebhook(url=webhook_url, content="Here is the latest candlestick chart with the largest candle marked.")
    
    # Attach the image as a file
    webhook.add_file(file=image_data, filename='candlestick_chart.png')
    
    response = webhook.execute()
    if response.status_code == 200:
        print("Image sent to Discord successfully!")
    else:
        print(f"Failed to send image. Status code: {response.status_code}")

# Connect to MetaTrader 5
def connect_to_mt5(login, password, server):
    if not mt5.initialize():
        messagebox.showerror("Error", "initialize() failed")
        mt5.shutdown()
        return

    authorized = mt5.login(login=int(login), password=password, server=server)
    if authorized:
        print("Connected to MetaTrader 5")
        start_monitoring()
    else:
        messagebox.showerror("Error", "Failed to connect to MetaTrader 5")

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
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 30)  # Get the last 30 candles (30 minutes)
    if rates is None or len(rates) == 0:
        print("Failed to retrieve rates")
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    df['body_size'] = abs(df['close'] - df['open'])  # Size of candlestick body
    
    # Find the largest candlestick body
    largest_candle_idx = df['body_size'].idxmax()
    largest_candle = df.iloc[largest_candle_idx]

    # Plot the candlesticks and mark the largest one
    img_data = plot_candlesticks(df, largest_candle)
    
    # Send the image to Discord
    send_to_discord(img_data)

# Plot the 30-minute candlestick chart and mark the largest candle
def plot_candlesticks(df, largest_candle):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the candlesticks
    for idx, row in df.iterrows():
        color = 'green' if row['close'] > row['open'] else 'red'
        ax.bar(row['time'], row['close'] - row['open'], width=0.03, bottom=row['open'], color=color, edgecolor='black')
    
    # Mark the largest candle with a yellow star
    ax.plot(largest_candle['time'], largest_candle['close'], marker='*', color='yellow', markersize=15, label='Largest Candle')
    
    ax.set_title("30-Minute Candlestick Chart")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()

    # Save the plot to a buffer instead of showing it
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)  # Move to the beginning of the buffer
    
    # Close the plot to avoid showing it in the UI
    plt.close(fig)
    
    return buf.getvalue()  # Return the image data in bytes

# Tkinter UI setup
def main():
    global root, connect_button
    root = tk.Tk()
    root.title("MYM-A Synthetic Signal Alerts")
    root.geometry("400x300")

    # Login UI
    login_frame = tk.Frame(root)
    login_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    login_title = tk.Label(login_frame, text="🏧MYM-A", font=("Helvetica", 20))
    login_title.pack(pady=20)

    login_label = tk.Label(login_frame, text="Login:", font=("Helvetica", 14))
    login_label.pack(pady=5)
    login_entry = tk.Entry(login_frame, font=("Helvetica", 14))
    login_entry.pack(pady=5)
    login_entry.insert(0, "312128713")

    password_label = tk.Label(login_frame, text="Password:", font=("Helvetica", 14))
    password_label.pack(pady=5)
    password_entry = tk.Entry(login_frame, show="*", font=("Helvetica", 14))
    password_entry.pack(pady=5)
    password_entry.insert(0, "Sexo247420@")

    server_label = tk.Label(login_frame, text="Server:", font=("Helvetica", 14))
    server_label.pack(pady=5)
    server_entry = tk.Entry(login_frame, font=("Helvetica", 14))
    server_entry.pack(pady=5)
    server_entry.insert(0, "XMGlobal-MT5 7")

    connect_button = tk.Button(login_frame, text="Connect", font=("Helvetica", 14),
                               command=lambda: connect_to_mt5(login_entry.get(), password_entry.get(), server_entry.get()))
    connect_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
