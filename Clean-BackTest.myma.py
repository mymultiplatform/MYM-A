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
from datetime import datetime

# Use a non-interactive backend to suppress GUI warnings
matplotlib.use('Agg')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to send the image to Discord
def send_to_discord(image_data, content):
    webhook_url = 'https://discord.com/api/webhooks/1298744394324639804/eY5xzTrrXoBaYfOsCgdx1UC8Lf-Tvy1uVnSc40JwOHHS4YTnUz1sUBpuvgTdHJZNmatd'
    webhook = DiscordWebhook(url=webhook_url, content=content)

    # Attach the image as a file
    webhook.add_file(file=image_data, filename='chart.png')

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
        backtest_strategy()
    else:
        messagebox.showerror("Error", "Failed to connect to MetaTrader 5")
        mt5.shutdown()

# Backtest the trading strategy from January to June 2022
def backtest_strategy():
    symbol = "BTCUSD"
    timeframe = mt5.TIMEFRAME_M30
    utc_from = datetime(2022, 1, 1)
    utc_to = datetime(2022, 6, 30)

    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
    if rates is None or len(rates) == 0:
        logging.error("Failed to retrieve historical rates for backtesting")
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Calculate EMAs
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

    # Generate signals
    df['signal'] = 0
    df['signal'][20:] = np.where(df['ema20'][20:] > df['ema50'][20:], 1, -1)
    df['position'] = df['signal'].shift(1)
    df['position'].fillna(0, inplace=True)

    # Calculate returns
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'] * df['returns']

    # Calculate cumulative returns
    df['cumulative_returns'] = (1 + df['returns']).cumprod()
    df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod()

    # Performance metrics
    total_trades = df['position'].diff().abs().sum()
    net_profit = df['strategy_returns'].sum()
    win_rate = (df['strategy_returns'] > 0).sum() / total_trades if total_trades != 0 else 0
    max_drawdown = (df['cumulative_strategy'].cummax() - df['cumulative_strategy']).max()

    performance_summary = (
        f"Backtest Results for {symbol} from {utc_from.strftime('%Y-%m-%d')} to {utc_to.strftime('%Y-%m-%d')}\n"
        f"Total Trades: {int(total_trades)}\n"
        f"Net Profit: {net_profit:.2%}\n"
        f"Win Rate: {win_rate:.2%}\n"
        f"Max Drawdown: {max_drawdown:.2%}\n"
    )
    logging.info(performance_summary)

    # Plotting the strategy
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df['time'], df['cumulative_returns'], label='Buy and Hold', color='blue')
    ax.plot(df['time'], df['cumulative_strategy'], label='EMA Strategy', color='orange')

    # Plot buy signals
    buy_signals = df[(df['signal'] == 1) & (df['position'] == 0)]
    ax.scatter(buy_signals['time'], df['cumulative_strategy'][buy_signals.index], marker='^', color='green', label='Buy Signal', s=100)

    # Plot sell signals
    sell_signals = df[(df['signal'] == -1) & (df['position'] == 1)]
    ax.scatter(sell_signals['time'], df['cumulative_strategy'][sell_signals.index], marker='v', color='red', label='Sell Signal', s=100)

    ax.set_title(f"EMA Strategy Backtest for {symbol}", fontsize=16)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Cumulative Returns", fontsize=14)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Close the plot to free memory
    plt.close(fig)

    # Send the performance summary and plot to Discord
    send_to_discord(buf.getvalue(), content=performance_summary)

    # Optionally, display a message box with the summary
    messagebox.showinfo("Backtest Completed", performance_summary)

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
        import numpy as np  # Import here to avoid issues if not already imported
        main()
    except Exception as e:
        logging.exception("An unexpected error occurred:")
        sys.exit(1)
