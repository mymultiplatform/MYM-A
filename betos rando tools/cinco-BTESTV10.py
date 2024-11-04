import MetaTrader5 as mt5
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import tkinter as tk

# Initialize MetaTrader 5 and set up the connection
def main():
    if not mt5.initialize():
        raise Exception("MetaTrader5 initialization failed")

    login = 312128713
    password = "Sexo247420@"
    server = "XMGlobal-MT5 7"

    if mt5.login(login=login, password=password, server=server):
        print("Connected to MetaTrader 5 for data retrieval")
        start_animation()
    else:
        raise Exception("Failed to connect to MetaTrader 5")

def fetch_historical_data(symbol, timeframe, start_date, days=1):
    # Fetch historical data for the specified symbol and date range
    utc_from = pd.Timestamp(start_date).to_pydatetime()
    rates = mt5.copy_rates_from(symbol, timeframe, utc_from, days * 48)  # 48 bars in 30 mins for 1 day
    if rates is None or len(rates) == 0:
        raise Exception(f"Failed to retrieve rates for {start_date}")
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df[['time', 'close']]

def plot_day_chart(day, data, trades, ax):
    ax.clear()
    ax.plot(data['time'], data['close'], color='black', label='Close Price')
    
    # Plot trade entry and exit points
    for trade in trades:
        entry_time = trade['entry_time']
        exit_time = trade['exit_time']
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        ax.plot(entry_time, entry_price, 'go', label='Entry' if trade['type'] == 'Buy' else 'Sell', markersize=10)
        ax.plot(exit_time, exit_price, 'ro', label='Exit', markersize=10)

    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title(f'BTCUSD Price Chart - {day.strftime("%Y-%m-%d")}')
    ax.legend()
    ax.grid()

def start_animation():
    symbol = "BTCUSD"
    timeframe = mt5.TIMEFRAME_M30  # 30-minute timeframe

    # Start and end dates
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-09-07')

    # Create a range of dates with a daily frequency
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    total_days = len(date_range)

    # Fetch all the data before starting the animation
    historical_data = {}
    trades = []
    balance = 1000  # Starting balance in USD

    for current_day in date_range[:250]:  # Limit to 250 days
        try:
            data = fetch_historical_data(symbol, timeframe, current_day)
            historical_data[current_day] = data
            
            # Simulate a trade for the day
            trade_type = random.choice(['Buy', 'Sell'])  # Randomly choose trade type
            entry_price = data['close'].iloc[0]  # Use the first price as entry price
            exit_price = entry_price * (1 + (0.02 if trade_type == 'Buy' else -0.02))  # Simulate a 2% move
            
            trades.append({
                'entry_time': data['time'].iloc[0],
                'exit_time': data['time'].iloc[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'type': trade_type
            })
            
            # Update balance based on trade outcome
            if trade_type == 'Buy' and exit_price > entry_price:
                balance *= 1.02  # Gain 2%
            elif trade_type == 'Sell' and exit_price < entry_price:
                balance *= 1.02  # Gain 2%
            else:
                balance *= 0.98  # Loss 2%

        except Exception as e:
            print(f"Error fetching data for {current_day.strftime('%Y-%m-%d')}: {e}")
            historical_data[current_day] = None

    # Setup the figure for plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    def update(frame):
        current_day = date_range[frame]
        data = historical_data[current_day]

        if data is None:
            print(f"Skipping day {current_day.strftime('%Y-%m-%d')} due to data issue")
            return

        # Plot the data for the current day
        plot_day_chart(current_day, data, trades, ax)
        print(f"Displayed chart for {current_day.strftime('%Y-%m-%d')} (Day {frame + 1}/{total_days})")

    # Set up the animation with a 100ms delay between frames (10 charts per second)
    ani = FuncAnimation(fig, update, frames=total_days, interval=100, repeat=False)

    plt.show()

def show_ui():
    root = tk.Tk()
    root.title("BTCUSD Daily Chart Animation")
    root.geometry("300x200")

    start_button = tk.Button(root, text="Start Animation", command=main, font=("Helvetica", 14))
    start_button.pack(pady=50)

    root.mainloop()

if __name__ == "__main__":
    show_ui()
