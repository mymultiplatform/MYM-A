import MetaTrader5 as mt5
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

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

def plot_day_chart(day, data, ax):
    ax.clear()
    ax.plot(data['time'], data['close'], color='black', label='Close Price')
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

    # Setup the figure for plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # Variable to track the frame number and status
    frame_status = {'plotted': False}

    def update(frame):
        # Get the current day from the date_range
        current_day = date_range[frame]

        # Fetch historical data for the current day
        try:
            data = fetch_historical_data(symbol, timeframe, current_day)
            frame_status['plotted'] = True  # Set the status as plotted
            print(f"Plotting chart for {current_day.strftime('%Y-%m-%d')} (Day {frame + 1}/{total_days})")
        except Exception as e:
            frame_status['plotted'] = False
            print(f"Error fetching data for {current_day.strftime('%Y-%m-%d')}: {e}")
            return

        # Plot the data for the current day
        plot_day_chart(current_day, data, ax)

    # Set up the animation with a 5-second delay between frames
    ani = FuncAnimation(fig, update, frames=total_days, interval=5000, repeat=False)

    plt.show()

def show_ui():
    import tkinter as tk

    root = tk.Tk()
    root.title("BTCUSD Daily Chart Animation")
    root.geometry("300x200")

    start_button = tk.Button(root, text="Start Animation", command=main, font=("Helvetica", 14))
    start_button.pack(pady=50)

    root.mainloop()

if __name__ == "__main__":
    show_ui()
