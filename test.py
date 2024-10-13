import tkinter as tk
from tkinter import messagebox, ttk
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import pandas as pd

class MT5DataChecker:
    def __init__(self, master):
        self.master = master
        master.title("MT5 Data Checker")
        master.geometry("600x500")

        self.create_widgets()

    def create_widgets(self):
        # Login Frame
        self.login_frame = ttk.Frame(self.master, padding="10")
        self.login_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(self.login_frame, text="🏧MT5 Data Checker", font=("Helvetica", 20)).grid(column=0, row=0, columnspan=2, pady=20)

        ttk.Label(self.login_frame, text="Login:").grid(column=0, row=1, sticky=tk.W, pady=5)
        self.login_entry = ttk.Entry(self.login_frame)
        self.login_entry.grid(column=1, row=1, sticky=(tk.W, tk.E), pady=5)
        self.login_entry.insert(0, "312128713")

        ttk.Label(self.login_frame, text="Password:").grid(column=0, row=2, sticky=tk.W, pady=5)
        self.password_entry = ttk.Entry(self.login_frame, show="*")
        self.password_entry.grid(column=1, row=2, sticky=(tk.W, tk.E), pady=5)
        self.password_entry.insert(0, "Sexo247420@")

        ttk.Label(self.login_frame, text="Server:").grid(column=0, row=3, sticky=tk.W, pady=5)
        self.server_entry = ttk.Entry(self.login_frame)
        self.server_entry.grid(column=1, row=3, sticky=(tk.W, tk.E), pady=5)
        self.server_entry.insert(0, "XMGlobal-MT5 7")

        self.connect_button = ttk.Button(self.login_frame, text="Connect", command=self.connect_to_mt5)
        self.connect_button.grid(column=0, row=4, columnspan=2, pady=20)

        self.status_label = ttk.Label(self.login_frame, text="")
        self.status_label.grid(column=0, row=5, columnspan=2, pady=10)

        # Data Checker Frame (initially hidden)
        self.data_frame = ttk.Frame(self.master, padding="10")

        ttk.Label(self.data_frame, text="Symbol:").grid(column=0, row=0, sticky=tk.W, pady=5)
        self.symbol_entry = ttk.Entry(self.data_frame)
        self.symbol_entry.grid(column=1, row=0, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(self.data_frame, text="Timeframe:").grid(column=0, row=1, sticky=tk.W, pady=5)
        self.timeframe_combo = ttk.Combobox(self.data_frame, values=["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"])
        self.timeframe_combo.grid(column=1, row=1, sticky=(tk.W, tk.E), pady=5)
        self.timeframe_combo.set("D1")

        self.check_button = ttk.Button(self.data_frame, text="Check Data", command=self.check_data)
        self.check_button.grid(column=0, row=2, columnspan=2, pady=20)

        self.result_text = tk.Text(self.data_frame, height=10, width=50)
        self.result_text.grid(column=0, row=3, columnspan=2, pady=10)

        self.save_button = ttk.Button(self.data_frame, text="Save Data", command=self.save_data)
        self.save_button.grid(column=0, row=4, columnspan=2, pady=10)

    def connect_to_mt5(self):
        login = int(self.login_entry.get())
        password = self.password_entry.get()
        server = self.server_entry.get()

        if not mt5.initialize():
            self.update_status("Failed to initialize MT5")
            return

        if mt5.login(login=login, password=password, server=server):
            self.update_status("Connected to MetaTrader 5")
            self.login_frame.grid_remove()
            self.data_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        else:
            self.update_status("Failed to connect to MT5")

    def update_status(self, message):
        self.status_label.config(text=message)

    def check_data(self):
        symbol = self.symbol_entry.get().upper()
        timeframe = self.get_timeframe()

        result, self.current_data = self.check_data_availability(symbol, timeframe)
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END, result)

    def get_timeframe(self):
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1
        }
        return timeframe_map.get(self.timeframe_combo.get(), mt5.TIMEFRAME_D1)

    def check_data_availability(self, symbol, timeframe):
        if not mt5.symbol_select(symbol, True):
            return f"Failed to select symbol: {symbol}", None

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*20)  # Try to fetch 20 years of data

        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            return f"No data available for {symbol}", None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        oldest_date = df['time'].min()
        newest_date = df['time'].max()
        total_days = (newest_date - oldest_date).days
        
        result = f"Data available for {symbol}:\n"
        result += f"Oldest date: {oldest_date}\n"
        result += f"Newest date: {newest_date}\n"
        result += f"Total days: {total_days}\n"
        result += f"Total data points: {len(df)}\n"
        
        return result, df

    def save_data(self):
        if hasattr(self, 'current_data') and self.current_data is not None:
            symbol = self.symbol_entry.get().upper()
            timeframe = self.timeframe_combo.get()
            filename = f"{symbol}_{timeframe}_data.csv"
            self.current_data.to_csv(filename, index=False)
            messagebox.showinfo("Save Complete", f"Data saved to {filename}")
        else:
            messagebox.showwarning("No Data", "No data available to save. Please check data first.")

def main():
    root = tk.Tk()
    app = MT5DataChecker(root)
    root.mainloop()

if __name__ == "__main__":
    main()