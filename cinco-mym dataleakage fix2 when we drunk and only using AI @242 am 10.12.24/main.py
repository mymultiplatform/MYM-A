import tkinter as tk
from tkinter import messagebox
import MetaTrader5 as mt5
import threading
from metatrader.mt5_functions import connect_to_mt5, run_trading_process
import numpy as np
import pandas as pd
from trading.trading_functions import export_trades_to_excel

def main():
    global root, connect_button, status_label

    root = tk.Tk()
    root.title("MYM-A MODO CHEZ")
    root.geometry("600x450")

    # Create frames
    login_frame = tk.Frame(root, width=600, height=450)
    login_frame.pack(fill=tk.BOTH, expand=True)

    # Setup Login UI
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
                               command=lambda: threading.Thread(target=connect_and_run, args=(login_entry.get(), password_entry.get(), server_entry.get())).start())
    connect_button.pack(pady=20)

    status_label = tk.Label(login_frame, text="", font=("Helvetica", 12))
    status_label.pack(pady=10)

    root.mainloop()

def connect_and_run(login, password, server):
    update_status("Connecting to MT5...")
    if connect_to_mt5(login, password, server):
        update_status("Connected. Starting trading process...")
        threading.Thread(target=run_trading_process).start()
    else:
        update_status("Failed to connect to MT5.")

def update_status(message):
    status_label.config(text=message)
    root.update_idletasks()

def export_to_excel():
    # Create data for each section
    ui_data = {
        "Title": ["MYM-A MODO CHEZ"],
        "Geometry": ["600x450"],
        "Frames": ["Login Frame"],
    }
    
    # Other data sections as needed
    
    # Create DataFrames
    ui_df = pd.DataFrame(ui_data)
    # Other DataFrames for other sections
    
    # Export to Excel
    with pd.ExcelWriter('codebase_overview.xlsx') as writer:
        ui_df.to_excel(writer, sheet_name='UI Setup', index=False)
        # Other DataFrames to Excel

if __name__ == "__main__":
    main()
    # Note: export_trades_to_excel() is now called within the trading process, not here