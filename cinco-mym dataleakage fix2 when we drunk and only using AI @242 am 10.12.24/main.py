import tkinter as tk
from tkinter import messagebox
import MetaTrader5 as mt5
import threading
from dice.dice_functions import setup_dice_ui, auto_roll
from metatrader.mt5_functions import connect_to_mt5, start_automation
import numpy as np
import pandas as pd  # Ensure pandas is imported
from trading.trading_functions import export_trades_to_excel  # Import the function

def main():
    global root, display_var, click_button, message_label, connect_button
    root = tk.Tk()
    root.title("MYM-A MODO CHEZ")
    root.geometry("600x400")

    # Create frames
    dice_frame = tk.Frame(root, width=300, height=400)
    dice_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    login_frame = tk.Frame(root, width=300, height=400)
    login_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

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
                               command=lambda: connect_to_mt5(login_entry.get(), password_entry.get(), server_entry.get()))
    connect_button.pack(pady=20)

    # Setup Dice UI
    display_var, click_button, message_label = setup_dice_ui(dice_frame, root, connect_button)

    # Start the automatic dice rolling in a separate thread
    threading.Thread(target=lambda: auto_roll(root, click_button, display_var), daemon=True).start()

    root.mainloop()

def export_to_excel():
    # Create data for each section
    ui_data = {
        "Title": ["MYM-A MODO CHEZ"],
        "Geometry": ["600x400"],
        "Frames": ["Dice Frame", "Login Frame"],
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
    export_trades_to_excel()  # Call the function here to export trade data after running the main function
