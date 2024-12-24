import MetaTrader5 as mt5
import tkinter as tk

def main():
    # Initialize and connect to MT5 to fetch historical data
    if not mt5.initialize():
        raise Exception("MetaTrader5 initialization failed")

    login = 312128713
    password = "Sexo247420@"
    server = "XMGlobal-MT5 7"

    if mt5.login(login=login, password=password, server=server):
        print("Connected to MetaTrader 5 for data retrieval")
    else:
        raise Exception("Failed to connect to MetaTrader 5")

def show_ui():
    root = tk.Tk()
    root.title("MetaTrader 5 Login")
    root.geometry("300x200")

    login_label = tk.Label(root, text="Login", font=("Helvetica", 12))
    login_label.pack(pady=5)
    login_entry = tk.Entry(root, font=("Helvetica", 12))
    login_entry.pack(pady=5)

    password_label = tk.Label(root, text="Password", font=("Helvetica", 12))
    password_label.pack(pady=5)
    password_entry = tk.Entry(root, font=("Helvetica", 12), show="*")
    password_entry.pack(pady=5)

    server_label = tk.Label(root, text="Server", font=("Helvetica", 12))
    server_label.pack(pady=5)
    server_entry = tk.Entry(root, font=("Helvetica", 12))
    server_entry.pack(pady=5)

    def on_connect():
        login = login_entry.get()
        password = password_entry.get()
        server = server_entry.get()

        # Initialize MT5 and login with user credentials
        if not mt5.initialize():
            print("MetaTrader5 initialization failed")
            return

        if mt5.login(login=int(login), password=password, server=server):
            print("Connected to MetaTrader 5 with credentials")
        else:
            print("Failed to connect to MetaTrader 5")

    connect_button = tk.Button(root, text="Connect", command=on_connect, font=("Helvetica", 14))
    connect_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    show_ui()
