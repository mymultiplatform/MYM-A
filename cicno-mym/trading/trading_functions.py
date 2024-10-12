import MetaTrader5 as mt5
from tkinter import messagebox
import time
import numpy as np

def place_trade(order_type):
    symbol = "BTCUSD"
    lot_size = 0.1
    price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "deviation": 10,
        "magic": 234000,
        "comment": "Automated trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print("Trade successfully placed")
        monitor_trade(result.order)
    else:
        messagebox.showerror("Trade Error", f"Failed to place trade: {result.retcode}")

def monitor_trade(order_id):
    while True:
        position = mt5.positions_get(ticket=order_id)
        if position:
            position = position[0]
            profit = position.profit
            if profit >= 0.01:
                close_trade(order_id, position.type)
                break
        time.sleep(5)

def close_trade(order_id, order_type):
    symbol = "BTCUSD"
    price = mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.1,
        "type": mt5.ORDER_TYPE_SELL if order_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "position": order_id,
        "price": price,
        "deviation": 10,
        "magic": 234000,
        "comment": "Automated trade close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print("Trade successfully closed")
        # Restart the dice rolling (you might need to adjust this based on your main application structure)
        # root.after(0, on_button_click)
    else:
        messagebox.showerror("Trade Close Error", f"Failed to close trade: {result.retcode}")
