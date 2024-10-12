import MetaTrader5 as mt5
from tkinter import messagebox
import time
import numpy as np
from datetime import datetime, timedelta
import random
from metrics.performance_metrics import PerformanceTracker

# Initialize the performance tracker
performance_tracker = PerformanceTracker()

def place_trade(order_type, trade_date, price):
    symbol = "BTCUSD"
    lot_size = 0.1
    print(f"Simulated trade placed on {trade_date}: {order_type} {symbol} at {price}")
    
    # Simulate trade outcome
    exit_date = trade_date + timedelta(days=random.randint(1, 5))
    exit_price = price * (1 + random.uniform(-0.05, 0.05))
    profit = exit_price - price if order_type == "BUY" else price - exit_price
    
    print(f"Simulated trade closed on {exit_date}: Exit price {exit_price}, Profit: {profit}")
    
    # Update performance tracker
    performance_tracker.update(order_type, price, exit_price, trade_date, exit_date)

def generate_performance_report():
    performance_tracker.generate_report()

def simulate_trade_outcome(order_type, entry_price, entry_date):
    # Simulate a simple trade outcome
    exit_date = entry_date + timedelta(days=random.randint(1, 5))
    exit_price = entry_price * (1 + random.uniform(-0.05, 0.05))
    
    trade_type = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"
    
    if order_type == mt5.ORDER_TYPE_BUY:
        profit = exit_price - entry_price
    else:
        profit = entry_price - exit_price
    
    print(f"Simulated trade closed on {exit_date}: Exit price {exit_price}, Profit: {profit}")
    
    # Update performance tracker
    performance_tracker.update(trade_type, entry_price, exit_price, entry_date, exit_date)

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
    else:
        messagebox.showerror("Trade Close Error", f"Failed to close trade: {result.retcode}")
