import MetaTrader5 as mt5
from tkinter import messagebox
from datetime import datetime
from data_processing.data_functions import run_trading_process

def connect_to_mt5(login, password, server):
    if not mt5.initialize():
        messagebox.showerror("Error", "initialize() failed")
        mt5.shutdown()
        return False

    authorized = mt5.login(login=int(login), password=password, server=server)
    if authorized:
        print("Connected to MetaTrader 5")
        return True
    else:
        messagebox.showerror("Error", "Failed to connect to MetaTrader 5")
        return False

def get_mt5_timeframe(timeframe_str):
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
    return timeframe_map.get(timeframe_str.upper(), mt5.TIMEFRAME_D1)

def get_mt5_symbol_info(symbol):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to get symbol info for {symbol}")
        return None
    return symbol_info._asdict()

def place_market_order(symbol, volume, order_type, deviation=20):
    symbol_info = get_mt5_symbol_info(symbol)
    if symbol_info is None:
        return None

    point = symbol_info['point']
    price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "deviation": deviation,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
    }

    result = mt5.order_send(request)
    return result
