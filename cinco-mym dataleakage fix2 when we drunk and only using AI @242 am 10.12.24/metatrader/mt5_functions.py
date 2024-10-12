import MetaTrader5 as mt5
from tkinter import messagebox
from datetime import datetime
from data_processing.data_functions import fetch_historical_data, train_model
from trading.trading_functions import backtest_strategy, generate_performance_report, export_trades_to_excel

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

def run_trading_process():
    symbol = "BTCUSD"
    timeframe = mt5.TIMEFRAME_D1
    start_date = datetime(2010, 1, 1)
    train_end_date = datetime(2022, 1, 1)
    backtest_end_date = datetime(2023, 1, 1)

    # Fetch all historical data
    all_data = fetch_historical_data(mt5, symbol, timeframe, start_date, backtest_end_date)
    if all_data is None or len(all_data) == 0:
        print("Failed to fetch historical data")
        return

    # Split data into training and backtesting periods
    train_data = all_data[all_data['time'] < train_end_date]
    backtest_data = all_data[(all_data['time'] >= train_end_date) & (all_data['time'] < backtest_end_date)]

    # Train model on 2010-2021 data
    model, scaler = train_model(train_data)  # Remove the second argument here
    if model is None or scaler is None:
        print("Failed to train model")
        return

    # Run backtesting strategy on 2022-2023 data
    print(f"Running backtesting from {backtest_data.iloc[0]['time']} to {backtest_data.iloc[-1]['time']}")
    backtest_results = backtest_strategy(backtest_data, model, scaler)

    # Generate and export reports
    generate_performance_report(backtest_data.iloc[-1]['time'], "Backtest")
    export_trades_to_excel("trades_report.xlsx")

# The rest of your functions remain unchanged
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