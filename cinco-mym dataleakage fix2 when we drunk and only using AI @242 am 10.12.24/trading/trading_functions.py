import numpy as np
from datetime import datetime, timedelta
from metrics.performance_metrics import PerformanceTracker
import pandas as pd
from data_processing.data_functions import predict_future, determine_trend

# Initialize the performance tracker
performance_tracker = PerformanceTracker()

# Global dictionary to store trade data for different phases
trade_data = {"Backtest": []}

# Global variables for account balance and position
initial_balance = 1000
account_balance = initial_balance
current_position = 0

def place_trade(order_type, trade_date, price, phase):
    global trade_data, account_balance, current_position
    symbol = "BTCUSD"
    trade_amount = 100  # Fixed trade amount of $100
    
    execution_delay = timedelta(minutes=5)
    execution_time = trade_date + execution_delay
    
    price_change = np.random.normal(0, 0.001)
    execution_price = price * (1 + price_change)
    
    if order_type == "BUY" and account_balance >= trade_amount:
        btc_amount = trade_amount / execution_price
        account_balance -= trade_amount
        current_position += btc_amount
        print(f"[{phase}] Bought {btc_amount:.8f} BTC at ${execution_price:.2f}. Balance: ${account_balance:.2f}")
        performance_tracker.update("BUY", execution_price, btc_amount, execution_time, phase)
    elif order_type == "SELL" and current_position > 0:
        sell_amount = min(current_position, trade_amount / execution_price)
        account_balance += sell_amount * execution_price
        current_position -= sell_amount
        print(f"[{phase}] Sold {sell_amount:.8f} BTC at ${execution_price:.2f}. Balance: ${account_balance:.2f}")
        performance_tracker.update("SELL", execution_price, sell_amount, execution_time, phase)
    else:
        print(f"[{phase}] Insufficient balance or no position to trade. No action taken.")
        return None
    
    trade_result = {
        "Order Type": order_type,
        "Trade Date": execution_time,
        "Price": execution_price,
        "Amount": btc_amount if order_type == "BUY" else sell_amount,
        "Balance": account_balance,
        "Position": current_position
    }
    
    trade_data[phase].append(trade_result)
    
    return trade_result

def reset_account():
    global account_balance, current_position
    account_balance = initial_balance
    current_position = 0
    print(f"Account reset. Balance: ${account_balance:.2f}")

def export_trades_to_excel(filename):
    all_trades = []
    for phase, trades in trade_data.items():
        for trade in trades:
            trade['Phase'] = phase
            all_trades.append(trade)
    
    trade_df = pd.DataFrame(all_trades)
    
    with pd.ExcelWriter(filename) as writer:
        trade_df.to_excel(writer, sheet_name='Trade Details', index=False)
    
    print(f"Trade report exported to {filename}")

def generate_performance_report(end_date, phase):
    performance_tracker.generate_report(end_date, phase)

def backtest_strategy(all_data, model, scaler):
    train_end = datetime(2022, 1, 1)
    backtest_end = datetime(2024, 1, 1)
    backtest_data = all_data[(all_data['time'] >= train_end) & (all_data['time'] < backtest_end)]
    print("Starting Backtest phase...")
    backtest_results = run_phase(backtest_data, model, scaler, "Backtest")
    print("Backtest completed")

    return backtest_results

def run_phase(phase_data, model, scaler, phase):
    global account_balance, current_position
    phase_results = []
    
    for i in range(60, len(phase_data)):
        if account_balance <= 0 and current_position == 0:
            reset_account()
        
        historical_data = phase_data.iloc[:i]
        current_data = phase_data.iloc[i]

        predicted_prices = predict_future(model, historical_data['close'].values, scaler, 1)
        predicted_price = predicted_prices[0][0]

        trend = determine_trend(current_data['close'], predicted_price)

        if trend == "Bull" and account_balance >= 100:
            trade_result = place_trade("BUY", current_data['time'], current_data['close'], phase)
            if trade_result:
                phase_results.append(trade_result)
        elif trend == "Bear" and current_position > 0:
            trade_result = place_trade("SELL", current_data['time'], current_data['close'], phase)
            if trade_result:
                phase_results.append(trade_result)

        print(f"[{phase}] Date: {current_data['time']}, Actual: {current_data['close']:.2f}, Predicted: {predicted_price:.2f}, Trend: {trend}, Balance: ${account_balance:.2f}, Position: {current_position:.8f}")

    print(f"{phase} phase completed")
    return phase_results