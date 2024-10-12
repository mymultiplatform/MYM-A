import numpy as np
from datetime import datetime, timedelta
from metrics.performance_metrics import PerformanceTracker
import pandas as pd

# Initialize the performance tracker
performance_tracker = PerformanceTracker()

# Global dictionary to store trade data for different phases
trade_data = {"Backtest": [], "Live Simulation": []}

def place_trade(order_type, trade_date, price, phase):
    global trade_data
    symbol = "BTCUSD"
    lot_size = 0.1
    
    # Simulate a delay between prediction and execution
    execution_delay = timedelta(minutes=5)
    execution_time = trade_date + execution_delay
    
    # Simulate price change during the delay
    price_change = np.random.normal(0, 0.001)  # Small random price change
    execution_price = price * (1 + price_change)
    
    print(f"[{phase}] Simulated trade placed on {execution_time}: {order_type} {symbol} at {execution_price}")
    
    # Simulate trade outcome
    # Simulate trade outcome
    holding_period = timedelta(days=1) # Assume we hold the trade for 1 day
    exit_date = execution_time + holding_period
    exit_price = execution_price * (1 + np.random.normal(0, 0.02)) # Simulate price change with some randomness

    profit = exit_price - execution_price if order_type == "BUY" else execution_price - exit_price
    print(f"[{phase}] Simulated trade closed on {exit_date}: Exit price {exit_price}, Profit: {profit}")
    
    # Update performance tracker
    performance_tracker.update(order_type, execution_price, exit_price, execution_time, exit_date, phase)
    
    # Append trade details to the trade_data list
    trade_data[phase].append({
        "Order Type": order_type,
        "Trade Date": execution_time,
        "Entry Price": execution_price,
        "Exit Date": exit_date,
        "Exit Price": exit_price,
        "Profit": profit
    })

def export_trades_to_excel(filename):
    # Convert trade data to DataFrame
    all_trades = []
    for phase, trades in trade_data.items():
        for trade in trades:
            trade['Phase'] = phase
            all_trades.append(trade)
    
    trade_df = pd.DataFrame(all_trades)
    
    # Export DataFrame to Excel
    with pd.ExcelWriter(filename) as writer:
        trade_df.to_excel(writer, sheet_name='Trade Details', index=False)
    
    print(f"Trade report exported to {filename}")

def generate_performance_report(end_date, phase):
    performance_tracker.generate_report(end_date, phase)

def simulate_market_conditions(base_price, volatility=0.02, trend=0):
    """Simulate market conditions for more realistic backtesting."""
    return base_price * (1 + np.random.normal(trend, volatility))

def backtest_strategy(backtest_data, model, scaler, phase):
    print(f"Starting {phase}...")
    for i in range(60, len(backtest_data)):
        historical_data = backtest_data.iloc[:i]
        current_data = backtest_data.iloc[i]

        scaled_historical_data = scaler.transform(historical_data[['close']].values.reshape(-1, 1))
        predicted_prices = predict_future(model, scaled_historical_data[-60:], scaler, 1)
        predicted_price = predicted_prices[0][0]

        trend = determine_trend([current_data['close'], predicted_price])

        if trend == "Bull":
            place_trade("BUY", current_data['time'], current_data['close'], phase)
        elif trend == "Bear":
            place_trade("SELL", current_data['time'], current_data['close'], phase)

        print(f"[{phase}] Date: {current_data['time']}, Actual: {current_data['close']:.2f}, Predicted: {predicted_price:.2f}, Trend: {trend}")

    print(f"{phase} completed")



# You'll need to import these functions if they're not in this file
from data_processing.data_functions import predict_future, determine_trend
