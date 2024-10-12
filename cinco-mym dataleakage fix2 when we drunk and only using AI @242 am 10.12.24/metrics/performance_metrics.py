import numpy as np

class PerformanceTracker:
    def __init__(self):
        self.performance = {"Backtest": {"total_profit": 0, "total_trades": 0, "winning_trades": 0, "losing_trades": 0, "trade_history": []},
                            "Live Simulation": {"total_profit": 0, "total_trades": 0, "winning_trades": 0, "losing_trades": 0, "trade_history": []}}

    def update(self, trade_type, entry_price, exit_price, entry_date, exit_date, phase):
        profit = exit_price - entry_price if trade_type == "BUY" else entry_price - exit_price
        self.performance[phase]["total_profit"] += profit
        self.performance[phase]["total_trades"] += 1
        if profit > 0:
            self.performance[phase]["winning_trades"] += 1
        else:
            self.performance[phase]["losing_trades"] += 1
        self.performance[phase]["trade_history"].append({
            "type": trade_type,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "profit": profit
        })

    def calculate_forward_looking_metrics(self, current_date, phase):
        relevant_trades = [trade for trade in self.performance[phase]["trade_history"] if trade["entry_date"] <= current_date]

        if not relevant_trades:
            return 0, 0, 0, 0

        total_profit = sum(trade["profit"] for trade in relevant_trades)
        win_rate = sum(1 for trade in relevant_trades if trade["profit"] > 0) / len(relevant_trades)
        avg_profit = total_profit / len(relevant_trades)

        cumulative_profits = np.cumsum([trade["profit"] for trade in relevant_trades])
        max_drawdown = np.max(np.maximum.accumulate(cumulative_profits) - cumulative_profits)

        return total_profit, win_rate, avg_profit, max_drawdown

    def generate_report(self, current_date, phase):
        total_profit, win_rate, avg_profit, max_drawdown = self.calculate_forward_looking_metrics(current_date, phase)

        print("\n" + "="*50)
        print(f"Performance Report for {phase} (as of {current_date}):")
        print(f"Total Profit/Loss: ${total_profit:.2f}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Profit per Trade: ${avg_profit:.2f}")
        print(f"Maximum Drawdown: ${max_drawdown:.2f}")
        print("="*50 + "\n")
