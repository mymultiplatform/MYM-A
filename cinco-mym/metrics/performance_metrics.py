import numpy as np

class PerformanceTracker:
    def __init__(self):
        self.total_profit = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.trade_history = []

    def update(self, trade_type, entry_price, exit_price, entry_date, exit_date):
        profit = exit_price - entry_price if trade_type == "BUY" else entry_price - exit_price
        self.total_profit += profit
        self.total_trades += 1
        if profit > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        self.trade_history.append({
            "type": trade_type,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "profit": profit
        })

    def calculate_win_rate(self):
        return (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0

    def calculate_average_profit(self):
        return self.total_profit / self.total_trades if self.total_trades > 0 else 0

    def calculate_max_drawdown(self):
        cumulative_profits = np.cumsum([trade["profit"] for trade in self.trade_history])
        max_drawdown = 0
        peak = cumulative_profits[0]
        for profit in cumulative_profits:
            if profit > peak:
                peak = profit
            drawdown = peak - profit
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown

    def calculate_sharpe_ratio(self, risk_free_rate=0.02):
        profits = [trade["profit"] for trade in self.trade_history]
        returns = np.array(profits) / [trade["entry_price"] for trade in self.trade_history]
        excess_returns = returns - risk_free_rate / 252  # Assuming 252 trading days in a year
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns) if len(returns) > 0 else 0

    def generate_report(self):
        print("\n" + "="*50)
        print("Performance Report:")
        print(f"Total Profit/Loss: ${self.total_profit:.2f}")
        print(f"Total Trades: {self.total_trades}")
        print(f"Winning Trades: {self.winning_trades}")
        print(f"Losing Trades: {self.losing_trades}")
        print(f"Win Rate: {self.calculate_win_rate():.2f}%")
        print(f"Average Profit per Trade: ${self.calculate_average_profit():.2f}")
        print(f"Maximum Drawdown: ${self.calculate_max_drawdown():.2f}")
        print(f"Sharpe Ratio: {self.calculate_sharpe_ratio():.2f}")
        print("="*50 + "\n")
