import numpy as np
from collections import defaultdict

class PerformanceTracker:
    def __init__(self):
        self.performance = defaultdict(lambda: {
            "total_profit": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "break_even_trades": 0,
            "open_positions": []
        })

    def update(self, order_type, price, amount, time, phase):
        if order_type == "BUY":
            self.performance[phase]["open_positions"].append((price, amount))
        elif order_type == "SELL":
            if self.performance[phase]["open_positions"]:
                buy_price, buy_amount = self.performance[phase]["open_positions"].pop(0)
                sell_amount = min(amount, buy_amount)
                profit = (price - buy_price) * sell_amount
                
                self.performance[phase]["total_profit"] += profit
                self.performance[phase]["total_trades"] += 1
                
                if profit > 0:
                    self.performance[phase]["winning_trades"] += 1
                elif profit < 0:
                    self.performance[phase]["losing_trades"] += 1
                else:
                    self.performance[phase]["break_even_trades"] += 1

    def generate_report(self, end_date, phase):
        if phase not in self.performance:
            print(f"No performance data available for phase: {phase}")
            return

        performance = self.performance[phase]
        total_trades = performance["total_trades"]
        
        if total_trades == 0:
            print(f"No trades were executed in the {phase} phase.")
            return

        win_rate = (performance["winning_trades"] / total_trades) * 100 if total_trades > 0 else 0
        loss_rate = (performance["losing_trades"] / total_trades) * 100 if total_trades > 0 else 0
        break_even_rate = (performance["break_even_trades"] / total_trades) * 100 if total_trades > 0 else 0

        print(f"\nPerformance Report for {phase} phase (as of {end_date}):")
        print(f"Total Profit: ${performance['total_profit']:.2f}")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {performance['winning_trades']} ({win_rate:.2f}%)")
        print(f"Losing Trades: {performance['losing_trades']} ({loss_rate:.2f}%)")
        print(f"Break-even Trades: {performance['break_even_trades']} ({break_even_rate:.2f}%)")
        print(f"Win Rate: {win_rate:.2f}%")

        if total_trades > 0:
            avg_profit_per_trade = performance['total_profit'] / total_trades
            print(f"Average Profit per Trade: ${avg_profit_per_trade:.2f}")
        
        print(f"Open Positions: {len(performance['open_positions'])}")
        print("\n")