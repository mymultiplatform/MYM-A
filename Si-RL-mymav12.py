import MetaTrader5 as mt5
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import time

# Initialize MetaTrader5 and login
login = 312128713
password = "Sexo247420@"
server = "XMGlobal-MT5 7"

if not mt5.initialize():
    raise Exception("MetaTrader5 initialization failed")

if mt5.login(login=login, password=password, server=server):
    print("Connected to MetaTrader 5 for data retrieval")
else:
    raise Exception("Failed to connect to MetaTrader 5")

# RL Neural Network model for decision making
class TradeRLModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TradeRLModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Environment for the RL agent interacting with MT5
class MT5Environment:
    def __init__(self):
        self.balance = mt5.account_info().balance
        self.initial_balance = self.balance
    
    def get_state(self):
        """Get current state information for the RL agent, including market data."""
        # Retrieve account info
        account_info = mt5.account_info()
        self.balance = account_info.balance
        equity = account_info.equity
        open_trades = len(mt5.positions_get())
        
        # Historical market data for EURUSD (last 100 candles on M1 timeframe)
        rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M1, 0, 100)
        close_prices = [rate['close'] for rate in rates]

        # Latest bid/ask prices for EURUSD
        tick = mt5.symbol_info_tick("EURUSD")
        bid, ask = tick.bid, tick.ask
        
        # State: [balance, number of trades, equity, last close price, mean close price, bid, ask]
        state = [
            self.balance,
            open_trades,
            equity,
            close_prices[-1],  # Most recent close price
            np.mean(close_prices),  # Average of last 100 close prices
            bid,
            ask
        ]
        return torch.tensor(state, dtype=torch.float32)
    
    def step(self, action):
        """Execute a step in the environment based on action (buy/sell/no action)."""
        if action == 0:
            self.buy()
        elif action == 1:
            self.sell()
        else:
            pass  # Neutral (no action)
        
        # Update balance after the action
        self.balance = mt5.account_info().balance
        reward = self.get_reward()
        next_state = self.get_state()
        
        return next_state, reward
    
    def buy(self):
        """Execute a buy order."""
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": "EURUSD",
            "volume": 0.1,
            "type": mt5.ORDER_BUY,
            "price": mt5.symbol_info_tick("EURUSD").ask,
            "deviation": 20,
            "magic": 234000,
            "comment": "RL Buy Order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(request)
    
    def sell(self):
        """Execute a sell order."""
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": "EURUSD",
            "volume": 0.1,
            "type": mt5.ORDER_SELL,
            "price": mt5.symbol_info_tick("EURUSD").bid,
            "deviation": 20,
            "magic": 234000,
            "comment": "RL Sell Order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(request)
    
    def get_reward(self):
        """Calculate reward based on balance change."""
        return self.balance - self.initial_balance
    
    def reset(self):
        """Reset the environment to its initial state."""
        self.balance = mt5.account_info().balance
        return self.get_state()

# Agent interacting with the environment
class TradeAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = TradeRLModel(state_size, 64, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()  # Best action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward
            with torch.no_grad():
                target += self.gamma * torch.max(self.model(next_state)).item()
            current_q = self.model(state)[action]
            loss = self.criterion(current_q, torch.tensor([target]))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent():
    env = MT5Environment()
    agent = TradeAgent(state_size=7, action_size=3)  # State: balance, trades, equity, prices, bid/ask
    episodes = 1000
    batch_size = 32

    for episode in range(episodes):
        state = env.reset()
        state = state.unsqueeze(0)  # Add batch dimension

        for t in range(200):  # Limit the steps per episode
            action = agent.act(state)
            next_state, reward = env.step(action)
            next_state = next_state.unsqueeze(0)
            agent.remember(state, action, reward, next_state)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        print(f"Episode {episode + 1}/{episodes} - Balance: {env.balance}, Epsilon: {agent.epsilon}")

if __name__ == "__main__":
    train_agent()
