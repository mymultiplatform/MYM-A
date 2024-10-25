import MetaTrader5 as mt5
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pandas_ta as ta
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from arch import arch_model  # Ensure arch is installed
import time
import os
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import messagebox
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from hyperopt import hp, tpe, fmin, Trials
from hyperopt.pyll.base import scope
from statsmodels.tsa.arima.model import ARIMA
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import threading
import random
import logging

# ============================================================
# Configuration and Constants
# ============================================================

# Configure logging
logging.basicConfig(level=logging.INFO, filename='trading_app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DEFAULT_SYMBOL = "EURUSD"
DEFAULT_VOLUME = 0.1
DEFAULT_DEVIATION = 20
DEFAULT_MAGIC_NUMBER = 234000  # Unique identifier for the EA
DEFAULT_SEQUENCE_LENGTH = 20
DEFAULT_STATE_SIZE = 13
DEFAULT_RISK_REWARD_RATIO = 1.5
DEFAULT_STOP_LOSS_PIPS = 20
DEFAULT_TAKE_PROFIT_PIPS = DEFAULT_STOP_LOSS_PIPS * DEFAULT_RISK_REWARD_RATIO
MAX_STEPS_PER_EPISODE = 100

# ============================================================
# Reinforcement Learning Environment and Model Definitions
# ============================================================

class MT5Environment(gym.Env):
    def __init__(self):
        super(MT5Environment, self).__init__()
        self.symbol = DEFAULT_SYMBOL
        self.volume = DEFAULT_VOLUME
        self.deviation = DEFAULT_DEVIATION
        self.magic = DEFAULT_MAGIC_NUMBER
        self.sequence_length = DEFAULT_SEQUENCE_LENGTH
        self.state_size = DEFAULT_STATE_SIZE
        self.position_opened = False

        # Risk management parameters
        self.risk_reward_ratio = DEFAULT_RISK_REWARD_RATIO
        self.stop_loss_pips = DEFAULT_STOP_LOSS_PIPS
        self.take_profit_pips = DEFAULT_TAKE_PROFIT_PIPS

        # Define action and observation spaces
        self.action_space = spaces.Discrete(5)  # Actions: Buy, Sell, Hold, Close Buy, Close Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.sequence_length, self.state_size),
                                            dtype=np.float32)
        self.reset()

    def get_state(self):
        """Get current state information for the RL agent, including market data and technical indicators."""
        try:
            # Retrieve account info
            account_info = mt5.account_info()
            if account_info is None:
                raise ValueError("Failed to get account info.")

            balance = account_info.balance
            equity = account_info.equity
            open_trades = len(mt5.positions_get())

            # Historical market data for the symbol
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, 200)
            if rates is None or len(rates) == 0:
                raise ValueError("No rates retrieved.")

            df = pd.DataFrame(rates)

            # Calculate returns for GARCH model
            df['returns'] = df['close'].pct_change()

            # GARCH(1,1) volatility estimation
            if len(df['returns'].dropna()) >= 30:
                # Need at least 30 data points to fit GARCH model
                am = arch_model(df['returns'].dropna(), vol='Garch', p=1, q=1)
                res = am.fit(disp='off')
                # Forecast volatility for the next period
                forecast = res.forecast(horizon=1)
                garch_volatility = np.sqrt(forecast.variance.values[-1, :][0])
            else:
                # If not enough data, set volatility to 0
                garch_volatility = 0.0

            # Calculate technical indicators
            df['RSI'] = ta.rsi(df['close'], length=14)
            macd = ta.macd(df['close'])
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_signal'] = macd['MACDs_12_26_9']
            df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['EMA'] = ta.ema(df['close'], length=14)
            bbands = ta.bbands(df['close'], length=20)
            df['BB_upper'] = bbands['BBU_20_2.0']
            df['BB_middle'] = bbands['BBM_20_2.0']
            df['BB_lower'] = bbands['BBL_20_2.0']

            # Fill NaN values
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)

            # Get the latest values
            latest_data = df.iloc[-1]

            # State: [balance, equity, number of trades, close price, RSI, MACD, MACD_signal,
            #         ATR, EMA, BB_upper, BB_lower, returns, GARCH_volatility]
            returns = df['returns'].iloc[-1]
            state = [
                balance,
                equity,
                open_trades,
                latest_data['close'],
                latest_data['RSI'],
                latest_data['MACD'],
                latest_data['MACD_signal'],
                latest_data['ATR'],
                latest_data['EMA'],
                latest_data['BB_upper'],
                latest_data['BB_lower'],
                returns,
                garch_volatility
            ]

            # Update state sequence
            if not hasattr(self, 'state_sequence'):
                self.state_sequence = []
            self.state_sequence.append(state)
            if len(self.state_sequence) > self.sequence_length:
                self.state_sequence.pop(0)

            # Ensure the sequence has the required length
            if len(self.state_sequence) < self.sequence_length:
                padding = [self.state_sequence[0]] * (self.sequence_length - len(self.state_sequence))
                self.state_sequence = padding + self.state_sequence

            return np.array(self.state_sequence, dtype=np.float32)
        except Exception as e:
            logging.error(f"Error in get_state: {e}")
            # Return the last known state or zeros
            if hasattr(self, 'state_sequence') and len(self.state_sequence) > 0:
                return np.array(self.state_sequence, dtype=np.float32)
            else:
                return np.zeros((self.sequence_length, self.state_size), dtype=np.float32)

    def step(self, action):
        """Execute a step in the environment based on action."""
        # Actions: 0 = Buy, 1 = Sell, 2 = Hold, 3 = Close Buy, 4 = Close Sell
        # Execute the action
        if action == 0:
            self.buy()
        elif action == 1:
            self.sell()
        elif action == 3:
            self.close_positions(order_type=mt5.ORDER_TYPE_BUY)
        elif action == 4:
            self.close_positions(order_type=mt5.ORDER_TYPE_SELL)
        # For action == 2 (Hold), do nothing

        # Calculate reward based on positions' profit
        reward = self.get_reward()
        next_state = self.get_state()

        # Determine if episode is done
        self.current_step += 1
        done = self.current_step >= MAX_STEPS_PER_EPISODE

        info = {}

        return next_state, reward, done, info

    def buy(self):
        """Execute a buy order with stop-loss and take-profit."""
        # Check if a buy position is already open
        positions = mt5.positions_get(symbol=self.symbol, type=mt5.POSITION_TYPE_BUY)
        if positions:
            logging.info("Buy position already open.")
            return

        # Get current price
        symbol_info = mt5.symbol_info_tick(self.symbol)
        if symbol_info is None:
            logging.error(f"Failed to get symbol info for {self.symbol}")
            return

        price = symbol_info.ask
        point = mt5.symbol_info(self.symbol).point

        # Calculate stop-loss and take-profit prices
        sl = price - self.stop_loss_pips * point
        tp = price + self.take_profit_pips * point

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.volume,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": "RL Buy Order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Buy order failed, retcode={result.retcode}")
        else:
            self.position_opened = True
            logging.info("Buy order placed successfully.")

    def sell(self):
        """Execute a sell order with stop-loss and take-profit."""
        # Check if a sell position is already open
        positions = mt5.positions_get(symbol=self.symbol, type=mt5.POSITION_TYPE_SELL)
        if positions:
            logging.info("Sell position already open.")
            return

        # Get current price
        symbol_info = mt5.symbol_info_tick(self.symbol)
        if symbol_info is None:
            logging.error(f"Failed to get symbol info for {self.symbol}")
            return

        price = symbol_info.bid
        point = mt5.symbol_info(self.symbol).point

        # Calculate stop-loss and take-profit prices
        sl = price + self.stop_loss_pips * point
        tp = price - self.take_profit_pips * point

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.volume,
            "type": mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": "RL Sell Order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Sell order failed, retcode={result.retcode}")
        else:
            self.position_opened = True
            logging.info("Sell order placed successfully.")

    def close_positions(self, order_type=None):
        """Close open positions."""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return
        for position in positions:
            if order_type is None or position.type == order_type:
                close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": position.ticket,
                    "symbol": position.symbol,
                    "volume": position.volume,
                    "type": close_type,
                    "price": price,
                    "deviation": self.deviation,
                    "magic": self.magic,
                    "comment": "RL Close Order",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logging.error(f"Close position failed, retcode={result.retcode}")
                else:
                    logging.info("Position closed successfully.")
                    self.position_opened = False

    def get_reward(self):
        """Calculate reward based on positions' profit."""
        account_info = mt5.account_info()
        if account_info is None:
            logging.error("Failed to get account info for reward calculation.")
            return 0.0
        equity = account_info.equity
        if not hasattr(self, 'last_equity'):
            self.last_equity = equity
        reward = equity - self.last_equity
        self.last_equity = equity
        return reward

    def reset(self):
        """Reset the environment to its initial state."""
        self.state_sequence = []
        self.last_equity = mt5.account_info().equity if mt5.account_info() else 0.0
        self.current_step = 0
        # Close any open positions
        self.close_positions()
        # Initialize state sequence
        state = self.get_state()
        return state

# Custom LSTM Features Extractor
class CustomLSTMFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, lstm_hidden_size=128, lstm_num_layers=2, dropout=0.2):
        super(CustomLSTMFeaturesExtractor, self).__init__(observation_space, features_dim=lstm_hidden_size)
        self.sequence_length, self.input_size = observation_space.shape
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, observations):
        h_0 = torch.zeros(self.lstm.num_layers, observations.size(0), self.lstm.hidden_size).to(observations.device)
        c_0 = torch.zeros(self.lstm.num_layers, observations.size(0), self.lstm.hidden_size).to(observations.device)
        out, _ = self.lstm(observations, (h_0, c_0))
        out = self.dropout(out[:, -1, :])
        return out

# Custom LSTM Policy
class CustomLSTMPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomLSTMPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomLSTMFeaturesExtractor,
            features_extractor_kwargs=dict(
                lstm_hidden_size=128,
                lstm_num_layers=2,
                dropout=0.2
            ),
        )

# ============================================================
# Reinforcement Learning Model Management
# ============================================================

class RLAgent:
    def __init__(self, model_dir="models"):
        self.env = DummyVecEnv([lambda: MT5Environment()])
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_file = os.path.join(self.model_dir, "LSTM_PPO_Model.zip")
        self.model = self.load_or_create_model()

    def load_or_create_model(self):
        if os.path.exists(self.model_file):
            logging.info(f"Loading existing model from {self.model_file}")
            model = PPO.load(self.model_file, env=self.env)
        else:
            logging.info("Model not found, creating a new one.")
            model = PPO(
                CustomLSTMPolicy,
                self.env,
                verbose=1,
                tensorboard_log="./ppo_trading_tensorboard/",
                learning_rate=1e-5,
                batch_size=256
            )
        return model

    def evaluate_agent(self, episodes=5):
        total_rewards = []
        for episode in range(episodes):
            obs = self.env.reset()
            done = False
            cumulative_reward = 0
            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, done, info = self.env.step(action)
                cumulative_reward += reward
            total_rewards.append(cumulative_reward)
        avg_reward = np.mean(total_rewards)
        logging.info(f'Average Reward over {episodes} episodes: {avg_reward}')

    def train(self, total_timesteps=int(1e6), max_iterations=10, callback=None):
        try:
            for iteration in range(max_iterations):
                logging.info(f"Starting training iteration {iteration + 1}/{max_iterations}")
                self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, callback=callback)
                # Save the agent periodically
                self.model.save(self.model_file)
                logging.info(f"Model saved to {self.model_file}")
                # Evaluate the agent
                self.evaluate_agent()
        except KeyboardInterrupt:
            # Allow graceful exit
            logging.info("Training interrupted by user.")

# ============================================================
# Transformer Model Definition (Outside the Class)
# ============================================================

class TransformerModel(nn.Module):
    def __init__(self, feature_size=1, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=feature_size, nhead=1, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(feature_size, 1)

    def forward(self, src, tgt):
        output = self.transformer(src, tgt)
        output = self.linear(output)
        return output.squeeze(-1)

# ============================================================
# Tkinter GUI and Automated Trading Management
# ============================================================

class TradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MYM-A MODO CHEZ")
        self.root.geometry("600x400")  # Adjusted size for better layout

        # Initialize RL Agent
        self.rl_agent = RLAgent()

        # Event to signal threads to stop
        self.stop_event = threading.Event()

        # Create frames
        self.login_frame = tk.Frame(root, width=600, height=400)
        self.login_frame.pack(fill=tk.BOTH, expand=True)

        self.create_login_ui()

        # Handle window closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_login_ui(self):
        """Create the Login UI components."""
        login_title = tk.Label(self.login_frame, text="🏧MYM-A", font=("Helvetica", 20))
        login_title.pack(pady=20)

        # Login
        login_label = tk.Label(self.login_frame, text="Login:", font=("Helvetica", 14))
        login_label.pack(pady=5)
        self.login_entry = tk.Entry(self.login_frame, font=("Helvetica", 14))
        self.login_entry.pack(pady=5)

        # Password
        password_label = tk.Label(self.login_frame, text="Password:", font=("Helvetica", 14))
        password_label.pack(pady=5)
        self.password_entry = tk.Entry(self.login_frame, show="*", font=("Helvetica", 14))
        self.password_entry.pack(pady=5)

        # Server
        server_label = tk.Label(self.login_frame, text="Server:", font=("Helvetica", 14))
        server_label.pack(pady=5)
        self.server_entry = tk.Entry(self.login_frame, font=("Helvetica", 14))
        self.server_entry.pack(pady=5)

        # Connect Button
        self.connect_button = tk.Button(self.login_frame, text="Connect", font=("Helvetica", 14),
                                        command=self.connect_to_mt5)
        self.connect_button.pack(pady=20)

    def connect_to_mt5(self):
        """Handle MT5 connection using provided credentials."""
        login = self.login_entry.get()
        password = self.password_entry.get()
        server = self.server_entry.get()

        # Initialize MetaTrader5 and login
        if not mt5.initialize():
            messagebox.showerror("Error", "MetaTrader5 initialization failed")
            mt5.shutdown()
            return

        try:
            authorized = mt5.login(login=int(login), password=password, server=server)
        except Exception as e:
            messagebox.showerror("Error", f"Login failed: {e}")
            mt5.shutdown()
            return

        if authorized:
            messagebox.showinfo("Success", "Connected to MetaTrader 5")
            logging.info("Connected to MetaTrader 5")
            # Start RL training in a separate thread
            threading.Thread(target=self.start_rl_training, daemon=True).start()
            # Start automated trading in a separate thread
            threading.Thread(target=self.start_automation, daemon=True).start()
        else:
            messagebox.showerror("Error", "Failed to connect to MetaTrader 5")
            mt5.shutdown()

    def start_rl_training(self):
        """Start the RL agent's training loop."""
        logging.info("Starting RL Training")
        # Ensure thread safety
        with threading.Lock():
            self.rl_agent.train()

    def start_automation(self):
        """Start the automated trading loop."""
        logging.info("Starting Automated Trading")
        try:
            symbol = DEFAULT_SYMBOL  # Use consistent symbol
            timeframe = mt5.TIMEFRAME_D1
            days = 600
            data = self.fetch_historical_data(symbol, timeframe, days)
            scaled_data, scaler = self.preprocess_data(data)

            # Train models
            lstm_model = self.train_lstm_model(scaled_data)
            gb_model = self.train_gb_model(scaled_data)
            hyperopt_model = self.train_hyperopt_model(scaled_data)
            arima_model = self.train_arima_model(data['close'])
            svr_model = self.train_svr_model(scaled_data)
            rf_model = self.train_rf_model(scaled_data)
            mlp_model = self.train_mlp_model(scaled_data)
            transformer_model = self.train_transformer_model(scaled_data)
            gru_model = self.train_gru_model(scaled_data)

            while not self.stop_event.is_set():
                try:
                    # Fetch latest data
                    data = self.fetch_historical_data(symbol, timeframe, days)
                    scaled_data, scaler = self.preprocess_data(data)

                    # Predict future prices
                    future_days = 60
                    lstm_predictions = self.predict_future_lstm(lstm_model, scaled_data, future_days)
                    gb_predictions = self.predict_future_gb(gb_model, scaled_data, future_days)
                    hyperopt_predictions = self.predict_future_hyperopt(hyperopt_model, scaled_data, future_days)
                    arima_predictions = self.predict_future_arima(arima_model, len(data), future_days)
                    svr_predictions = self.predict_future_svr(svr_model, scaled_data, future_days)
                    rf_predictions = self.predict_future_rf(rf_model, scaled_data, future_days)
                    mlp_predictions = self.predict_future_mlp(mlp_model, scaled_data, future_days)
                    transformer_predictions = self.predict_future_transformer(transformer_model, scaled_data, future_days)
                    gru_predictions = self.predict_future_gru(gru_model, scaled_data, future_days)

                    # Combine predictions with weighted average
                    combined_predictions = (
                        0.2 * np.array(lstm_predictions) +
                        0.1 * np.array(gb_predictions) +
                        0.15 * np.array(hyperopt_predictions) +
                        0.1 * np.array(arima_predictions) +
                        0.05 * np.array(svr_predictions) +
                        0.05 * np.array(rf_predictions) +
                        0.05 * np.array(mlp_predictions) +
                        0.2 * np.array(transformer_predictions) +
                        0.1 * np.array(gru_predictions)
                    )

                    combined_predictions = scaler.inverse_transform(combined_predictions.reshape(-1, 1))

                    # Determine trend
                    trend = self.determine_trend(combined_predictions)
                    logging.info(f"Determined trend: {trend}")
                    if trend == "Bull":
                        self.place_trade(mt5.ORDER_TYPE_BUY)
                    elif trend == "Bear":
                        self.place_trade(mt5.ORDER_TYPE_SELL)

                    # Wait for the next prediction cycle
                    for _ in range(3600):  # Run the prediction and trading every hour
                        if self.stop_event.is_set():
                            break
                        time.sleep(1)
                except Exception as e:
                    logging.error(f"Error in automation loop: {e}")
                    time.sleep(60)  # Wait a minute before retrying in case of error
        except Exception as e:
            logging.error(f"Error in automation initialization: {e}")

    def fetch_historical_data(self, symbol, timeframe, days):
        """Fetch historical data for a given symbol and timeframe."""
        utc_to = datetime.utcnow()
        utc_from = utc_to - timedelta(days=days)
        rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
        if rates is None or len(rates) == 0:
            raise ValueError("No historical data retrieved.")
        data = pd.DataFrame(rates)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        data.set_index('time', inplace=True)
        return data

    def preprocess_data(self, data):
        """Preprocess data by scaling."""
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['close']])
        return scaled_data, scaler

    def create_sequences(self, data, seq_length):
        """Create sequences of data for time series models."""
        X = []
        y = []
        for i in range(len(data) - seq_length - 1):
            X.append(data[i:(i + seq_length), 0])
            y.append(data[i + seq_length, 0])
        X = np.array(X)
        y = np.array(y)
        return X, y

    def train_lstm_model(self, scaled_data):
        """Train an LSTM model."""
        seq_length = 60
        X, y = self.create_sequences(scaled_data, seq_length)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(64))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)
        return model

    def predict_future_lstm(self, model, scaled_data, future_days):
        """Predict future prices using LSTM model."""
        seq_length = 60
        last_sequence = scaled_data[-seq_length:]
        predictions = []
        for _ in range(future_days):
            X = np.reshape(last_sequence, (1, seq_length, 1))
            pred = model.predict(X)
            predictions.append(pred[0][0])
            last_sequence = np.append(last_sequence[1:], pred[0][0])
        return predictions

    def train_gru_model(self, scaled_data):
        """Train a GRU model."""
        seq_length = 60
        X, y = self.create_sequences(scaled_data, seq_length)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = Sequential()
        model.add(GRU(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(GRU(64))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)
        return model

    def predict_future_gru(self, model, scaled_data, future_days):
        """Predict future prices using GRU model."""
        seq_length = 60
        last_sequence = scaled_data[-seq_length:]
        predictions = []
        for _ in range(future_days):
            X = np.reshape(last_sequence, (1, seq_length, 1))
            pred = model.predict(X)
            predictions.append(pred[0][0])
            last_sequence = np.append(last_sequence[1:], pred[0][0])
        return predictions

    def train_gb_model(self, scaled_data):
        """Train a Gradient Boosting model."""
        seq_length = 60
        X, y = self.create_sequences(scaled_data, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = GradientBoostingRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        return model

    def predict_future_gb(self, model, scaled_data, future_days):
        """Predict future prices using Gradient Boosting model."""
        seq_length = 60
        last_sequence = scaled_data[-(seq_length + future_days - 1):]
        predictions = []
        for i in range(future_days):
            X = last_sequence[i:i + seq_length].reshape(1, -1)
            pred = model.predict(X)
            predictions.append(pred[0])
        return predictions

    def train_rf_model(self, scaled_data):
        """Train a Random Forest model."""
        seq_length = 60
        X, y = self.create_sequences(scaled_data, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        return model

    def predict_future_rf(self, model, scaled_data, future_days):
        """Predict future prices using Random Forest model."""
        seq_length = 60
        last_sequence = scaled_data[-(seq_length + future_days - 1):]
        predictions = []
        for i in range(future_days):
            X = last_sequence[i:i + seq_length].reshape(1, -1)
            pred = model.predict(X)
            predictions.append(pred[0])
        return predictions

    def train_svr_model(self, scaled_data):
        """Train an SVR model."""
        seq_length = 60
        X, y = self.create_sequences(scaled_data, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Reshape X_train and X_test
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        model = SVR()
        model.fit(X_train, y_train)
        return model

    def predict_future_svr(self, model, scaled_data, future_days):
        """Predict future prices using SVR model."""
        seq_length = 60
        last_sequence = scaled_data[-(seq_length + future_days - 1):]
        predictions = []
        for i in range(future_days):
            X = last_sequence[i:i + seq_length].reshape(1, -1)
            pred = model.predict(X)
            predictions.append(pred[0])
        return predictions

    def train_mlp_model(self, scaled_data):
        """Train an MLP model."""
        seq_length = 60
        X, y = self.create_sequences(scaled_data, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500)
        model.fit(X_train, y_train)
        return model

    def predict_future_mlp(self, model, scaled_data, future_days):
        """Predict future prices using MLP model."""
        seq_length = 60
        last_sequence = scaled_data[-(seq_length + future_days - 1):]
        predictions = []
        for i in range(future_days):
            X = last_sequence[i:i + seq_length].reshape(1, -1)
            pred = model.predict(X)
            predictions.append(pred[0])
        return predictions

    def train_hyperopt_model(self, scaled_data):
        """Train a model using hyperopt for hyperparameter optimization."""
        seq_length = 60
        X, y = self.create_sequences(scaled_data, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        def objective(params):
            model = GradientBoostingRegressor(**params)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mse = mean_squared_error(y_test, pred)
            return mse

        space = {
            'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 10)),
            'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
            'learning_rate': hp.loguniform('learning_rate', -3, 0)
        }

        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
        best_params = {
            'n_estimators': int(best['n_estimators']),
            'max_depth': int(best['max_depth']),
            'learning_rate': best['learning_rate']
        }
        model = GradientBoostingRegressor(**best_params)
        model.fit(X_train, y_train)
        return model

    def predict_future_hyperopt(self, model, scaled_data, future_days):
        """Predict future prices using the hyperopt model."""
        seq_length = 60
        last_sequence = scaled_data[-(seq_length + future_days - 1):]
        predictions = []
        for i in range(future_days):
            X = last_sequence[i:i + seq_length].reshape(1, -1)
            pred = model.predict(X)
            predictions.append(pred[0])
        return predictions

    def train_arima_model(self, data_close):
        """Train an ARIMA model."""
        model = ARIMA(data_close, order=(5, 1, 0))
        model_fit = model.fit()
        return model_fit

    def predict_future_arima(self, model_fit, start_index, future_days):
        """Predict future prices using ARIMA model."""
        forecast = model_fit.forecast(steps=future_days)
        return forecast.values

    def train_transformer_model(self, scaled_data):
        """Train a Transformer model."""
        seq_length = 60
        X, y = self.create_sequences(scaled_data, seq_length)
        X = X.reshape(X.shape[0], seq_length, 1)
        y = y.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Convert to tensors
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()

        model = TransformerModel(feature_size=1, num_layers=2, dropout=0.1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        epochs = 10

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train.transpose(0, 1), X_train.transpose(0, 1))
            loss = criterion(output[-1], y_train.squeeze())
            loss.backward()
            optimizer.step()

        return model

    def predict_future_transformer(self, model, scaled_data, future_days):
        """Predict future prices using Transformer model."""
        seq_length = 60
        last_sequence = scaled_data[-seq_length:]
        predictions = []
        for _ in range(future_days):
            X = torch.from_numpy(last_sequence).float().reshape(seq_length, 1, 1)
            model.eval()
            with torch.no_grad():
                output = model(X, X)
                pred = output[-1].item()
            predictions.append(pred)
            last_sequence = np.append(last_sequence[1:], pred)
        return predictions

    def determine_trend(self, predictions):
        """Determine market trend based on predictions."""
        # Calculate the slope of the predictions
        x = np.arange(len(predictions))
        y = predictions.flatten()
        slope, _ = np.polyfit(x, y, 1)
        threshold = 0.0001  # Adjust the threshold based on your data

        if slope > threshold:
            return "Bull"
        elif slope < -threshold:
            return "Bear"
        else:
            return "Neutral"

    def place_trade(self, order_type):
        """Place a trade based on the order type with risk management."""
        symbol = self.rl_agent.env.envs[0].symbol  # Use the symbol from the environment
        volume = DEFAULT_VOLUME
        deviation = DEFAULT_DEVIATION
        magic = DEFAULT_MAGIC_NUMBER

        symbol_info = mt5.symbol_info_tick(symbol)
        if symbol_info is None:
            logging.error(f"Failed to get symbol info for {symbol}")
            return

        point = mt5.symbol_info(symbol).point
        price = symbol_info.ask if order_type == mt5.ORDER_TYPE_BUY else symbol_info.bid

        # Calculate stop-loss and take-profit prices
        stop_loss_pips = DEFAULT_STOP_LOSS_PIPS
        take_profit_pips = stop_loss_pips * DEFAULT_RISK_REWARD_RATIO

        if order_type == mt5.ORDER_TYPE_BUY:
            sl = price - stop_loss_pips * point
            tp = price + take_profit_pips * point
        else:
            sl = price + stop_loss_pips * point
            tp = price - take_profit_pips * point

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": magic,
            "comment": "Automated trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info("Trade successfully placed")
        else:
            messagebox.showerror("Trade Error", f"Failed to place trade: {result.retcode}")
            logging.error(f"Failed to place trade: {result.retcode}")

    def close_trade(self, order_id, order_type):
        """Close an open trade."""
        symbol = self.rl_agent.env.envs[0].symbol  # Use the symbol from the environment
        price = mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": DEFAULT_VOLUME,
            "type": mt5.ORDER_TYPE_SELL if order_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": order_id,
            "price": price,
            "deviation": DEFAULT_DEVIATION,
            "magic": DEFAULT_MAGIC_NUMBER,
            "comment": "Automated trade close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info("Trade successfully closed")
        else:
            messagebox.showerror("Trade Close Error", f"Failed to close trade: {result.retcode}")
            logging.error(f"Failed to close trade: {result.retcode}")

    def on_closing(self):
        """Handle application closing event."""
        self.stop_event.set()
        mt5.shutdown()
        self.root.destroy()
        logging.info("Application closed.")

# ============================================================
# Main Function to Start the Application
# ============================================================

def main():
    root = tk.Tk()
    app = TradingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
