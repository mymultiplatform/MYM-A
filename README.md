# LSTM Trading Backtester

## Overview

This project implements an LSTM (Long Short-Term Memory) model for predicting cryptocurrency prices and performs a backtest using historical BTCUSD data from MetaTrader 5. The goal is to simulate trading with a starting balance and leverage while utilizing an LSTM model to make buy/sell decisions based on predicted prices.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Data Fetching](#data-fetching)
- [Model Training](#model-training)
- [Trading Logic](#trading-logic)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Historical Data Fetching:** Retrieves historical BTCUSD price data from MetaTrader 5.
- **LSTM Model Training:** Trains an LSTM model to predict future price movements.
- **Backtesting Framework:** Simulates trading based on predictions with features like stop-loss and profit targets.
- **Visualizations:** Displays real-time price predictions and trade executions through interactive plots.
- **User Interface:** A simple Tkinter GUI to initiate backtesting.

## Requirements

Ensure you have the following libraries installed:

- Python 3.6 or higher
- MetaTrader5
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- TensorFlow
- Tkinter (usually included with Python installations)

To install the required packages, you can use pip:

```bash
pip install MetaTrader5 pandas numpy matplotlib scikit-learn tensorflow
