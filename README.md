# **Algotrader: LSTM-Based Algorithmic Trading Backtest**

## **Table of Contents**
1. [Project Description](#project-description)
2. [Features](#features)
3. [Setup and Installation](#setup-and-installation)
4. [File Structure](#file-structure)
5. [Usage](#usage)
6. [Trading Logic](#trading-logic)
7. [Model Training](#model-training)
8. [Results and Visualization](#results-and-visualization)
9. [Limitations](#limitations)
10. [Future Improvements](#future-improvements)
11. [License](#license)

---

## **Project Description**

The **Algotrader** project provides a backtesting engine for a cryptocurrency trading algorithm using **Long Short-Term Memory (LSTM)** models. It connects to **MetaTrader 5** (MT5) to fetch historical data and uses deep learning models to predict future prices. The backtest is run using predefined trading rules, allowing users to simulate trades and analyze results based on historical BTC/USD price data.

---

## **Features**

- **MetaTrader 5 Integration**: Automatically connects to MT5 to fetch historical market data.
- **LSTM Model**: Utilizes LSTM models to predict cryptocurrency prices based on historical data.
- **Backtesting Engine**: Implements logic to simulate trades, evaluate performance, and log results.
- **Graphical User Interface (GUI)**: A simple UI built with **Tkinter** for starting the backtest.
- **Trade Simulation**: Automatically simulates 8 trades using prediction results.
- **Visualization**: Plots the trade outcomes with entry and exit points on a price chart.

---

## **Setup and Installation**

### **Prerequisites**
- **MetaTrader 5** installed on your machine.
- **Python 3.x** installed.
- Required Python libraries:
  - `MetaTrader5`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `tensorflow`
  - `scikit-learn`
  - `tkinter`

### **Installation Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/algotrader.git
   ```
2. Navigate to the project directory:
   ```bash
   cd algotrader
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your **MT5** credentials:
   - **Login**: Your MT5 account login
   - **Password**: Your MT5 account password
   - **Server**: MT5 broker server (e.g., "XMGlobal-MT5 7")

---

## **File Structure**

The main files are:
1. `backtestchartmym-av2.py`: Contains the LSTM model, backtest logic, and MT5 connection setup.
2. `BTESTV3.2.py`, `BTESTV4.py`, etc.: Various versions of the backtest scripts with different configurations.
3. `mym-a coliseo.py`: Contains the GUI and enhanced models for future prediction.
4. `README.md`: This document.

---

## **Usage**

1. **Run the Backtest**:
   Start the backtest by running the main script:
   ```bash
   python backtestchartmym-av2.py
   ```
   Alternatively, launch the GUI with:
   ```bash
   python mym-a coliseo.py
   ```

2. **Connecting to MT5**:
   The script will automatically attempt to connect to MT5 using the credentials provided. Ensure that your credentials are correctly set up in the script.

3. **Start the Backtest**:
   Once connected, the script fetches 600 days of historical data for the symbol `BTCUSD` and performs a series of trades based on LSTM predictions.

4. **Plotting the Results**:
   After the trades are simulated, the results are plotted showing the close price, trade entry, and exit points.

---

## **Trading Logic**

1. **Fetching Data**:
   The script fetches historical data from MT5 using a **4-hour timeframe** for 600 days.

2. **Data Preprocessing**:
   The close price data is scaled using **MinMaxScaler** to fit the LSTM model.

3. **LSTM Model**:
   The model is trained with a **60-timestep window**, meaning the model looks back at the previous 60 points to predict the next one.

4. **Trade Simulation**:
   - The model predicts the future price at specific intervals.
   - A trade (buy/sell) is opened if the predicted price suggests a favorable direction.
   - The trade is closed when a profit threshold (99.9/lot size) is met, or it hits a stop-loss.

---

## **Model Training**

The LSTM model is trained using 80% of the historical data, with 20% reserved for validation. The model structure is as follows:

1. **LSTM Layers**:
   - Two LSTM layers with 100 units each.
   - **Dropout** layers to prevent overfitting.
2. **Dense Layers**:
   - Two fully connected layers with 50 and 1 unit(s), respectively.
3. **Loss Function**:
   The model uses **mean squared error (MSE)** as the loss function and **Adam optimizer**.

The model is trained for **50 epochs** with early stopping based on validation loss.

---

## **Results and Visualization**

At the end of the backtest, a chart is generated showing:
- **Close Price**: The BTC/USD price.
- **Entry Points**: Where trades were opened.
- **Exit Points**: Where trades were closed.
- **Profit/Loss**: Displayed on the title with the final balance after the backtest.

---

## **Limitations**

- The backtest is based on historical data and does not guarantee future performance.
- The model might suffer from overfitting despite using dropout layers.
- **Random trade positions** may introduce unpredictability in results.

---

## **Future Improvements**

- **Additional Features**: Consider adding more indicators such as moving averages to refine trading decisions.
- **Real-time Trading**: Extend the backtest to support live trading on MT5.
- **Extended Models**: Try more advanced models like transformers or ensemble methods for better predictions.
- **Hyperparameter Tuning**: Use techniques like **Bayesian Optimization** to fine-tune the model's parameters.

---

## **License**

This project is licensed under the MIT License.

---

Feel free to modify the README to suit your specific project requirements!
