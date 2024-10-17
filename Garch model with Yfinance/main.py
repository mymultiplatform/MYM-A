import yfinance as yf
from datetime import datetime, timedelta
from garch_analysis import garch_analysis, perform_garch_analysis, perform_rolling_volatility_forecast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_valid_stock_symbol():
    while True:
        symbol = input("Enter a valid stock symbol (e.g., AAPL for Apple Inc.): ").upper()
        try:
            stock = yf.Ticker(symbol)
            # Check if we can fetch any data for this symbol
            if not stock.history(period="1d").empty:
                return symbol
            else:
                print(f"No data available for {symbol}. Please try another symbol.")
        except Exception as e:
            print(f"Error: {e}. Please try another symbol.")

def get_time_period():
    periods = {
        "1": ("1 month", 30),
        "2": ("3 months", 90),
        "3": ("6 months", 180),
        "4": ("1 year", 365),
        "5": ("2 years", 730),
        "6": ("5 years", 1825)
    }

    print("\nChoose a time period:")
    for key, value in periods.items():
        print(f"{key}. {value[0]}")

    while True:
        choice = input("Enter your choice (1-6): ")
        if choice in periods:
            return periods[choice][1]
        else:
            print("Invalid choice. Please try again.")

def main():
    print("Welcome to the GARCH Analysis Tool")

    symbol = get_valid_stock_symbol()
    days = get_time_period()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"\nFetching data for {symbol} from {start_date.date()} to {end_date.date()}...")
    
    # Fetch data
    data = yf.download(symbol, start=start_date, end=end_date)
    
    if data is None or len(data) == 0:
        print(f"Failed to fetch data for {symbol}")
        return

    # Calculate returns
    returns = 100 * np.log(data['Close'] / data['Close'].shift(1)).dropna()

    # Perform GARCH analysis
    model_fit, forecasts = perform_garch_analysis(returns)

    # Perform rolling volatility forecast
    test_size = min(100, len(returns) // 3)  # Use at most 1/3 of the data for testing
    rolling_predictions = perform_rolling_volatility_forecast(returns, test_size)

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['Close'])
    plt.title(f'{symbol} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')

    plt.subplot(2, 1, 2)
    plt.plot(returns.index[-test_size:], returns[-test_size:], label='Actual Returns')
    plt.plot(returns.index[-test_size:], rolling_predictions, label='Predicted Volatility')
    plt.title('GARCH Volatility Forecast')
    plt.xlabel('Date')
    plt.ylabel('Returns / Volatility')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    stats_df = pd.DataFrame({
        'Statistic': ['Mean', 'Median', 'Standard Deviation', 'Skewness', 'Kurtosis', 
                      'Minimum', 'Maximum', '25th Percentile', '75th Percentile'],
        'Value': [returns.mean(), returns.median(), returns.std(), 
                  returns.skew(), returns.kurtosis(),
                  returns.min(), returns.max(), 
                  returns.quantile(0.25), returns.quantile(0.75)]
    })

    print("\nSummary Statistics:")
    print(stats_df.to_string(index=False))

    # Print model summary
    print("\nGARCH Model Summary:")
    print(model_fit.summary())

if __name__ == "__main__":
    main()