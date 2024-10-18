import yfinance as yf
from datetime import datetime, timedelta
from arch_analysis import arch_analysis, perform_arch_analysis, perform_rolling_volatility_forecast
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

def main():
    print("Welcome to the ARCH Analysis Tool")

    symbol = get_valid_stock_symbol()

    end_date = datetime.now()
    start_date = datetime(2021, 1, 1)  # Start from 2021-01-01

    print(f"\nFetching data for {symbol} from {start_date.date()} to {end_date.date()}...")
    
    # Fetch data
    data = yf.download(symbol, start=start_date, end=end_date)
    
    if data is None or len(data) == 0:
        print(f"Failed to fetch data for {symbol}")
        return

    # Perform ARCH analysis
    model_fit, rolling_forecasts, actual_next_5_days, forecast_dates, returns = arch_analysis(data)

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['Close'])
    plt.title(f'{symbol} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')

    plt.subplot(2, 1, 2)
    plt.plot(forecast_dates, np.abs(actual_next_5_days), label='Actual Volatility', alpha=0.5)
    plt.plot(forecast_dates, rolling_forecasts, label='Predicted Volatility', color='red')
    plt.title('ARCH Volatility Forecast (Rolling 5-day)')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
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
    print("\nARCH Model Summary:")
    print(model_fit.summary())

if __name__ == "__main__":
    main()