import yfinance as yf
import os

def fetch_stock_data():
    # Available time periods
    periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max']
    
    # Get stock symbol from user
    symbol = input("Enter stock symbol (e.g., AAPL): ").upper()
    
    # Show available periods
    print("\nAvailable time periods:")
    for i, period in enumerate(periods, 1):
        print(f"{i}. {period}")
    
    # Get period choice from user
    while True:
        try:
            choice = int(input("\nEnter the number of your chosen time period: "))
            if 1 <= choice <= len(periods):
                selected_period = periods[choice-1]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Create directory if it doesn't exist
    save_dir = "/Users/jazzhashzzz/Desktop/data for scripts/Yfinance Data"
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Fetch the data
        stock = yf.Ticker(symbol)
        df = stock.history(period=selected_period)
        
        # Save to CSV
        filename = f"{symbol}_{selected_period}.csv"
        filepath = os.path.join(save_dir, filename)
        df.to_csv(filepath)
        print(f"\nData saved successfully to: {filepath}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the function
if __name__ == "__main__":
    fetch_stock_data()