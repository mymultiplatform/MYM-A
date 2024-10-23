import os
import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
from datetime import datetime

def load_and_prepare_data(folder_path, start_date, end_date):
    """
    Load all CSV files in the folder and prepare the data between start and end dates.
    """
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    columns = ['ts_recv', 'ts_event', 'rtype', 'publisher_id', 'instrument_id', 
               'action', 'side', 'depth', 'price', 'size', 'flags', 'sequence', 'symbol']

    df_list = []

    for file in all_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, names=columns, header=None)
        
        # Specify the format for the timestamp columns
        timestamp_format = '%Y-%m-%dT%H:%M:%S.%fZ'

        # Parse the timestamps using the specified format
        df['ts_event'] = pd.to_datetime(df['ts_event'], format=timestamp_format, errors='coerce')
        df['ts_recv'] = pd.to_datetime(df['ts_recv'], format=timestamp_format, errors='coerce')

        # Drop rows with invalid timestamps
        df = df.dropna(subset=['ts_event', 'ts_recv'])

        # Remove timezone information if present
        df['ts_event'] = df['ts_event'].dt.tz_localize(None)
        df['ts_recv'] = df['ts_recv'].dt.tz_localize(None)

        # Filter based on the start and end date
        df = df[(df['ts_event'] >= pd.to_datetime(start_date)) & (df['ts_event'] <= pd.to_datetime(end_date))]
        
        df_list.append(df)

    # Concatenate all data
    combined_df = pd.concat(df_list, ignore_index=True)

    # Separate Bid and Ask data
    bid_df = combined_df[combined_df['side'] == 'B']
    ask_df = combined_df[combined_df['side'] == 'A']

    # Process both sides (e.g., VWAP and size)
    grouped_bid = bid_df.groupby('ts_event').agg({
        'price': lambda x: np.average(x, weights=bid_df.loc[x.index, 'size']),
        'size': 'sum'
    }).reset_index()

    grouped_ask = ask_df.groupby('ts_event').agg({
        'price': lambda x: np.average(x, weights=ask_df.loc[x.index, 'size']),
        'size': 'sum'
    }).reset_index()

    return grouped_bid, grouped_ask

def calculate_realized_vol(returns, window=20):
    """Calculate realized volatility using rolling standard deviation"""
    return returns.rolling(window=window).std() * np.sqrt(252 * 1440)  # Annualized for minute data

def forecast_garch(train_data, forecast_dates):
    """
    Implement rolling GARCH forecasts for specific timestamps
    """
    results_list = []
    forecast_dates = pd.to_datetime(forecast_dates).tz_localize(None)
    
    print(f"Number of forecast dates: {len(forecast_dates)}")
    print("First few forecast dates:")
    print(forecast_dates[:3])
    
    for date in forecast_dates:
        print(f"\nProcessing forecast for {date}")
        # Get data up to this timestamp
        current_train = train_data[train_data['ts_event'] < date].copy()
        
        if len(current_train) < 100:
            print(f"Warning: Not enough training data for {date}. Only {len(current_train)} samples available.")
            continue
            
        try:
            # Prepare returns series for GARCH
            returns_series = current_train['returns']
            returns_series.index = current_train['ts_event']
            
            # Fit model on current training data
            current_model = arch_model(returns_series, p=2, q=2, rescale=True)
            current_fit = current_model.fit(disp='off')
            
            # Make forecast
            forecast = current_fit.forecast(horizon=1)
            predicted_vol = np.sqrt(forecast.variance.values[-1, 0])
            
            # Calculate realized volatility
            realized_vol = calculate_realized_vol(returns_series).iloc[-1]
            
            # Get actual return if available
            actual_return = train_data[train_data['ts_event'] == date]['returns'].iloc[0] if len(train_data[train_data['ts_event'] == date]) > 0 else np.nan
            
            results_list.append({
                'timestamp': date,
                'predicted_volatility': predicted_vol,
                'realized_volatility': realized_vol,
                'actual_return': actual_return
            })
            print(f"Successfully processed forecast for {date}")
            
        except Exception as e:
            print(f"Error processing forecast for {date}: {str(e)}")
            continue
    
    # Create DataFrame from successful forecasts
    if not results_list:
        raise ValueError("No successful forecasts were generated")
    
    results_df = pd.DataFrame(results_list)
    results_df.set_index('timestamp', inplace=True)
    
    print("\nNumber of successful forecasts:", len(results_df))
    return results_df

def plot_results(results):
    """
    Plot the forecasting results
    """
    plt.figure(figsize=(15, 8))
    
    # Plot volatilities
    plt.subplot(2, 1, 1)
    plt.plot(results.index, results['predicted_volatility'].values, 
             label='Predicted Volatility', marker='o')
    plt.plot(results.index, results['realized_volatility'].values, 
             label='Realized Volatility', marker='x')
    plt.title('Volatility Forecast vs Realized Volatility')
    plt.xticks(rotation=45)
    plt.legend()
    
    # Plot actual returns
    plt.subplot(2, 1, 2)
    valid_returns = results['actual_return'].dropna()
    if len(valid_returns) > 0:
        plt.plot(valid_returns.index, valid_returns.values, 
                label='Actual Returns', color='green', marker='o')
    plt.title('Actual Returns')
    plt.xticks(rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Define forecast dates
    forecast_dates = [
        '2024-09-03T23:59:53.247714364Z',
        '2024-09-03T23:59:58.337805416Z',
        '2024-09-03T23:59:58.409650975Z',
        '2024-09-03T23:59:58.428861596Z',
        '2024-09-03T23:59:58.635108520Z',
        '2024-09-03T23:59:59.619064982Z',
        '2024-09-03T23:59:59.884911621Z'
    ]
    
    # Folder containing CSV files
    folder_path = r'C:\Users\cinco\Desktop\DATA FOR SCRIPTS\data bento data\NVDA'
    
    # Training period (from 2022-01-20 to 2024-09-03)
    start_date = '2022-01-20'
    end_date = '2024-09-03'
    
    try:
        # Load and prepare data
        grouped_bid, grouped_ask = load_and_prepare_data(folder_path, start_date, end_date)
        print("\nData loaded successfully")
        
        # Merge bid and ask data (for returns, etc. you might want to use bid or midprice)
        train_data = pd.merge(grouped_bid, grouped_ask, on='ts_event', suffixes=('_bid', '_ask'))
        train_data['returns'] = np.log(train_data['price_bid'] / train_data['price_bid'].shift(1))
        
        # Drop NaN values in returns
        train_data.dropna(subset=['returns'], inplace=True)
        
        # Forecast GARCH
        results = forecast_garch(train_data, forecast_dates)
        print("\nForecasting completed successfully")
        
        # Plot results
        plot_results(results)
        print("\nPlotting completed")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == '__main__':
    main()
