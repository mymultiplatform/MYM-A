import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def prepare_daily_data(file_path):
    """
    Aggregate minute data to daily level for GARCH modeling
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        print(f"\nRaw data shape: {df.shape}")
        
        # Convert timestamp and set index
        df['ts_event'] = pd.to_datetime(df['ts_event'])
        df.set_index('ts_event', inplace=True)
        
        # Calculate daily OHLC prices
        daily = pd.DataFrame()
        daily['open'] = df['mid_price'].resample('D').first()
        daily['high'] = df['mid_price'].resample('D').max()
        daily['low'] = df['mid_price'].resample('D').min()
        daily['close'] = df['mid_price'].resample('D').last()
        
        # Calculate true daily returns (close-to-close)
        daily['returns'] = np.log(daily['close'] / daily['close'].shift(1))
        
        # Calculate daily realized volatility from minute returns
        daily['realized_vol'] = df.groupby(pd.Grouper(freq='D'))['returns'].std()
        
        # Remove days with no trading
        daily = daily.dropna()
        
        print("\nDaily price and return statistics:")
        print(daily.describe())
        
        return daily
    
    except Exception as e:
        print(f"Error in prepare_daily_data: {str(e)}")
        raise

def rolling_garch_forecast(daily_data, train_window=60, forecast_days=5):  # Reduced train_window
    """
    Perform rolling GARCH forecasts on daily data
    """
    returns = daily_data['returns']
    print(f"\nTotal number of daily returns: {len(returns)}")
    
    # Initialize predictions DataFrame with the same index as daily_data
    predictions_df = pd.DataFrame(index=daily_data.index)
    predictions_df['Actual_Vol'] = daily_data['realized_vol']
    predictions_df['Predicted_Vol'] = np.nan  # Initialize the column with NaN
    
    # Start from the end and work backwards
    current_idx = len(returns) - forecast_days
    
    while current_idx > train_window:
        try:
            # Get training data
            train_returns = returns[current_idx - train_window:current_idx]
            
            # Fit GARCH(1,1) model
            model = arch_model(train_returns, vol='Garch', p=1, q=1, rescale=True)
            model_fit = model.fit(disp=False, show_warning=False)
            
            # Generate forecast
            forecast = model_fit.forecast(horizon=forecast_days)
            forecast_vol = np.sqrt(forecast.variance.values[-1, :])
            
            # Store predictions
            forecast_dates = returns.index[current_idx:current_idx + forecast_days]
            for i, date in enumerate(forecast_dates[:len(forecast_vol)]):  # Added safety check
                if date in predictions_df.index:  # Check if date exists in index
                    predictions_df.loc[date, 'Predicted_Vol'] = forecast_vol[i]
            
            if current_idx == len(returns) - forecast_days:
                print("\nGARCH Model Summary:")
                print(model_fit.summary().tables[1])
                print("\nParameter Estimates:")
                params = model_fit.params
                print(f"Omega (constant): {params['omega']:.6f}")
                print(f"Alpha (ARCH term): {params['alpha[1]']:.6f}")
                print(f"Beta (GARCH term): {params['beta[1]']:.6f}")
                print(f"Persistence: {params['alpha[1]'] + params['beta[1]']:.6f}")
            
            # Move window back
            current_idx -= forecast_days
            
        except Exception as e:
            print(f"Warning: Error in forecast window ending at {returns.index[current_idx]}: {str(e)}")
            current_idx -= forecast_days
            continue
    
    # Calculate errors where we have both predictions and actual values
    mask = ~(predictions_df['Predicted_Vol'].isna() | predictions_df['Actual_Vol'].isna())
    predictions_df['Error'] = np.where(mask,
                                     predictions_df['Predicted_Vol'] - predictions_df['Actual_Vol'],
                                     np.nan)
    
    # Drop rows without predictions
    predictions_df = predictions_df.dropna(subset=['Predicted_Vol'])
    
    print("\nPredictions summary:")
    print(predictions_df.describe())
    
    return predictions_df

def plot_predictions(predictions_df, window_days=30):
    """
    Plot the predictions and actual values
    """
    try:
        # Get the last window_days of data
        plot_data = predictions_df.tail(window_days)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot volatility
        ax1.plot(plot_data.index, plot_data['Actual_Vol'] * 100, 
                label='Realized Volatility', color='blue', linewidth=2)
        ax1.plot(plot_data.index, plot_data['Predicted_Vol'] * 100, 
                label='Predicted Volatility', color='red', linestyle='--', 
                marker='o', markersize=4)
        ax1.set_title('Daily Volatility: Realized vs Predicted (%)')
        ax1.set_ylabel('Volatility (%)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot prediction errors
        ax2.bar(plot_data.index, plot_data['Error'] * 100, color='green', alpha=0.6)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Prediction Errors (%)')
        ax2.set_ylabel('Error (%)')
        ax2.grid(True)
        
        # Format dates
        fig.autofmt_xdate()
        
        plt.tight_layout()
        plt.savefig('garch_predictions.png', dpi=300, bbox_inches='tight')
        print("\nPrediction plot saved as 'garch_predictions.png'")
        plt.close()
    
    except Exception as e:
        print(f"Error in plotting: {str(e)}")

def main():
    file_path = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\data bento data\NVDA2yearGarch.csv"
    
    try:
        print("Loading and processing data...")
        daily_data = prepare_daily_data(file_path)
        
        print("\nGenerating rolling GARCH forecasts...")
        predictions = rolling_garch_forecast(daily_data)
        
        # Calculate error metrics
        valid_errors = predictions['Error'].dropna()
        if len(valid_errors) > 0:
            mae = abs(valid_errors).mean() * 100
            rmse = np.sqrt((valid_errors**2).mean()) * 100
            
            print("\nForecast Error Metrics (in percentage points):")
            print(f"Mean Absolute Error: {mae:.4f}%")
            print(f"Root Mean Square Error: {rmse:.4f}%")
        
        print("\nGenerating plots...")
        plot_predictions(predictions)
        
        predictions.to_csv('volatility_predictions.csv')
        print("\nPredictions saved to 'volatility_predictions.csv'")
        
    except Exception as e:
        print(f"Error in execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()