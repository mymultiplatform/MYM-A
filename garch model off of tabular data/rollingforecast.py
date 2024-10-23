import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

# Read and prepare the data
def prepare_data(df):
    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Calculate returns using close prices
    df['returns'] = 100 * df['close'].pct_change()
    
    # Sort by time to ensure proper sequence
    df = df.sort_values('time')
    
    # Calculate rolling volatility
    # Using a 24-hour window for daily volatility estimation
    df['realized_volatility'] = df['returns'].rolling(
        window=4,  # 24 hours
        min_periods=4  # Require at least 12 hours of data
    ).std() * np.sqrt(24)  # Annualize
    
    return df

def main():
    # Read your forex data
    print("Reading data...")
    df = pd.read_csv('/Users/jazzhashzzz/Desktop/data for scripts/hourly data/drive-download-20241022T200454Z-001/NVDA_CFD.US_H4_data.csv')
    
    print("\nPreparing data...")
    df = prepare_data(df)
    
    # Define specific forecast dates
    forecast_dates = pd.to_datetime([
    '1/2/24 16:00',
    '1/3/24 16:00',
    '1/4/24 16:00',
    '1/5/24 16:00',
    ])
    
    # Filter data for training (2022-2023)
    train_end = '2023-12-29'
    train_start = '2022-01-01'
    
    print("\nFiltering training data...")
    train_data = df[(df['time'] >= train_start) & (df['time'] <= train_end)]['returns'].dropna()
    
    print(f"\nTraining data shape: {train_data.shape}")
    print("\nFitting GARCH model...")
    
    # Fit GARCH model
    model = arch_model(train_data, vol='Garch', p=1, o=0, q=1, dist='normal')
    results = model.fit(disp='off')
    
    print("\nGenerating forecasts...")
    # Generate forecasts for each specific time point
    forecasts = results.forecast(horizon=len(forecast_dates))
    forecast_variance = np.sqrt(forecasts.variance.iloc[-1]) * np.sqrt(24)  # Annualize
    
    # Get actual volatility for comparison
    actual_data = df[df['time'].isin(forecast_dates)].copy()
    
    # Create forecast DataFrame with specific dates
    forecast_df = pd.DataFrame({
        'DateTime': forecast_dates,
        'Predicted_Volatility': forecast_variance[:len(forecast_dates)]
    }).set_index('DateTime')
    
    # Prepare actual volatility data
    actual_vol_df = pd.DataFrame({
        'DateTime': actual_data['time'],
        'Actual_Volatility': actual_data['realized_volatility']
    }).set_index('DateTime')
    
    # Combine predicted and actual
    comparison_df = pd.DataFrame({
        'Predicted_Volatility': forecast_df['Predicted_Volatility'],
        'Actual_Volatility': actual_vol_df['Actual_Volatility']
    })
    
    # Print data availability check
    print("\nData Check:")
    print(f"Number of forecast dates: {len(forecast_dates)}")
    print(f"Number of actual data points: {len(actual_data)}")
    print("\nSample of actual data:")
    print(actual_data[['time', 'returns', 'realized_volatility']].head())
    
    print("\nPlotting results...")
    # Plot results
    plt.figure(figsize=(15, 7))
    plt.plot(comparison_df.index, comparison_df['Predicted_Volatility'], 
             label='Predicted Volatility', marker='o')
    if not comparison_df['Actual_Volatility'].isna().all():
        plt.plot(comparison_df.index, comparison_df['Actual_Volatility'], 
                label='Realized Volatility', marker='s')
    plt.title('Forecasted vs Realized Volatility (Hourly)')
    plt.xlabel('Date Time')
    plt.ylabel('Annualized Volatility (%)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Print numerical comparison
    print("\nVolatility Comparison:")
    print(comparison_df.round(2))
    
    # Calculate forecast accuracy metrics
    valid_comparisons = comparison_df.dropna()
    if len(valid_comparisons) > 0:
        mape = np.mean(abs(valid_comparisons['Predicted_Volatility'] - valid_comparisons['Actual_Volatility']) / 
                      valid_comparisons['Actual_Volatility']) * 100
        rmse = np.sqrt(np.mean((valid_comparisons['Predicted_Volatility'] - valid_comparisons['Actual_Volatility'])**2))
        
        print("\nForecast Accuracy Metrics:")
        print(f"MAPE: {mape:.2f}%")
        print(f"RMSE: {rmse:.2f}")
    else:
        print("\nNot enough data points for accuracy metrics calculation")

if __name__ == "__main__":
    main()