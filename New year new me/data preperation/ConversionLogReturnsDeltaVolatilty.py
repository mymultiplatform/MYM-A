import pandas as pd
import numpy as np
from typing import Dict, Union, List, Tuple
import os
from pathlib import Path

INPUT_DIR = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\data bento data\NVDA"
OUTPUT_DIR = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\data bento data\finaltest"

# Configuration class for split information
class Config:
    SPLITS_INFO = [
        ('2021-07-20', 4.0),
        ('2024-06-10', 10.0)
    ]

config = Config()

def adjust_for_splits(df):
    """
    Adjust price data for stock splits and add as new column.
    
    Args:
        df: DataFrame containing price and timestamp data
        
    Returns:
        DataFrame with added adjusted_price column
    """
    adjusted_df = df.copy()
    original_columns = list(adjusted_df.columns)
    
    # Process timestamps and sort data
    adjusted_df = adjusted_df.sort_values('ts_event')
    
    # Convert to pandas if needed
    if not isinstance(adjusted_df, pd.DataFrame):
        adjusted_df = adjusted_df.to_pandas()
    
    # Create new adjusted_price column
    adjusted_df['adjusted_price'] = adjusted_df['price'].copy()
    
    # Convert timestamps and handle timezone
    timestamps = pd.to_datetime(adjusted_df['ts_event'])
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize('UTC')
    
    # Apply split adjustments directly without using index
    for split_date, ratio in config.SPLITS_INFO:
        split_datetime = pd.to_datetime(split_date)
        if split_datetime.tz is None:
            split_datetime = split_datetime.tz_localize('UTC')
        
        mask = timestamps < split_datetime
        adjusted_df.loc[mask, 'adjusted_price'] = adjusted_df.loc[mask, 'adjusted_price'] / ratio
    
    # Make sure columns are in the right order
    final_columns = original_columns + ['adjusted_price']
    adjusted_df = adjusted_df[final_columns]
    
    print("\nSplit Adjustment Summary:")
    print(f"Time range: {timestamps.min()} to {timestamps.max()}")
    print(f"Original price range: {adjusted_df['price'].min():.2f} to {adjusted_df['price'].max():.2f}")
    print(f"Adjusted price range: {adjusted_df['adjusted_price'].min():.2f} to {adjusted_df['adjusted_price'].max():.2f}")
    
    return adjusted_df

def process_data_for_lstm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process market data for LSTM training.
    """
    # Apply split adjustments first
    df = adjust_for_splits(df)
    
    # Calculate returns using adjusted prices
    df['returns'] = df['adjusted_price'].pct_change()
    
    # Calculate time delta between events (in seconds)
    df['delta'] = pd.to_datetime(df['ts_event']).diff().dt.total_seconds()
    
    # Calculate rolling volatilities for different periods
    windows = [5, 15, 30]
    for window in windows:
        df[f'rolling_vol_{window}'] = df['returns'].rolling(window).std() * np.sqrt(252)
        df[f'rolling_mean_{window}'] = df['returns'].rolling(window).mean()
        
    # Log transform relevant columns
    df['returns_squared'] = df['returns'] ** 2
        
    columns_to_log = ['delta', 'returns_squared'] + \
                    [f'rolling_vol_{w}' for w in windows] + \
                    [f'rolling_mean_{w}' for w in [5, 15, 30]]
    
    for col in columns_to_log:
        df[f'{col}_log'] = np.log1p(df[col].abs()) * np.sign(df[col])
    
    # Z-score normalization
    def normalize(series):
        return (series - series.mean()) / series.std()
    
    # Initialize result DataFrame with ts_event
    result_df = pd.DataFrame({'ts_event': df['ts_event']})
    
    # Add normalized columns
    final_columns = {
        'delta_log': 'n_delta',
        'returns': 'normalized_returns',
        'returns': 'log_normalized_returns',
        'returns_squared_log': 'returns_squared_log_normalized',
        'rolling_vol_5_log': 'rolling_vol_5_log_normalized',
        'rolling_vol_15_log': 'rolling_vol_15_log_normalized',
        'rolling_vol_30_log': 'rolling_vol_30_log_normalized',
        'rolling_mean_5_log': 'rolling_mean_5_log_normalized',
        'rolling_mean_15_log': 'rolling_mean_15_log_normalized',
        'rolling_mean_30_log': 'rolling_mean_30_log_normalized'
    }
    
    for old_col, new_col in final_columns.items():
        if new_col == 'log_normalized_returns':
            # Special handling for logged normalized returns
            result_df[new_col] = np.log1p(df[old_col].abs()) * np.sign(df[old_col])
            result_df[new_col] = normalize(result_df[new_col])
        else:
            result_df[new_col] = normalize(df[old_col])
    
    return result_df

def process_all_files():
    """
    Process all CSV files in the input directory and save results to output directory.
    """
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('.csv'):
            print(f"\nProcessing {filename}...")
            
            input_path = os.path.join(INPUT_DIR, filename)
            df = pd.read_csv(input_path)
            
            processed_df = process_data_for_lstm(df)
            
            output_filename = f"processed_{filename}"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            processed_df.to_csv(output_path, index=False)
            
            print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    process_all_files()