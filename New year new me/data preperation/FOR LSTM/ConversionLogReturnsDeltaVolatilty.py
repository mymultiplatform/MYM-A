import pandas as pd
import numpy as np
from typing import Dict, Union, List, Tuple
import os
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = "/Users/jazzhashzzz/Desktop/data for scripts/data bento data/NVDA"
OUTPUT_DIR = "/Users/jazzhashzzz/Desktop/data for scripts/lstm data"

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
    """
    adjusted_df = df.copy()
    original_columns = list(adjusted_df.columns)
    adjusted_df = adjusted_df.sort_values('ts_event')
    
    if not isinstance(adjusted_df, pd.DataFrame):
        adjusted_df = adjusted_df.to_pandas()
    
    adjusted_df['adjusted_price'] = adjusted_df['price'].copy()
    timestamps = pd.to_datetime(adjusted_df['ts_event'])
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize('UTC')
    
    for split_date, ratio in config.SPLITS_INFO:
        split_datetime = pd.to_datetime(split_date)
        if split_datetime.tz is None:
            split_datetime = split_datetime.tz_localize('UTC')
        mask = timestamps < split_datetime
        adjusted_df.loc[mask, 'adjusted_price'] = adjusted_df.loc[mask, 'adjusted_price'] / ratio
    
    final_columns = original_columns + ['adjusted_price']
    adjusted_df = adjusted_df[final_columns]
    return adjusted_df

def process_data_for_lstm(df: pd.DataFrame) -> pd.DataFrame:
    SPLIT_TIME = '2024-09-16T08:00:04.248723179Z'
    
    df = adjust_for_splits(df)
    
    # Calculate all features on full dataset
    df['returns'] = df['adjusted_price'].pct_change()
    df['delta'] = pd.to_datetime(df['ts_event']).diff().dt.total_seconds()
    df['returns_squared'] = df['returns'] ** 2
    
    windows = [5, 15, 30]
    for window in windows:
        df[f'rolling_vol_{window}'] = df['returns'].rolling(window).std() * np.sqrt(252)
        df[f'rolling_mean_{window}'] = df['returns'].rolling(window).mean()
    
    columns_to_log = ['delta', 'returns_squared'] + \
                    [f'rolling_vol_{w}' for w in windows] + \
                    [f'rolling_mean_{w}' for w in windows]
    
    for col in columns_to_log:
        df[f'{col}_log'] = np.log1p(df[col].abs()) * np.sign(df[col])
    
    # Split data
    train_mask = pd.to_datetime(df['ts_event']) < pd.to_datetime(SPLIT_TIME)
    train_df = df[train_mask].copy()
    test_df = df[~train_mask].copy()
    
    def normalize_features(split_df: pd.DataFrame) -> pd.DataFrame:
        def normalize(series):
            return (series - series.mean()) / series.std()
        
        result_df = pd.DataFrame({'ts_event': split_df['ts_event']})
        
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
                result_df[new_col] = np.log1p(split_df[old_col].abs()) * np.sign(split_df[old_col])
                result_df[new_col] = normalize(result_df[new_col])
            else:
                result_df[new_col] = normalize(split_df[old_col])
        
        return result_df
    
    processed_train = normalize_features(train_df)
    processed_test = normalize_features(test_df)
    
    processed_train['dataset'] = 'train'
    processed_test['dataset'] = 'test'
    combined_df = pd.concat([processed_train, processed_test])
    
    return combined_df

def process_all_files():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    for filename in tqdm(files, desc="Processing files"):
        input_path = os.path.join(INPUT_DIR, filename)
        df = pd.read_csv(input_path)
        processed_df = process_data_for_lstm(df)
        output_filename = f"processed_{filename}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        processed_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    process_all_files()