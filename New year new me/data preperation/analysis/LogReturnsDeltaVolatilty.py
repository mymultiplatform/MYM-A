import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import torch

INPUT_DIR = r'C:\Users\cinco\Desktop\DATA FOR SCRIPTS\log'
OUTPUT_DIR = r'C:\Users\cinco\Desktop\DATA FOR SCRIPTS\final'

def calculate_enhanced_features(df):
    """Calculate additional features for the LSTM model"""
    
    # Calculate returns squared
    df['returns_squared'] = df['normalized_returns'] ** 2
    
    # Calculate rolling volatilities
    df['rolling_vol_5'] = df['normalized_returns'].rolling(window=5).std()
    df['rolling_vol_15'] = df['normalized_returns'].rolling(window=15).std()
    df['rolling_vol_30'] = df['normalized_returns'].rolling(window=30).std()
    
    # Calculate rolling means
    df['rolling_mean_5'] = df['normalized_returns'].rolling(window=5).mean()
    df['rolling_mean_15'] = df['normalized_returns'].rolling(window=15).mean()
    
    # Fill NaN values created by rolling calculations
    rolling_cols = ['rolling_vol_5', 'rolling_vol_15', 'rolling_vol_30', 
                    'rolling_mean_5', 'rolling_mean_15']
    df[rolling_cols] = df[rolling_cols].fillna(method='bfill')
    
    return df

def process_single_file(file_path, output_dir):
    """Process a single CSV file"""
    essential_columns = [
        'ts_event',
        'normalized_returns',
        'n_delta'
    ]
    
    dtype_dict = {
        'normalized_returns': 'float64',
        'n_delta': 'float64'
    }
    
    df = pd.read_csv(
        file_path,
        usecols=essential_columns,
        dtype=dtype_dict,
        parse_dates=['ts_event']
    )
    
    # Calculate enhanced features
    df = calculate_enhanced_features(df)
    
    # Save to new location
    output_path = output_dir / file_path.name
    df.to_csv(output_path, index=False)
    
    return output_path

def prepare_dataset():
    """Prepare the enhanced dataset"""
    # Create output directory if it doesn't exist
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of input files
    input_dir = Path(INPUT_DIR)
    input_files = list(input_dir.glob('*.csv'))
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for file_path in input_files:
            future = executor.submit(process_single_file, file_path, output_dir)
            futures.append(future)
        
        # Show progress bar
        for _ in tqdm(futures, desc="Processing files"):
            _.result()

    print(f"Processed {len(input_files)} files. Enhanced dataset saved to {output_dir}")

def load_test_data(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dfs = []
    files = sorted(Path(OUTPUT_DIR).glob('*.csv'))  # Changed to OUTPUT_DIR
    
    chunksize = 5000000
    
    usecols = [
        'ts_event', 
        'normalized_returns', 
        'n_delta',
        'returns_squared',
        'rolling_vol_5',
        'rolling_vol_15',
        'rolling_vol_30',
        'rolling_mean_5',
        'rolling_mean_15'
    ]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        for f in tqdm(files, desc="Processing CSV files"):
            df = pd.read_csv(
                f,
                dtype=config.DTYPE_DICT,
                parse_dates=['ts_event'],
                usecols=usecols,
                low_memory=False,
                engine='c'
            )
            dfs.append(df)

    final_data = pd.concat(dfs, ignore_index=True)
    final_data['ts_event'] = pd.to_datetime(final_data['ts_event']).dt.tz_localize('UTC')
    filtered_data = final_data[final_data['ts_event'] >= config.TEST_START]
    
    return filtered_data.sort_values('ts_event')

if __name__ == "__main__":
    # First prepare the enhanced dataset
    prepare_dataset()