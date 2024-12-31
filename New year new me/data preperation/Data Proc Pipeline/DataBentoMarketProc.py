import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Optional
import warnings
from scipy import stats
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

class FinancialDataPreprocessor:
    def __init__(self):
        """Initialize the preprocessor with paths and configurations."""
        self.BASE_DIR = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\data bento data"
        self.INPUT_DIR = os.path.join(self.BASE_DIR, 'NVDA')
        self.OUTPUT_DIR = os.path.join(self.BASE_DIR, 'finalLSTM')
        
        # Set number of CPU cores to use (leave 2 cores free for system)
        self.n_cores = max(1, multiprocessing.cpu_count() - 2)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize storage for transformation parameters
        self.scaling_params = {}
        self.normalization_params = {}
        
        # Define final dtype dictionary for output columns
        self.FINAL_DTYPE_DICT = {
            'n_delta': 'float32',
            'normalized_returns': 'float32',
            'returns_squared': 'float32',
            'rolling_vol_5': 'float32',
            'rolling_vol_15': 'float32',
            'rolling_vol_30': 'float32',
            'rolling_mean_5': 'float32',
            'rolling_mean_15': 'float32',
            'returns_squared_log_normalized': 'float32',
            'rolling_vol_5_log_normalized': 'float32',
            'rolling_vol_15_log_normalized': 'float32',
            'rolling_vol_30_log_normalized': 'float32',
            'rolling_mean_5_log_normalized': 'float32',
            'rolling_mean_15_log_normalized': 'float32'
        }

    def setup_logging(self):
        """Configure logging with detailed formatting."""
        log_file = os.path.join(self.OUTPUT_DIR, 'preprocessing.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def read_csv_file(self, file):
        """Read a single CSV file."""
        try:
            df = pd.read_csv(
                file,
                skiprows=1,
                names=['ts_recv', 'ts_event', 'rtype', 'publisher_id', 
                      'instrument_id', 'action', 'side', 'depth', 'price',
                      'size', 'flags', 'ts_in_delta', 'sequence', 'symbol'],
                skipinitialspace=True
            )
            
            # Add filename column
            df.insert(0, 'filename', file.name)
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading {file.name}: {str(e)}")
            return None
        

    def read_input_data(self):
        """Read and combine all CSV files from the input directory using parallel processing."""
        self.logger.info("Starting data loading process")
        
        files = list(Path(self.INPUT_DIR).glob('*.csv'))
        
        # Use ThreadPoolExecutor for parallel file reading
        with ThreadPoolExecutor(max_workers=self.n_cores) as executor:
            dfs = list(executor.map(self.read_csv_file, files))
        
        # Remove None values from failed reads
        dfs = [df for df in dfs if df is not None]
        
        if not dfs:
            raise ValueError("No files were successfully read")
        
        # Combine all dataframes
        self.df = pd.concat(dfs, ignore_index=True)
        
        # Convert timestamps
        self.df['ts_event'] = pd.to_datetime(self.df['ts_event'])
        
        self.logger.info(f"Loaded {len(files)} files, total rows: {len(self.df):,}")
        self.logger.info(f"Time range: {self.df['ts_event'].min()} to {self.df['ts_event'].max()}")
        
        # Create backup
        self.original_df = self.df.copy()
        

    def save_processed_data(self, filename='processed_data.parquet'):
        """Save the processed DataFrame to the output directory."""
        output_path = os.path.join(self.OUTPUT_DIR, filename)
        
        # Verify all required columns are present with correct dtypes
        for col, dtype in self.FINAL_DTYPE_DICT.items():
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
            self.df[col] = self.df[col].astype(dtype)
        
        # Save to parquet format
        self.df.to_parquet(output_path)
        self.logger.info(f"Saved processed data to {output_path}")
        
        # Save transformation parameters
        params_path = os.path.join(self.OUTPUT_DIR, 'transformation_params.json')
        pd.Series({
            **self.scaling_params,
            **self.normalization_params
        }).to_json(params_path)
        self.logger.info(f"Saved transformation parameters to {params_path}")

    def process_pipeline(self):
        """Execute the full preprocessing pipeline."""
        try:
            self.read_input_data()
            
            # Execute all preprocessing steps
            self.adjust_for_splits()
            self.implement_price_scaling()
            self.calculate_time_deltas()
            self.process_sequence_deltas()
            self.calculate_log_returns()
            self.transform_time_deltas()
            self.implement_normalization()
            self.calculate_squared_returns()
            self.calculate_rolling_volatility()
            self.calculate_rolling_means()
            self.log_normalize_squared_returns()
            self.log_normalize_volatility()
            self.log_normalize_means()
            
            # Save the processed data
            self.save_processed_data()
            
            self.logger.info("Pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
        
def adjust_for_splits(self):
    """Adjust price data for stock splits and add as new column"""
    self.logger.info("Starting price split adjustment")
    
    # Store original data
    df = self.df.copy()
    
    # Sort by timestamp
    df = df.sort_values('ts_event')
    
    # Create new adjusted_price column
    df['adjusted_price'] = df['price'].copy()
    
    # Convert timestamps and handle timezone
    timestamps = pd.to_datetime(df['ts_event'])
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize('UTC')
    
    # Apply split adjustments
    for split_date, ratio in [
        ('2021-07-20T00:00:00Z', 4.0),
        ('2024-06-10T00:00:00Z', 10.0)
    ]:
        split_datetime = pd.to_datetime(split_date)
        if split_datetime.tz is None:
            split_datetime = split_datetime.tz_localize('UTC')
        
        mask = timestamps < split_datetime
        df.loc[mask, 'adjusted_price'] = df.loc[mask, 'adjusted_price'] / ratio
    
    self.logger.info("Split Adjustment Summary:")
    self.logger.info(f"Time range: {timestamps.min()} to {timestamps.max()}")
    self.logger.info(f"Original price range: {df['price'].min():.2f} to {df['price'].max():.2f}")
    self.logger.info(f"Adjusted price range: {df['adjusted_price'].min():.2f} to {df['adjusted_price'].max():.2f}")
    
    self.df = df

def implement_price_scaling(self):
    """Scale the split-adjusted price data using Z-score normalization"""
    self.logger.info("Starting price scaling")
    
    # Prepare data using adjusted_price column
    price_data = self.df['adjusted_price'].values.reshape(-1, 1)
    
    # Calculate mean and standard deviation
    mean = np.mean(price_data)
    std = np.std(price_data)
    
    # Add buffer to standard deviation (10% buffer)
    buffered_std = std * 1.1
    
    # Perform Z-score normalization
    scaled_data = (price_data - mean) / buffered_std
    
    # Add scaled price as new column
    self.df['scaled_price'] = scaled_data.flatten()
    
    # Store scaling parameters
    self.scaling_params.update({
        'price_mean': float(mean),
        'price_std': float(buffered_std)
    })
    
    self.logger.info("\nScaling Summary:")
    self.logger.info(f"Mean: {mean:.2f}")
    self.logger.info(f"Standard Deviation: {std:.2f}")
    self.logger.info(f"Buffered Standard Deviation: {buffered_std:.2f}")
    self.logger.info(f"Scaled range: {scaled_data.min():.4f} to {scaled_data.max():.4f}")

def calculate_time_deltas(self):
    """Calculate time differences between consecutive events"""
    self.logger.info("Calculating time deltas")
    
    # Calculate time differences in nanoseconds
    self.df['t_delta'] = self.df['ts_event'].diff()
    
    # Replace NaN in first row with 0
    self.df['t_delta'] = self.df['t_delta'].fillna(pd.Timedelta(0))
    
    self.logger.info("\nTime delta statistics (in nanoseconds):")
    self.logger.info(self.df['t_delta'].describe())

def transform_time_deltas(self):
    """Normalize time deltas using z-score normalization"""
    self.logger.info("Normalizing time deltas")
    
    # Convert timedelta to numeric seconds
    time_seconds = self.df['t_delta'].apply(lambda x: x.total_seconds())
    
    # Calculate z-scores
    mean = time_seconds.mean()
    std = time_seconds.std()
    
    # Store normalization parameters
    self.normalization_params.update({
        'delta_mean': float(mean),
        'delta_std': float(std)
    })
    
    # Calculate normalized deltas
    self.df['n_delta'] = (time_seconds - mean) / std
    
    self.logger.info("\nNormalized time delta statistics:")
    self.logger.info(self.df['n_delta'].describe())
# Set pandas to use all available CPU cores
pd.options.compute.use_numba = True
pd.options.compute.use_threads = True

# Initialize and run the pipeline
if __name__ == '__main__':
    preprocessor = FinancialDataPreprocessor()
    preprocessor.process_pipeline()