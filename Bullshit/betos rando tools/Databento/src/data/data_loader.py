import pandas as pd
from pathlib import Path
from tqdm import tqdm



def process_csv_file(file_path, start_date, end_date):
    """Process a single CSV file and return cleaned DataFrame.
    
    Args:
        file_path: Path to the CSV file
        start_date: Start date for filtering data
        end_date: End date for filtering data
        
    Returns:
        DataFrame with ts_event and price columns
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['ts_event'], dtype={'price': 'float32'})
        # Filter data between start and end dates
        mask = (df['ts_event'] >= start_date) & (df['ts_event'] <= end_date)
        df = df[mask]
        if not df.empty:
            df = df[['ts_event', 'price']].set_index('ts_event')
        return df
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

