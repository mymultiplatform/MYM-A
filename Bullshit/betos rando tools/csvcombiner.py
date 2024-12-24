import pandas as pd
import numpy as np
from glob import glob
import os
from datetime import datetime

def read_csv_file(file_path):
    """
    Read a single CSV file with proper column handling
    """
    try:
        # Read the first few lines to check the structure
        sample = pd.read_csv(file_path, nrows=5)
        
        # If file has a header, read normally
        if 'ts_recv' in sample.columns:
            df = pd.read_csv(file_path)
        else:
            # Define column names
            columns = ['ts_recv', 'ts_event', 'rtype', 'publisher_id', 'instrument_id', 
                      'action', 'side', 'depth', 'price', 'size', 'flags', 'sequence', 'symbol']
            df = pd.read_csv(file_path, names=columns)
        
        return df
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None

def combine_csv_files(folder_path, output_file):
    """
    Combine multiple CSV files into a single file
    
    Parameters:
    folder_path (str): Path to folder containing CSV files
    output_file (str): Path for output Excel file
    """
    print("Starting data processing...")
    
    all_data = []
    csv_files = glob(os.path.join(folder_path, '*.csv'))
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        try:
            print(f"\nProcessing file: {os.path.basename(csv_file)}")
            
            # Read the CSV file
            df = read_csv_file(csv_file)
            if df is None:
                continue
                
            # Convert timestamps
            df['ts_event'] = pd.to_datetime(df['ts_event'])
            df['ts_recv'] = pd.to_datetime(df['ts_recv'])
            
            # Convert numeric columns
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['size'] = pd.to_numeric(df['size'], errors='coerce')
            
            all_data.append(df)
            print(f"Processed {len(df)} rows")
            
        except Exception as e:
            print(f"Error processing file {csv_file}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("No data was successfully loaded from any CSV file")
    
    # Combine all dataframes
    print("\nCombining dataframes...")
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total combined rows: {len(combined_df)}")
    
    # Sort by event timestamp
    print("Sorting by timestamp...")
    combined_df = combined_df.sort_values('ts_event')
    
    # Save to Excel
    print(f"\nSaving data to {output_file}")
    combined_df.to_excel(output_file, index=False)
    
    print("\nProcessed data summary:")
    print(f"Total rows: {len(combined_df)}")
    print(f"Date range: {combined_df['ts_event'].min()} to {combined_df['ts_event'].max()}")
    
    return combined_df

if __name__ == "__main__":
    # Example usage
    folder_path = r'C:\Users\cinco\Desktop\DATA FOR SCRIPTS\data bento data\NVDA'
    output_file = r'C:\Users\cinco\Desktop\DATA FOR SCRIPTS\data bento data\combined_data.xlsx'
    
    try:
        combined_df = combine_csv_files(folder_path, output_file)
        print("\nData processing completed successfully!")
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")