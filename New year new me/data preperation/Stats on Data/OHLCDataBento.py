import pandas as pd
import os
from pathlib import Path

# Input and output directory paths
input_dir = '/Users/jazzhashzzz/Desktop/data for scripts/data bento data/SPY/SPY'
output_dir = '/Users/jazzhashzzz/Desktop/data for scripts/data bento data/SPY/OHCL SPY'

# Create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

def process_file(file_path):
    print(f"Starting to process file: {file_path}")
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Convert ts_event to datetime
    df['ts_event'] = pd.to_datetime(df['ts_event'], unit='ns')
    
    # Set ts_event as index for resampling
    df.set_index('ts_event', inplace=True)
    
    # Create 15-minute groups and aggregate
    result = pd.DataFrame()
    result['open'] = df['price'].resample('15min').first()
    result['high'] = df['price'].resample('15min').max()
    result['low'] = df['price'].resample('15min').min()
    result['close'] = df['price'].resample('15min').last()
    result['volume'] = df['size'].resample('15min').sum()
    
    # Handle side information
    def combine_sides(x):
        return ','.join(x.astype(str).unique())
    
    result['sides'] = df['side'].resample('15min').apply(combine_sides)
    
    # Reset index to get ts_event as a column
    result.reset_index(inplace=True)
    
    # Add 15-minute interval timestamp
    result['interval_timestamp'] = result['ts_event'].dt.floor('15min')
    
    # Reorder columns
    result = result[['ts_event', 'interval_timestamp', 'open', 'high', 'low', 'close', 'volume', 'sides']]
    
    return result

def main():
    # Process all CSV files in input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")
            
            try:
                # Process the file
                result_df = process_file(input_path)
                
                # Create output filename
                output_filename = f"OHLC_{filename}"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save to CSV
                result_df.to_csv(output_path, index=False)
                print(f"Successfully processed {filename} -> {output_filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# Run the main function
if __name__ == "__main__":
    print("Starting script...")
    main()
    print("Script completed.")