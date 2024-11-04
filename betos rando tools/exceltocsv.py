import pandas as pd
import os
from pathlib import Path

def process_itch_files(directory_path='C:\\Users\\cinco\\Desktop\\DATA FOR SCRIPTS\\data bento data\\NVDA'):
    """
    Process all ITCH data files in a directory and combine them into a single CSV
    
    Parameters:
    directory_path (str): Path to the directory containing ITCH files
    """
    try:
        directory = Path(directory_path)
        all_data = []
        processed_files = 0
        
        print(f"Scanning directory: {directory}")
        
        # Process each file in the directory
        for file_path in directory.iterdir():
            if file_path.is_file() and 'itch' in file_path.name.lower():
                try:
                    print(f"\nProcessing file: {file_path.name}")
                    # Read the data with space separator
                    df = pd.read_csv(file_path, sep='\s+')
                    
                    # Convert timestamp columns to datetime
                    timestamp_cols = ['ts_recv', 'ts_event']
                    for col in timestamp_cols:
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col])
                    
                    # Add source file information
                    df['source_file'] = file_path.name
                    
                    all_data.append(df)
                    processed_files += 1
                    print(f"Successfully processed: {file_path.name} - {len(df)} rows")
                    
                except Exception as e:
                    print(f"Error processing {file_path.name}: {str(e)}")
                    continue
        
        if not all_data:
            print("No ITCH files were found or processed successfully.")
            return
            
        # Combine all dataframes
        print("\nCombining all data...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by timestamp if it exists
        if 'ts_recv' in combined_df.columns:
            print("Sorting by timestamp...")
            combined_df = combined_df.sort_values('ts_recv')
        
        # Create output filename
        output_file = "combined_NVDA_ITCH_data.csv"
        output_path = directory / output_file
        
        # Save to CSV
        print(f"Saving combined data to {output_path}")
        combined_df.to_csv(output_path, index=False)
        
        print(f"\nProcess Summary:")
        print(f"Processed {processed_files} files")
        print(f"Total rows in combined file: {len(combined_df)}")
        print(f"Columns: {', '.join(combined_df.columns)}")
        print(f"Date range: {combined_df['ts_recv'].min()} to {combined_df['ts_recv'].max()}")
        print(f"Output saved to: {output_path}")
        
        # Display sample of combined data
        print("\nSample of combined data:")
        print(combined_df.head())
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    process_itch_files()