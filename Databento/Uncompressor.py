import databento as db
import pandas as pd
from pathlib import Path
import os
from datetime import datetime

def convert_dbn_directory(input_dir: str, output_dir: str):
    """
    Convert all .dbn.zst files in a directory to CSV files, organizing them by date.
    
    Args:
        input_dir (str): Directory containing .dbn.zst files
        output_dir (str): Directory where CSV files will be saved
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .dbn.zst files in input directory and its subdirectories
    input_path = Path(input_dir)
    dbn_files = list(input_path.rglob("*.dbn.zst"))
    
    print(f"Found {len(dbn_files)} .dbn.zst files to convert")
    
    for file_path in dbn_files:
        try:
            # Read the DBN file
            store = db.DBNStore.from_file(str(file_path))
            
            # Get the date range for this file
            start_date = store.start.strftime('%Y%m%d')
            end_date = store.end.strftime('%Y%m%d') if store.end else start_date
            
            # Create filename with date range and symbol info
            symbols_str = '-'.join(store.symbols) if store.symbols else 'all'
            if len(symbols_str) > 50:  # Truncate if too long
                symbols_str = symbols_str[:47] + '...'
            
            filename = f"{store.dataset.lower()}_{symbols_str}_{start_date}_to_{end_date}.csv"
            output_file = output_path / filename
            
            # Convert to CSV
            store.to_csv(
                str(output_file),
                pretty_ts=True,
                pretty_px=True,
                map_symbols=True
            )
            
            print(f"\nProcessed file: {file_path.name}")
            print(f"Created: {output_file.name}")
            print(f"Dataset: {store.dataset}")
            print(f"Schema: {store.schema}")
            print(f"Date Range: {start_date} to {end_date}")
            print(f"Size: {store.nbytes:,} bytes")
            print("-" * 80)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

def verify_conversion(output_dir: str):
    """
    Verify the converted CSV files and print summary information.
    
    Args:
        output_dir (str): Directory containing converted CSV files
    """
    output_path = Path(output_dir)
    csv_files = list(output_path.glob("*.csv"))
    
    print("\nConversion Summary:")
    print(f"Total CSV files created: {len(csv_files)}")
    
    if csv_files:
        print("\nCreated files:")
        for csv_file in csv_files:
            file_size = csv_file.stat().st_size
            print(f"- {csv_file.name} ({file_size:,} bytes)")

if __name__ == "__main__":
    # Specify your input and output directories
    input_directory = "/Users/jazzhashzzz/Desktop/MYM-A/databento_data"  # Directory containing .dbn.zst files
    output_directory = "/Users/jazzhashzzz/Desktop/data for scripts/data bento data/NVDA"  # Directory where CSV files will be saved
    
    # Convert all files
    convert_dbn_directory(input_directory, output_directory)
    
    # Verify the conversion
    verify_conversion(output_directory)