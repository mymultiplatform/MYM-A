import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys
from tqdm import tqdm
import gc
import torch
import dask.dataframe as dd
import psutil

def get_gpu_memory_usage():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0

def clear_gpu_memory():
    """Clear GPU memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def batch_to_gpu(df_batch: pd.DataFrame) -> dict:
    """Convert batch data to GPU tensors"""
    gpu_data = {}
    numeric_cols = df_batch.select_dtypes(include=[np.number]).columns
    
    try:
        for col in numeric_cols:
            gpu_data[col] = torch.tensor(df_batch[col].values, device='cuda', dtype=torch.float32)
    except Exception as e:
        print(f"Error converting column {col} to GPU: {str(e)}")
        clear_gpu_memory()
        
    return gpu_data

def validate_data(log_dir: str, dtype_dict: Dict, batch_size_gb: float = 10.5) -> Tuple[bool, List[str]]:
    """
    CUDA-optimized data validation with batch processing
    
    Args:
        log_dir: Directory containing CSV files
        dtype_dict: Dictionary of column dtypes
        batch_size_gb: Target batch size in GB
    """
    issues = []
    
    def estimate_rows_per_batch(sample_df: pd.DataFrame, target_gb: float) -> int:
        """Estimate number of rows that fit in target_gb"""
        bytes_per_row = sample_df.memory_usage(deep=True).sum() / len(sample_df)
        return int((target_gb * 1024**3) / bytes_per_row)
    
    def process_gpu_batch(gpu_data: dict) -> List[str]:
        batch_issues = []
        try:
            # Process each column on GPU
            for col_name, tensor in gpu_data.items():
                # Check for nulls (NaN values)
                nan_count = torch.isnan(tensor).sum().item()
                if nan_count > 0:
                    batch_issues.append(f"Found {nan_count} null values in {col_name}")
                
                # Check for infinites
                inf_count = torch.isinf(tensor).sum().item()
                if inf_count > 0:
                    batch_issues.append(f"Found {inf_count} infinite values in {col_name}")
                
                # Statistical checks
                if tensor.numel() > 0:
                    mean = tensor.mean()
                    std = tensor.std()
                    if not torch.isnan(mean) and not torch.isnan(std):
                        if std > mean * 1000:
                            batch_issues.append(f"Extreme values detected in {col_name}")
                
        except Exception as e:
            batch_issues.append(f"Error processing GPU batch: {str(e)}")
            
        return batch_issues

    def process_file(file_path: Path) -> List[str]:
        file_issues = []
        try:
            # Read first chunk to estimate batch size
            sample_df = pd.read_csv(file_path, nrows=1000, dtype=dtype_dict)
            rows_per_batch = estimate_rows_per_batch(sample_df, batch_size_gb)
            
            # Process file in batches
            for chunk in tqdm(pd.read_csv(file_path, dtype=dtype_dict, chunksize=rows_per_batch),
                            desc=f"Processing {file_path.name}"):
                
                # Convert batch to GPU
                print(f"\nCurrent GPU memory usage: {get_gpu_memory_usage():.2f} GB")
                gpu_data = batch_to_gpu(chunk)
                
                # Process the batch
                batch_issues = process_gpu_batch(gpu_data)
                file_issues.extend(batch_issues)
                
                # Clear GPU memory
                for tensor in gpu_data.values():
                    del tensor
                clear_gpu_memory()
                print(f"GPU memory after cleanup: {get_gpu_memory_usage():.2f} GB")
                
        except Exception as e:
            file_issues.append(f"Error processing file {file_path.name}: {str(e)}")
            
        return file_issues

    print("Starting data validation...")
    
    try:
        # Configure CUDA for optimal performance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        csv_files = list(Path(log_dir).glob('*.csv'))
        if not csv_files:
            issues.append(f"No CSV files found in {log_dir}")
            return False, issues
        
        print(f"Found {len(csv_files)} CSV files to validate")
        
        for file_path in csv_files:
            print(f"\nProcessing file: {file_path.name}")
            file_issues = process_file(file_path)
            
            if file_issues:
                issues.append(f"\nIssues in {file_path.name}:")
                issues.extend([f"  - {issue}" for issue in file_issues])
            
            clear_gpu_memory()
            
    except Exception as e:
        issues.append(f"Critical error during validation: {str(e)}")
        return False, issues
    
    return len(issues) == 0, issues

def main():
    config_dtype_dict = {
        'rtype': 'int64',
        'publisher_id': 'int64',
        'instrument_id': 'int64',
        'action': 'object',
        'side': 'object',
        'depth': 'int64',
        'price': 'float64',
        'size': 'float64',
        'flags': 'float64',
        'ts_in_delta': 'float64',
        'sequence': 'float64',
        'symbol': 'object',
        'adjusted_price': 'float64',
        'scaled_price': 'float64',
        't_delta': 'float64',
        'n_delta': 'float64',
        't_delta_numeric': 'float64'
    }
    
    log_dir = r'C:\Users\cinco\Desktop\DATA FOR SCRIPTS\log'
    
    # Print system info
    print("System Information:")
    print(f"Available GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"System RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB\n")
    
    is_valid, issues = validate_data(log_dir, config_dtype_dict)
    
    if is_valid:
        print("\n✅ Data validation passed. No issues found.")
    else:
        print("\n❌ Data validation failed. Issues found:")
        for issue in issues:
            print(issue)
        sys.exit(1)

if __name__ == "__main__":
    main()