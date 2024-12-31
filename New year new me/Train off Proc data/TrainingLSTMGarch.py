import multiprocessing
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from pathlib import Path
import torch
import numpy as np
from tqdm.auto import tqdm
import gc
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import gc
import torch
import psutil
import sys
import torch
import pandas as pd
import numpy as np
from torch import nn
import yfinance as yf
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from torch.optim.lr_scheduler import OneCycleLR
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from multiprocessing import freeze_support
# GPU Optimization Settings
def setup_training_environment():
    if torch.cuda.is_available():
        # Increase memory fraction
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available memory
        
        # Enable TF32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory allocator settings
        torch.cuda.memory.set_per_process_memory_fraction(0.95)
        
        # Set to faster backends
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        return True
    return False



@dataclass
class Config:
    def __init__(self):
        # Local path configurations
        self.BASE_DIR = r'C:\Users\cinco\Desktop\DATA FOR SCRIPTS\data bento'
        self.DATA_DIR = r'C:\Users\cinco\Desktop\DATA FOR SCRIPTS\data bento data\finaltest'
        self.BEST_PATH_DIR = r'C:\Users\cinco\Desktop\DATA FOR SCRIPTS\Paths\12.31'
        self.MODEL_PATH = f'{self.BEST_PATH_DIR}\\best_model.pth'
        self.RESULTS_DIR = r'C:\Users\cinco\Desktop\DATA FOR SCRIPTS\resultss'
        
        # Create directories if they don't exist
        Path(self.BEST_PATH_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        
        # Rest of the config remains the same
        self.CUTOFF_TIME = '2024-09-16T08:00:04.248723179Z'
        
        self.INPUT_SIZE = 9
        self.HIDDEN_SIZE = 256
        self.NUM_LAYERS = 2
        self.OUTPUT_SIZE = 1
        self.SEQUENCE_LENGTH = 30
        self.CHUNK_SIZE = 100000
        
        self.PIN_MEMORY = True
        self.PREFETCH_FACTOR = 2
        self.BATCH_SIZE = 4096  
        self.CUDA_ALLOC_CONF = 'max_split_size_mb:128,expandable_segments:True'
        self.NUM_WORKERS = max(multiprocessing.cpu_count() - 1, 1)
        
        self.DTYPE_DICT = {
            'n_delta': 'float64',
            'log_normalized_returns': 'float64',
            'returns_squared_log_normalized': 'float64',
            'rolling_vol_5_log_normalized': 'float64',
            'rolling_vol_15_log_normalized': 'float64',
            'rolling_vol_30_log_normalized': 'float64',
            'rolling_mean_5_log_normalized': 'float64',
            'rolling_mean_15_log_normalized': 'float64',
            'rolling_mean_30_log_normalized': 'float64'
        }


    def verify_paths(self):
        """Verify all necessary paths exist"""
        paths = {
            'DATA_DIR': Path(self.DATA_DIR),
            'BEST_PATH_DIR': Path(self.BEST_PATH_DIR)
        }

        # Create best path directory if it doesn't exist
        Path(self.BEST_PATH_DIR).mkdir(parents=True, exist_ok=True)

        for name, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{name} not found: {path}")
            print(f"✓ {name} verified: {path}")

        # Verify CSV files exist
        csv_files = list(paths['DATA_DIR'].glob('*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.DATA_DIR}")
        print(f"✓ Found {len(csv_files)} CSV files in data directory")

        return True




def process_batch_gpu(batch_df: pd.DataFrame) -> Tuple[torch.Tensor, pd.Series]:
    try:
        with torch.cuda.stream(torch.cuda.Stream()):
            timestamps = batch_df['ts_event']
            numeric_data = batch_df.select_dtypes(include=[np.number])
            
            # Convert to tensor without pinned memory first
            cpu_tensor = torch.from_numpy(numeric_data.values).float()
            
            # Then transfer to GPU
            gpu_tensor = cpu_tensor.cuda()
            
            torch.cuda.synchronize()
            return gpu_tensor, timestamps
    except Exception as e:
        print(f"Error processing batch on GPU: {str(e)}")
        return None, None
# Increase workers based on CPU cores
import multiprocessing



def process_csv_file(csv_file, config, device):
    chunks = []
    df_chunks = pd.read_csv(
        csv_file,
        dtype={
            'n_delta': 'float64',
            'log_normalized_returns': 'float64',
            'returns_squared_log_normalized': 'float64',
            'rolling_vol_5_log_normalized': 'float64',
            'rolling_vol_15_log_normalized': 'float64',
            'rolling_vol_30_log_normalized': 'float64',
            'rolling_mean_5_log_normalized': 'float64',
            'rolling_mean_15_log_normalized': 'float64',
            'rolling_mean_30_log_normalized': 'float64'
        },
        parse_dates=['ts_event'],
        chunksize=1000000,
        low_memory=False,
        engine='c'
    )
    
    feature_columns = [
        'n_delta',
        'log_normalized_returns',
        'returns_squared_log_normalized',
        'rolling_vol_5_log_normalized',
        'rolling_vol_15_log_normalized',
        'rolling_vol_30_log_normalized',
        'rolling_mean_5_log_normalized',
        'rolling_mean_15_log_normalized',
        'rolling_mean_30_log_normalized'
    ]
    
    for chunk in df_chunks:
        if len(chunk) > 0:
            numeric_data = chunk[feature_columns].values
            
            with torch.cuda.stream(torch.cuda.Stream()):
                cpu_tensor = torch.from_numpy(numeric_data).pin_memory()
                gpu_tensor = cpu_tensor.to(device, non_blocking=True)
                
                mean = gpu_tensor.mean(dim=0)
                std = gpu_tensor.std(dim=0)
                gpu_tensor = (gpu_tensor - mean) / std
                
                processed_data = gpu_tensor.cpu().numpy()
                processed_chunk = pd.DataFrame(processed_data, columns=feature_columns)
                processed_chunk['ts_event'] = chunk['ts_event']
                chunks.append(processed_chunk)
                
                del gpu_tensor, cpu_tensor
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    
    return pd.concat(chunks, ignore_index=True)

def load_data_from_csvs(config: Config) -> pd.DataFrame:
    print("\nStarting GPU-accelerated data loading...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

    NUM_WORKERS = MAX_WORKERS = max(multiprocessing.cpu_count() - 1, 1)  # Leave one core free

    Load_batch_size = 128
    
    # Changed from config.log_DIR to config.DATA_DIR
    csv_files = sorted(Path(config.DATA_DIR).glob('*.csv'))
    print(f"Found {len(csv_files)} CSV files to process")
    print(f"Using {NUM_WORKERS} workers optimized for i9-12900K")

    # Rest of the function remains the same...

    processed_dfs = []
    
    try:
        process_file = partial(process_csv_file, config=config, device=device)
        
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for i in range(0, len(csv_files), Load_batch_size):
                batch_files = csv_files[i:i + Load_batch_size]
                results = list(tqdm(
                    executor.map(process_file, batch_files), 
                    total=len(batch_files),
                    desc=f"Processing batch {i//Load_batch_size + 1}/{(len(csv_files)-1)//Load_batch_size + 1}"
                ))
                
                processed_dfs.extend(results)
                if len(processed_dfs) > 5:
                    processed_dfs = [pd.concat(processed_dfs, ignore_index=True)]
                    gc.collect()
                
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        torch.cuda.empty_cache()
        raise e
    finally:
        torch.cuda.empty_cache()
        gc.collect()

    print("\nCombining processed data...")
    final_df = pd.concat(processed_dfs, ignore_index=True)
    final_df = final_df.sort_values('ts_event')
    
    print(f"\nFinal DataFrame shape: {final_df.shape}")
    if device.type == 'cuda':
        print(f"\nGPU Memory Used: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    return final_df
# Add this function to monitor GPU usage
def print_gpu_utilization():
    if torch.cuda.is_available():
        print("\nCurrent GPU Utilization:")
        print(f"Memory Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        print(f"Memory Cached: {torch.cuda.memory_reserved()/1024**3:.2f}GB")


class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(
            config.INPUT_SIZE,
            config.HIDDEN_SIZE, 
            config.NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dtype=torch.float32  # Change to float32 initially
        )
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(config.HIDDEN_SIZE * 2, config.HIDDEN_SIZE * 2, dtype=torch.float32)
        self.fc2 = nn.Linear(config.HIDDEN_SIZE * 2, 128, dtype=torch.float32)
        self.fc3 = nn.Linear(128, config.OUTPUT_SIZE * 5, dtype=torch.float32)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Handle input conversion consistently
        x = x.to(torch.float32)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1]
        out = self.dropout(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        return self.fc3(out)


def prepare_training_sequences(df, config, sequence_length=30, prediction_length=5):
    print("Preparing sequences...")
    
    feature_columns = [
        'n_delta',
        'log_normalized_returns',  # This will be our target column instead of 'normalized_returns'
        'returns_squared_log_normalized',
        'rolling_vol_5_log_normalized',
        'rolling_vol_15_log_normalized',
        'rolling_vol_30_log_normalized',
        'rolling_mean_5_log_normalized',
        'rolling_mean_15_log_normalized',
        'rolling_mean_30_log_normalized'
    ]
    
    features = df[feature_columns].astype(np.float32)
    features = features.fillna(0)

    total_sequences = len(features) - sequence_length - prediction_length
    train_size = int(total_sequences * 0.8)
    train_size = (train_size // config.BATCH_SIZE) * config.BATCH_SIZE
    val_size = ((total_sequences - train_size) // config.BATCH_SIZE) * config.BATCH_SIZE

    return (create_sequence_generator(0, train_size, features, True, config, sequence_length, prediction_length), train_size), \
           (create_sequence_generator(train_size, train_size + val_size, features, False, config, sequence_length, prediction_length), val_size)

def create_sequence_generator(start_idx, end_idx, features, is_training=True, config=None, sequence_length=30, prediction_length=5):
    def generator():
        adjusted_end_idx = start_idx + ((end_idx - start_idx) // config.BATCH_SIZE) * config.BATCH_SIZE
        total_batches = (adjusted_end_idx - start_idx) // config.BATCH_SIZE
        
        chunk_size = config.CHUNK_SIZE
        desc = "Training sequences" if is_training else "Validation sequences"
        
        with tqdm(total=total_batches, desc=desc) as pbar:
            for batch_start in range(start_idx, adjusted_end_idx, chunk_size):
                chunk_end = min(batch_start + chunk_size, adjusted_end_idx)
                
                # Get all features for the chunk
                chunk_features = features.iloc[batch_start:chunk_end + sequence_length].values
                
                # Get target values (log_normalized_returns) for prediction
                chunk_targets = features['log_normalized_returns'].iloc[  # Changed from normalized_returns
                    batch_start + sequence_length:chunk_end + sequence_length + prediction_length
                ].values
                
                for i in range(0, chunk_end - batch_start, config.BATCH_SIZE):
                    if i + config.BATCH_SIZE > chunk_end - batch_start:
                        break
                    
                    # Create sequences for all features
                    batch_features = chunk_features[i:i + config.BATCH_SIZE + sequence_length]
                    X_batch = torch.tensor(
                        np.stack([
                            batch_features[j:j + sequence_length] 
                            for j in range(config.BATCH_SIZE)
                        ]),
                        dtype=torch.float32,
                        device='cuda'
                    )
                    
                    # Create target sequences
                    batch_targets_reshaped = np.array([
                        chunk_targets[i+j:i+j+prediction_length] 
                        for j in range(config.BATCH_SIZE)
                    ], dtype=np.float32)
                    y_batch = torch.tensor(batch_targets_reshaped, dtype=torch.float32, device='cuda')
                    
                    yield X_batch, y_batch
                    pbar.update(1)
                    
                    if i % (chunk_size//4) == 0:
                        torch.cuda.empty_cache()
                        
    return generator
def weighted_mse_loss(pred, target):
    weights = torch.linspace(1.0, 0.5, pred.shape[1], device='cuda')  # Decreasing weights for future predictions
    squared_diff = (pred - target) ** 2
    weighted_diff = squared_diff * weights
    return weighted_diff.mean()
def predict_future(model, sequence):
    model.eval()
    with torch.no_grad():
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).cuda()
        predictions = model(sequence_tensor)
        return predictions.cpu().numpy()
    
def calculate_safe_batch_size(total_gpu_memory, memory_threshold=0.95):  # Increase from 0.9 to 0.95
    available_memory = total_gpu_memory * memory_threshold
    memory_per_sample = 4 * (30 * 2 + 5)
    return int((available_memory * 1024**3) // memory_per_sample)


def train_model_with_dynamic_memory(train_data, val_data, model, config, train_size, val_size):
    device = torch.device('cuda')
    model = model.to(device)
    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Calculate optimal batch size based on available GPU memory
    safe_batch_size = calculate_safe_batch_size(total_gpu_memory)
    config.BATCH_SIZE = min(config.BATCH_SIZE, safe_batch_size)
    
    # Initialize optimizer with larger learning rate and gradient clipping
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)

    # Use CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=1,  # Number of epochs
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Initialize criterion in FP32
    criterion = nn.MSELoss().to(device)
    best_val_loss = float('inf')
    
    # Gradient accumulation steps
    accumulation_steps = 8
    
    epochs_pbar = tqdm(range(1), desc="Training epochs", unit="epoch")
    for epoch in epochs_pbar:
        model.train()
        train_loss = 0
        batch_count = 0
        optimizer.zero_grad(set_to_none=True)
        
        train_generator = train_data[0]()
        train_pbar = tqdm(total=train_size//config.BATCH_SIZE, 
                         desc=f"Training batches (Epoch {epoch+1})", 
                         leave=False)
        
        # Training loop with gradient accumulation
        for batch_x, batch_y in train_generator:
            # Ensure inputs are FP32
            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device, dtype=torch.float32)
            
            # Forward pass
            pred = model(batch_x)
            loss = criterion(pred, batch_y) / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            if (batch_count + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.item() * accumulation_steps
            batch_count += 1
            
            train_pbar.update(1)
            train_pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.6f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
            
            # Memory management
            del batch_x, batch_y, pred, loss
            if batch_count % 50 == 0:
                torch.cuda.synchronize()
                if torch.cuda.memory_allocated()/torch.cuda.max_memory_allocated() > 0.95:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        train_pbar.close()
        
        # Call scheduler.step() once per epoch
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batch_count = 0
        
        val_generator = val_data[0]()
        val_pbar = tqdm(total=val_size//config.BATCH_SIZE, 
                       desc=f"Validation batches (Epoch {epoch+1})", 
                       leave=False)
        
        with torch.no_grad():
            for batch_x, batch_y in val_generator:
                # Ensure inputs are FP32
                batch_x = batch_x.to(device, dtype=torch.float32)
                batch_y = batch_y.to(device, dtype=torch.float32)
                
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
                val_batch_count += 1
                
                val_pbar.update(1)
                val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
                
                del batch_x, batch_y, pred, loss
                if val_batch_count % 50 == 0:
                    torch.cuda.synchronize()
                    if torch.cuda.memory_allocated()/torch.cuda.max_memory_allocated() > 0.85:
                        torch.cuda.empty_cache()
                        gc.collect()
        
        val_pbar.close()
        
        # Calculate and display metrics
        avg_train_loss = train_loss/batch_count if batch_count > 0 else float('inf')
        avg_val_loss = val_loss/val_batch_count if val_batch_count > 0 else float('inf')
        
        epochs_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.6f}',
            'val_loss': f'{avg_val_loss:.6f}'
        })
        print(f"GPU Memory Usage: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss,
            }, config.MODEL_PATH)
            
        # Print memory stats every epoch
        print(f"\nEpoch {epoch+1} Stats:")
        print(f"Max GPU Memory Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
        print(f"Current GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
        print(f"Training size: {train_size}")
        print(f"Batch size: {config.BATCH_SIZE}")



if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    try:
        setup_training_environment()
        config = Config()
        
        print("\nLoading data...")
        df = load_data_from_csvs(config)
        
        print("\nPreparing training sequences...")
        train_data, val_data = prepare_training_sequences(df, config)  # Removed cutoff_time parameter
        
        print("\nInitializing model...")
        model = LSTMModel(config).cuda()
        
        print("\nStarting training...")
        train_model_with_dynamic_memory(
            train_data=train_data,
            val_data=val_data,
            model=model,
            config=config,
            train_size=train_data[1],
            val_size=val_data[1]
        )
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()