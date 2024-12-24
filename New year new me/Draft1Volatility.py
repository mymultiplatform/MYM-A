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

# Call before training
setup_training_environment()

@dataclass
class Config:
    def __init__(self):
        # Local path configurations
        self.BASE_DIR = r'C:\Users\cinco\Desktop\DATA FOR SCRIPTS'
        self.DATA_DIR = f'{self.BASE_DIR}\log'
        self.BEST_PATH_DIR = r'C:\Users\cinco\Desktop\DATA FOR SCRIPTS\Paths\12.23'
        self.MODEL_PATH = f'{self.BEST_PATH_DIR}\\best_model.pth'
        self.log_DIR = r'C:\Users\cinco\Desktop\DATA FOR SCRIPTS\log'

        # Data processing configurations
        self.CUTOFF_TIME = '2024-09-16T08:00:04.248723179Z'
        self.BUFFER_FACTOR = 0.1


        # Model configurations
        self.INPUT_SIZE = 2
        self.HIDDEN_SIZE = 64
        self.NUM_LAYERS = 2
        self.OUTPUT_SIZE = 1
        # Data processing configurations
        self.SEQUENCE_LENGTH = 30
        self.CHUNK_SIZE = 100000

        # Model configurations
        self.PIN_MEMORY = True
        self.PREFETCH_FACTOR = 4
        # GPU configurations
        self.CUDA_ALLOC_CONF = 'max_split_size_mb:1024,expandable_segments:True'
        # In Config class
        self.BATCH_SIZE = 32768

        # DataFrame dtype configurations
        self.DTYPE_DICT = {
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
            'log_returns': 'float64',
            'normalized_returns': 'float64'

        }

        # Processing configurations
        self.N_EVENTS_ANALYSIS = 5

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

    def setup_gpu(self):
        """Configure GPU settings"""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = self.CUDA_ALLOC_CONF
            torch.cuda.empty_cache()
            print("✓ GPU setup complete - CUDA is available")
            return True
        print("✗ GPU not available - using CPU")
        return False

# Initialize config with drive mount
try:
    print("Setting up configuration...")
    config = Config()
    config.verify_paths()
    config.setup_gpu()
    print("\nConfiguration setup complete!")
except Exception as e:
    print(f"Configuration setup failed: {str(e)}")

def setup_gpu_environment():
    """Configure GPU settings for optimal performance."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
        torch.cuda.empty_cache()
        print("GPU setup complete. CUDA is available.")
        return True
    print("GPU not available. Using CPU.")
    return False

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
MAX_WORKERS = max(multiprocessing.cpu_count() - 1, 1)  # Leave one core free



"""# Load the data"""
def process_csv_file(csv_file, config, device):
    chunks = []
    df_chunks = pd.read_csv(
        csv_file,
        dtype=config.DTYPE_DICT,
        parse_dates=['ts_recv', 'ts_event'],
        chunksize=1000000,  # 1 million rows per chunk
        low_memory=False,
        engine='c'
    )
    
    for chunk in df_chunks:
        if len(chunk) > 0:
            numeric_cols = chunk.select_dtypes(include=[np.number]).columns
            numeric_data = chunk[numeric_cols].values
            
            with torch.cuda.stream(torch.cuda.Stream()):
                cpu_tensor = torch.from_numpy(numeric_data).pin_memory()
                gpu_tensor = cpu_tensor.to(device, non_blocking=True)
                
                mean = gpu_tensor.mean(dim=0)
                std = gpu_tensor.std(dim=0)
                gpu_tensor = (gpu_tensor - mean) / std
                
                processed_data = gpu_tensor.cpu().numpy()
                
                processed_chunk = pd.DataFrame(processed_data, columns=numeric_cols)
                
                non_numeric_cols = chunk.select_dtypes(exclude=[np.number]).columns
                for col in non_numeric_cols:
                    processed_chunk[col] = chunk[col].values
                
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

    NUM_WORKERS = 20
    Load_batch_size = 20
    
    csv_files = sorted(Path(config.log_DIR).glob('*.csv'))
    print(f"Found {len(csv_files)} CSV files to process")
    print(f"Using {NUM_WORKERS} workers optimized for i9-12900K")

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
"""df.head()

# LSTM config
"""

class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=512, num_layers=6):  # Increased from 256 and 4
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dtype=torch.float32
        )
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 2, dtype=torch.float32)
        self.fc2 = nn.Linear(hidden_size * 2, 128, dtype=torch.float32)  # Increased from 64
        self.fc3 = nn.Linear(128, 5, dtype=torch.float32)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Ensure input is in the correct format and on GPU
        if not x.is_cuda:
            x = x.cuda()
        x = x.to(torch.float32)
        
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1]
        out = self.dropout(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        return self.fc3(out)

def prepare_training_sequences(df, cutoff_time, config, sequence_length=30, prediction_length=5):
    """
    Prepare sequences for training using only normalized_returns and n_delta
    """
    print("Preparing sequences...")
    
    # Convert cutoff time
    cutoff_time = pd.to_datetime(cutoff_time).tz_localize(None)
    ts_event = df['ts_event']
    if ts_event.dt.tz is not None:
        ts_event = ts_event.dt.tz_localize(None)

    # Find cutoff index with progress bar
    with tqdm(total=1, desc="Finding cutoff index") as pbar:
        cutoff_idx = ts_event[ts_event > cutoff_time].index[0]
        pbar.update(1)

    # Select only the two features we want
    with tqdm(total=1, desc="Preparing features DataFrame") as pbar:
        features = pd.DataFrame({
            'normalized_returns': df['normalized_returns'].fillna(0),
            'n_delta': df['n_delta']
        })
        pbar.update(1)

    # Calculate total valid sequences and adjust sizes
    total_sequences = len(features) - sequence_length - prediction_length
    train_end = cutoff_idx - sequence_length - prediction_length
    
    # Ensure training size is a multiple of batch size
    train_size = (train_end // config.BATCH_SIZE) * config.BATCH_SIZE
    val_size = ((total_sequences - train_end) // config.BATCH_SIZE) * config.BATCH_SIZE
    
    print(f"\nSequence Generation Stats:")
    print(f"Total sequences available: {total_sequences}")
    print(f"Cutoff index: {cutoff_idx}")
    print(f"Training end index: {train_end}")
    print(f"Adjusted training size: {train_size}")
    print(f"Adjusted validation size: {val_size}")
    print(f"Batch size: {config.BATCH_SIZE}")

    def create_sequence_generator(start_idx, end_idx, is_training=True):
        def generator():
            # Adjust end_idx to be a multiple of batch_size
            adjusted_end_idx = start_idx + ((end_idx - start_idx) // config.BATCH_SIZE) * config.BATCH_SIZE
            total_batches = (adjusted_end_idx - start_idx) // config.BATCH_SIZE
            
            desc = "Generating training sequences" if is_training else "Generating validation sequences"
            with tqdm(total=total_batches, desc=desc, unit="batch") as pbar:
                for batch_start in range(start_idx, adjusted_end_idx, config.BATCH_SIZE):
                    batch_end = min(batch_start + config.BATCH_SIZE, adjusted_end_idx)
                    batch_size_actual = batch_end - batch_start

                    # Skip if batch size is too small
                    if batch_size_actual < config.BATCH_SIZE:
                        continue

                    X_batch = torch.zeros((batch_size_actual, sequence_length, 2),
                                        dtype=torch.float32, device='cuda')
                    y_batch = torch.zeros((batch_size_actual, prediction_length),
                                        dtype=torch.float32, device='cuda')

                    try:
                        # Create input sequences
                        for i in range(sequence_length):
                            sequence_data = features.iloc[batch_start + i:batch_end + i].values
                            X_batch[:, i] = torch.tensor(sequence_data, dtype=torch.float32, device='cuda')

                        # Create target sequences
                        for i in range(prediction_length):
                            target_idx = batch_start + sequence_length + i
                            y_batch[:, i] = torch.tensor(
                                features['normalized_returns'].iloc[target_idx:target_idx + batch_size_actual].values,
                                dtype=torch.float32, device='cuda'
                            )

                        yield X_batch, y_batch
                        pbar.update(1)

                    except Exception as e:
                        print(f"Error in batch generation: {str(e)}")
                        continue

                    finally:
                        # Clean up GPU memory
                        torch.cuda.empty_cache()
        return generator

    # Return generators and sizes
    return (create_sequence_generator(0, train_size, True), train_size), \
           (create_sequence_generator(train_size, train_size + val_size, False), val_size)


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
    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Calculate optimal batch size based on available GPU memory
    safe_batch_size = calculate_safe_batch_size(total_gpu_memory)
    config.BATCH_SIZE = min(config.BATCH_SIZE, safe_batch_size)
    
    # Initialize optimizer with larger learning rate and gradient clipping
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)

    # Use CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,  # Number of epochs
        eta_min=1e-6  # Minimum learning rate
    )
    
    scaler = torch.amp.GradScaler()
    criterion = nn.MSELoss().cuda()
    best_val_loss = float('inf')
    
    # Gradient accumulation steps
    accumulation_steps = 2
    
    epochs_pbar = tqdm(range(100), desc="Training epochs", unit="epoch")
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
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = model(batch_x)
                loss = criterion(pred, batch_y) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_count + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
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
                if torch.cuda.memory_allocated()/torch.cuda.max_memory_allocated() > 0.95:  # Increase from 0.85 to 0.95
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
                with torch.autocast(device_type='cuda', dtype=torch.float16):
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
                'scaler_state_dict': scaler.state_dict(),
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
        print("\nSystem Information:")
        if torch.cuda.is_available():
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
        
        # Configure CUDA for optimal performance
        setup_training_environment()
        
        print("\nLoading data...")
        df = load_data_from_csvs(config)
        print("\nDataFrame Info:")
        print(df.info())
        
        print("\nPreparing training sequences...")
        train_data, val_data = prepare_training_sequences(df, config.CUTOFF_TIME, config)
        
        print("\nInitializing model...")
        model = LSTMModel(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS
        ).cuda()
        
        print("\nStarting training with dynamic memory management...")
        train_model_with_dynamic_memory(
            train_data=train_data,
            val_data=val_data,
            model=model,
            config=config,
            train_size=train_data[1],
            val_size=val_data[1]
        )
        
        # Print final GPU memory usage
        print_gpu_utilization()
        
        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()