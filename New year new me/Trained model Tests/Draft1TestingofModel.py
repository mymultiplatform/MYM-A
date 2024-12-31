import torch
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import multiprocessing

@dataclass
class Config:
    def __init__(self):
        self.BASE_DIR = r'C:\Users\cinco\Desktop\DATA FOR SCRIPTS'
        self.DATA_DIR = f'{self.BASE_DIR}\log'
        self.MODEL_PATH = r'C:\Users\cinco\Desktop\DATA FOR SCRIPTS\Paths\12.24\best_model.pth'
        self.TEST_START = pd.to_datetime('2024-09-16T08:00:04.248723179Z').tz_convert('UTC')        
        # Model configurations
        self.INPUT_SIZE = 2
        self.HIDDEN_SIZE = 256  # Added this from original LSTM config
        self.NUM_LAYERS = 2
        self.OUTPUT_SIZE = 1
        self.SEQUENCE_LENGTH = 30
        self.CHUNK_SIZE = 100000
        
        # Performance optimizations for your hardware
        self.PIN_MEMORY = True
        self.PREFETCH_FACTOR = 2  # Increased from 4
        self.BATCH_SIZE = 8192   # Doubled for 12GB VRAM
        self.CUDA_ALLOC_CONF = 'max_split_size_mb:128,expandable_segments:True'  # Increased from 1024
        self.NUM_WORKERS = MAX_WORKERS = max(multiprocessing.cpu_count() - 1, 1)  # Leave one core free
        self.DTYPE_DICT = {
            'normalized_returns': 'float32',
            'n_delta': 'float32'
        }

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=2, hidden_size=512, num_layers=6):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, 
                                batch_first=True, bidirectional=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc1 = torch.nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc2 = torch.nn.Linear(hidden_size * 2, 128)
        self.fc3 = torch.nn.Linear(128, 5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1]
        out = self.dropout(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        return self.fc3(out)

def load_test_data(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Process multiple files in parallel
    dfs = []
    files = sorted(Path(config.DATA_DIR).glob('*.csv'))
    
    # Increase chunk size
    chunksize = 5000000  # 5M rows per chunk
    
    # Use parallel processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        for f in tqdm(files, desc="Processing CSV files"):
            df = pd.read_csv(
                f,
                dtype=config.DTYPE_DICT,
                parse_dates=['ts_event'],
                usecols=['ts_event', 'normalized_returns', 'n_delta'],
                low_memory=False,
                engine='c'
            )
            dfs.append(df)

    final_data = pd.concat(dfs, ignore_index=True)
    final_data['ts_event'] = pd.to_datetime(final_data['ts_event']).dt.tz_localize('UTC')
    filtered_data = final_data[final_data['ts_event'] >= config.TEST_START]
    
    return filtered_data.sort_values('ts_event')
def prepare_sequence(data, sequence_length=30):
    features = data[['normalized_returns', 'n_delta']].values
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

def predict(model, sequences):
    model.eval()
    with torch.no_grad():
        return model(sequences).cpu().numpy()
# Add this function before main()
def plot_results(df):
    plt.figure(figsize=(15, 7))
    
    # Convert timestamp to datetime if it's not already
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Plot both lines
    plt.plot(df['Timestamp'], df['Actual'], label='Actual Returns', color='blue', alpha=0.7)
    plt.plot(df['Timestamp'], df['Predicted'], label='Predicted Returns', color='red', alpha=0.7)
    
    # Customize the plot
    plt.title('Actual vs Predicted Returns Over Time')
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.legend()
    
    # Rotate and format the x-axis dates
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(r'C:\Users\cinco\Desktop\DATA FOR SCRIPTS\returns_comparison_12_24_24_2.png')
    plt.close()
def main():
   config = Config()
   results = []
   batch_size = 32
   
   model = LSTMModel(
       input_size=config.INPUT_SIZE,
       hidden_size=config.HIDDEN_SIZE,
       num_layers=config.NUM_LAYERS
   )
   checkpoint = torch.load(config.MODEL_PATH, weights_only=True)    
   model.load_state_dict(checkpoint['model_state_dict'])
   model.cuda()
   
   test_data = load_test_data(config)
   
   for i in tqdm(range(0, len(test_data) - config.SEQUENCE_LENGTH - 5, batch_size * 5), desc="Making predictions"):
       sequences = []
       actuals_list = []
       timestamps = []
       
       for j in range(i, min(i + batch_size * 5, len(test_data) - config.SEQUENCE_LENGTH - 5), 5):
           window = test_data.iloc[j:j+config.SEQUENCE_LENGTH]
           sequences.append(prepare_sequence(window))
           actuals_list.append(test_data.iloc[j+config.SEQUENCE_LENGTH:j+config.SEQUENCE_LENGTH+5]['normalized_returns'].values)
           timestamps.append(window['ts_event'].iloc[-1])
       
       batch_sequences = torch.cat(sequences).cuda()
       batch_predictions = predict(model, batch_sequences)
       
       for timestamp, preds, acts in zip(timestamps, batch_predictions, actuals_list):
           for pred, actual in zip(preds, acts):
               results.append({
                   'Timestamp': timestamp,
                   'Predicted': pred,
                   'Actual': actual,
                   'Difference': abs(pred-actual)
               })
       
       if i % (batch_size * 100) == 0:
           torch.cuda.empty_cache()

   df = pd.DataFrame(results)
   df.to_csv(r'C:\Users\cinco\Desktop\DATA FOR SCRIPTS\prediction_results_12_24_24_2.csv', index=False)
   plot_results(df)
if __name__ == "__main__":
   main()