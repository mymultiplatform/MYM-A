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
        self.INPUT_SIZE = 8  # Increased for additional features
        self.HIDDEN_SIZE = 256
        self.NUM_LAYERS = 2
        self.OUTPUT_SIZE = 1
        self.SEQUENCE_LENGTH = 30
        self.CHUNK_SIZE = 100000
        self.PIN_MEMORY = True
        self.PREFETCH_FACTOR = 2
        self.BATCH_SIZE = 8192
        self.CUDA_ALLOC_CONF = 'max_split_size_mb:128,expandable_segments:True'
        self.NUM_WORKERS = max(multiprocessing.cpu_count() - 1, 1)
        self.DTYPE_DICT = {
            'normalized_returns': 'float32',
            'n_delta': 'float32'
        }

class ResidualLSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = torch.nn.LSTMCell(input_size, hidden_size)
        self.residual = torch.nn.Linear(input_size, hidden_size) if input_size != hidden_size else torch.nn.Identity()
        
    def forward(self, x, hidden):
        h, c = self.lstm(x, hidden)
        return h + self.residual(x), c

class EnhancedGARCHModel(torch.nn.Module):
    def __init__(self, input_size=8, hidden_size=256, num_layers=2):
        super().__init__()
        self.input_projection = torch.nn.Linear(input_size, hidden_size)
        
        self.lstm_layers = torch.nn.ModuleList([
            ResidualLSTMCell(hidden_size if i == 0 else hidden_size * 2, hidden_size)
            for i in range(num_layers)
        ])
        
        self.garch_layer = torch.nn.Linear(hidden_size, 1)
        self.arch_layer = torch.nn.Linear(hidden_size, 1)
        
        self.mean_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        
        self.variance_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Softplus()
        )
        
        self.volatility_attention = torch.nn.MultiheadAttention(hidden_size, 4)
    
    def forward(self, x, prev_variance=None):  # Note the proper indentation here
        # Add shape validation
        print(f"Input shape before processing: {x.shape}")
        
        # Ensure x is 2D [batch_size, features]
        if len(x.shape) == 3:
            x = x.squeeze(1)
        
        assert len(x.shape) == 2, f"Expected 2D input after processing, got shape {x.shape}"
        print(f"Input shape after processing: {x.shape}")
        
        batch_size = x.size(0)
        h_states = []
        c_states = []
        
        x = self.input_projection(x)
        
        for layer in self.lstm_layers:
            if len(h_states) == 0:
                h = torch.zeros(batch_size, layer.lstm.hidden_size).to(x.device)
                c = torch.zeros(batch_size, layer.lstm.hidden_size).to(x.device)
            else:
                h = h_states[-1]
                c = c_states[-1]
            
            h, c = layer(x, (h, c))
            h_states.append(h)
            c_states.append(c)
            x = torch.cat([x, h], dim=-1)
        
        combined_features = torch.cat(h_states, dim=-1)
        
        # Adjust attention input dimensions
        attn_output, _ = self.volatility_attention(
            combined_features.unsqueeze(0),
            combined_features.unsqueeze(0),
            combined_features.unsqueeze(0)
        )
        attn_output = attn_output.squeeze(0)
        
        mean = self.mean_head(combined_features)
        base_variance = self.variance_head(combined_features)
        garch_component = self.garch_layer(attn_output)
        arch_component = self.arch_layer(attn_output)
        
        if prev_variance is not None:
            variance = (base_variance + 
                       garch_component * prev_variance + 
                       arch_component * torch.square(mean))
        else:
            variance = base_variance
            
        return mean, variance
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
def prepare_enhanced_features(data, window_size=30):
    df = data.copy()
    
    df['returns_squared'] = df['normalized_returns'] ** 2
    df['rolling_vol_5'] = df['normalized_returns'].rolling(5).std()
    df['rolling_vol_15'] = df['normalized_returns'].rolling(15).std()
    df['rolling_vol_30'] = df['normalized_returns'].rolling(30).std()
    df['rolling_mean_5'] = df['normalized_returns'].rolling(5).mean()
    df['rolling_mean_15'] = df['normalized_returns'].rolling(15).mean()
    
    features = df[[
        'normalized_returns',
        'n_delta',
        'returns_squared',
        'rolling_vol_5',
        'rolling_vol_15',
        'rolling_vol_30',
        'rolling_mean_5',
        'rolling_mean_15'
    ]].values
    
    tensor_features = torch.tensor(features, dtype=torch.float32)
    print(f"Feature tensor shape: {tensor_features.shape}")
    validate_tensor_shapes(tensor_features, 2, "features")
    
    return tensor_features

def custom_loss(pred_mean, pred_var, true_returns):
    mse_loss = torch.nn.MSELoss()(pred_mean, true_returns)
    likelihood_loss = 0.5 * (torch.log(pred_var) + 
                            torch.square(true_returns - pred_mean) / pred_var)
    return mse_loss + likelihood_loss.mean()

def rolling_forecast_enhanced(model, data, window_size=30, horizon=5):
    model.eval()
    forecasts = []
    prev_variance = None
    
    with torch.no_grad():
        for i in range(0, len(data) - window_size - horizon + 1):
            window = data[i:i+window_size]
            sequence = prepare_enhanced_features(window, window_size)
            sequence = sequence.cuda()
            
            # Add shape validation
            print(f"Sequence shape before processing: {sequence.shape}")
            validate_tensor_shapes(sequence, 2, "sequence")
            
            # Add batch dimension if needed
            if len(sequence.shape) == 2:
                sequence = sequence.unsqueeze(0)
            
            print(f"Sequence shape after batch dimension: {sequence.shape}")
            
            # Process each timestep in the sequence
            batch_size = sequence.size(0)
            h_states = []
            c_states = []
            
            # Initialize hidden states
            for layer in model.lstm_layers:
                h = torch.zeros(batch_size, layer.lstm.hidden_size).to(sequence.device)
                c = torch.zeros(batch_size, layer.lstm.hidden_size).to(sequence.device)
                h_states.append(h)
                c_states.append(c)
            
            # Process sequence
            for t in range(sequence.size(1)):
                x_t = sequence[:, t, :]
                mean, variance = model(x_t, prev_variance)
                prev_variance = variance
            
            forecasts.append({
                'timestamp': data['ts_event'].iloc[i+window_size],
                'predicted_mean': mean.cpu().numpy()[0][0],
                'predicted_variance': variance.cpu().numpy()[0][0]
            })
    
    return pd.DataFrame(forecasts)
def plot_results_with_intervals(df):
    plt.figure(figsize=(15, 7))
    
    # Plot mean prediction
    plt.plot(df['timestamp'], df['predicted_mean'], 
            label='Predicted Mean', color='blue')
    
    # Plot confidence intervals
    plt.fill_between(df['timestamp'], 
                    df['lower_bound'], 
                    df['upper_bound'],
                    color='blue', alpha=0.2, 
                    label='95% Confidence Interval')
    
    plt.title('Returns Forecast with Volatility Intervals')
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.legend()
    
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(r'C:\Users\cinco\Desktop\DATA FOR SCRIPTS\garch_like_forecast.png')
    plt.close()


def validate_tensor_shapes(tensor, expected_dims, name="tensor"):
    if len(tensor.shape) != expected_dims:
        raise ValueError(f"Expected {name} to have {expected_dims} dimensions, got {len(tensor.shape)}")
    


def main():
    config = Config()
    
    model = EnhancedGARCHModel(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS
    )
    
    model.cuda()
    
    test_data = load_test_data(config)
    print(f"Test data shape: {test_data.shape}")  # Add this line
    
    forecasts_df = rolling_forecast_enhanced(
        model=model,
        data=test_data,
        window_size=config.SEQUENCE_LENGTH,
        horizon=5
    )
    
    forecasts_df['upper_bound'] = (forecasts_df['predicted_mean'] + 
                                 1.96 * np.sqrt(forecasts_df['predicted_variance']))
    forecasts_df['lower_bound'] = (forecasts_df['predicted_mean'] - 
                                 1.96 * np.sqrt(forecasts_df['predicted_variance']))
    
    forecasts_df.to_csv(f'{config.BASE_DIR}/enhanced_garch_forecasts.csv', 
                       index=False)
    plot_results_with_intervals(forecasts_df)

if __name__ == "__main__":
    main()