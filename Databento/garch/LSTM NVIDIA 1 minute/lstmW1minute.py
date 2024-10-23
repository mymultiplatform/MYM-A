import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import glob
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
import gc  # For garbage collection
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
warnings.filterwarnings('ignore')

# Previous TimeSeriesDataset and PricePredictionLSTM classes remain the same
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, pred_length, data_scaler, chunk_size=1000000):
        self.data = data
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.chunk_size = chunk_size
        self.data_scaler = data_scaler
        self.total_sequences = len(data) - seq_length - pred_length + 1
        self.valid_indices = np.arange(self.total_sequences, dtype=np.int32)
    
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.seq_length]
        target = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_length, 0]
        return torch.FloatTensor(sequence), torch.FloatTensor(target)

class PricePredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(PricePredictionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.attention = nn.Linear(hidden_size, 1)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, output_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        lstm_out = lstm_out * attention_weights
        out = lstm_out[:, -1, :]
        
        out = self.activation(self.fc1(out))
        out = self.dropout1(out)
        out = self.activation(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc3(out)
        
        return out

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
    
def add_technical_features(df):
    """Add technical indicators as features."""
    df = df.astype('float32')
    
    df['returns'] = df['price'].pct_change()
    
    windows = [5, 15, 30, 60]
    for window in windows:
        df[f'sma_{window}'] = df['price'].rolling(window=window).mean()
        df[f'std_{window}'] = df['price'].rolling(window=window).std()
    
    df['hour'] = df.index.hour.astype('float32')
    df['minute'] = df.index.minute.astype('float32')
    df['day_of_week'] = df.index.dayofweek.astype('float32')
    
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=50):
    """Train the model with early stopping."""
    best_val_loss = float('inf')
    no_improve = 0
    
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
def format_price(x, p):
    """Format price values for the y-axis"""
    return f'${x:,.2f}'

def create_price_plot(comparison_df, last_timestamp, save_path='price_prediction_plot.png'):
    """Create and save a plot of actual vs predicted prices"""
    plt.figure(figsize=(15, 8))
    
    # Convert timestamp to datetime if it's not already
    comparison_df['timestamp'] = pd.to_datetime(comparison_df['timestamp'])
    
    # Plot actual prices
    actual_mask = comparison_df['type'] == 'actual'
    if any(actual_mask):
        plt.plot(comparison_df[actual_mask]['timestamp'], 
                comparison_df[actual_mask]['price'],
                label='Actual Price', 
                color='blue',
                linewidth=2)
    
    # Plot predicted prices
    predicted_mask = comparison_df['type'] == 'predicted'
    if any(predicted_mask):
        plt.plot(comparison_df[predicted_mask]['timestamp'], 
                comparison_df[predicted_mask]['price'],
                label='Predicted Price', 
                color='red',
                linewidth=2,
                linestyle='--')
    
    # Add vertical line at cutoff point
    plt.axvline(x=last_timestamp, color='gray', linestyle=':', label='Prediction Start')
    
    # Customize the plot
    plt.title('Price Prediction Analysis', fontsize=16, pad=20)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()  # Rotate and align x-axis labels
    
    # Format y-axis
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_price))
    
    # Add padding to the layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

# Add configuration parameters at the top of the script

    # Data parameters
class Config:
    TRAIN_START_DATE = "2018-05-02T08:44:39.292071841Z"
    TRAIN_END_DATE = "2018-05-18T09:27:27.555026009Z"
    TEST_START_DATE = "2018-05-18T09:41:59.385291233Z"
    TEST_END_DATE = "2018-05-18T23:59:43.279929697Z"
    DATA_DIR = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\data bento data\test"  # Change to your production data directory
    
    # Model parameters
    SEQUENCE_LENGTH = 60
    PREDICTION_LENGTH = 1440
    BATCH_SIZE = 128
    HIDDEN_SIZE = 256
    NUM_LAYERS = 3
    LEARNING_RATE = 0.001
    EPOCHS = 30  # Increased from 30 to 100
    PATIENCE = 50  # Early stopping patience
    
    # Training parameters
    TRAIN_VAL_SPLIT = 0.8
    NUM_WORKERS = 4
    
    # Other parameters
    RANDOM_SEED = 42


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")
    
    # Load and process training data
    train_data = []
    csv_files = glob.glob(str(Path(Config.DATA_DIR) / "*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {Config.DATA_DIR}")
    
    print(f"Found {len(csv_files)} CSV files")
    print("Processing training data...")
    
    for file in tqdm(csv_files):
        df = process_csv_file(file, Config.TRAIN_START_DATE, Config.TRAIN_END_DATE)
        if not df.empty:
            train_data.append(df)
    
    if not train_data:
        raise ValueError(
            f"No training data found between {Config.TRAIN_START_DATE} and {Config.TRAIN_END_DATE}\n"
            f"Please check your date ranges and data directory: {Config.DATA_DIR}"
        )
    
    train_df = pd.concat(train_data)
    train_df = train_df.sort_index()
    
    # Load and process test data
    test_data = []
    print("Processing test data...")
    
    for file in tqdm(csv_files):
        df = process_csv_file(file, Config.TEST_START_DATE, Config.TEST_END_DATE)
        if not df.empty:
            test_data.append(df)
    
    if not test_data:
        raise ValueError(
            f"No test data found between {Config.TEST_START_DATE} and {Config.TEST_END_DATE}\n"
            f"Please check your date ranges and data directory: {Config.DATA_DIR}"
        )
    
    test_df = pd.concat(test_data)
    test_df = test_df.sort_index()
    
    # Print data information
    print("\nData Summary:")
    print(f"Training data shape: {train_df.shape}")
    print(f"Training date range: {train_df.index.min()} to {train_df.index.max()}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Test date range: {test_df.index.min()} to {test_df.index.max()}")
    
    # Combine data for feature engineering
    full_df = pd.concat([train_df, test_df])
    full_df = full_df.sort_index()
    
    print("\nFull dataset shape:", full_df.shape)
    print(f"Full date range: {full_df.index.min()} to {full_df.index.max()}")
    
    print("Resampling to minute intervals...")
    full_df = full_df.resample('1T').last().fillna(method='ffill')
    
    actual_data = pd.DataFrame({
        'timestamp': full_df.index,
        'actual_price': full_df['price']
    })
    
    print("Adding technical features...")
    full_df = add_technical_features(full_df)
    
    print("Scaling data...")
    data_scaler = MinMaxScaler()
    full_df = full_df.astype('float32')
    scaled_data = data_scaler.fit_transform(full_df.values)
    
    # Split data based on dates
    train_mask = (full_df.index >= Config.TRAIN_START_DATE) & (full_df.index <= Config.TRAIN_END_DATE)
    train_data = scaled_data[train_mask]
    test_data = scaled_data[~train_mask]
    
    # Create datasets
    train_size = int(Config.TRAIN_VAL_SPLIT * len(train_data))
    train_dataset = TimeSeriesDataset(train_data[:train_size], 
                                    Config.SEQUENCE_LENGTH, 
                                    Config.PREDICTION_LENGTH, 
                                    data_scaler)
    val_dataset = TimeSeriesDataset(train_data[train_size:], 
                                  Config.SEQUENCE_LENGTH, 
                                  Config.PREDICTION_LENGTH, 
                                  data_scaler)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Initialize model
    input_size = scaled_data.shape[1]
    model = PricePredictionLSTM(
        input_size=input_size,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        output_size=Config.PREDICTION_LENGTH
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    print(f"Starting training for {Config.EPOCHS} epochs...")
    train_model(model, train_loader, val_loader, criterion, optimizer, 
                Config.EPOCHS, device, Config.PATIENCE)
    
# [Previous code remains the same until the prediction generation part]

    print("Generating predictions...")
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    with torch.no_grad():
        last_sequence = scaled_data[-Config.SEQUENCE_LENGTH:].reshape(1, Config.SEQUENCE_LENGTH, -1)
        last_sequence = torch.FloatTensor(last_sequence).to(device)
        predictions = model(last_sequence)
        predictions = predictions.cpu().numpy()
    
    # Process predictions
    predictions_reshaped = np.zeros((len(predictions[0]), scaled_data.shape[1]))
    predictions_reshaped[:, 0] = predictions[0]
    predicted_prices = data_scaler.inverse_transform(predictions_reshaped)[:, 0]
    
    # Create timestamps for predictions
    last_timestamp = pd.Timestamp(Config.TEST_END_DATE)
    pred_index = pd.date_range(
        start=last_timestamp, 
        periods=Config.PREDICTION_LENGTH + 1, 
        freq='1T'
    )[1:]
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'timestamp': pred_index,
        'price': predicted_prices
    })
    
    # Get the actual prices for comparison
    last_actual_prices = actual_data[
        (actual_data['timestamp'] <= last_timestamp) & 
        (actual_data['timestamp'] > last_timestamp - pd.Timedelta(minutes=Config.PREDICTION_LENGTH))
    ].copy()
    
    last_actual_prices = last_actual_prices.rename(columns={'actual_price': 'price'})
    
    # Create comparison DataFrame by concatenating along the index
    comparison_df = pd.concat(
        [last_actual_prices, predictions_df],
        axis=0,
        ignore_index=True
    ).sort_values('timestamp').reset_index(drop=True)
    
    # Add a type column to distinguish between actual and predicted values
    comparison_df['type'] = 'predicted'
    comparison_df.loc[comparison_df['timestamp'] <= last_timestamp, 'type'] = 'actual'
    
    # Create separate DataFrames for saving
    predictions_to_save = predictions_df.copy()
    predictions_to_save = predictions_to_save.rename(columns={'price': 'predicted_price'})
    predictions_to_save.to_csv('price_predictions.csv', index=False)
    
    comparison_to_save = comparison_df.pivot(
        index='timestamp',
        columns='type',
        values='price'
    ).reset_index()
    comparison_to_save.to_csv('price_comparison.csv', index=False)
    
    print("Predictions saved to price_predictions.csv")
    print("Comparison data saved to price_comparison.csv")
    
    # Create and save the plot
    plot_path = create_price_plot(comparison_df, last_timestamp)
    print(f"Price prediction plot saved to {plot_path}")
    
    # Calculate statistics
    actual_mask = comparison_df['type'] == 'actual'
    predicted_mask = comparison_df['type'] == 'predicted'
    
    # Find overlapping timestamps
    overlap_timestamps = set(comparison_df[actual_mask]['timestamp']) & set(comparison_df[predicted_mask]['timestamp'])
    
    if overlap_timestamps:
        overlap_df = comparison_df[comparison_df['timestamp'].isin(overlap_timestamps)].copy()
        actual_values = overlap_df[actual_mask]['price'].values
        predicted_values = overlap_df[predicted_mask]['price'].values
        
        # Calculate metrics
        mse = np.mean((actual_values - predicted_values) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual_values - predicted_values))
        mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
        
        print("\nPrediction Statistics for Overlap Period:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Mean Absolute Percentage Error: {mape:.2f}%")
        
        # Additional analysis
        print("\nPrice Range Analysis:")
        print(f"Actual Price Range: {np.min(actual_values):.2f} to {np.max(actual_values):.2f}")
        print(f"Predicted Price Range: {np.min(predicted_values):.2f} to {np.max(predicted_values):.2f}")
        
        # Calculate correlation if there are enough points
        if len(actual_values) > 1:
            correlation = np.corrcoef(actual_values, predicted_values)[0, 1]
            print(f"\nCorrelation between actual and predicted prices: {correlation:.4f}")
        
        # Save detailed statistics
        stats_df = pd.DataFrame({
            'timestamp': list(overlap_timestamps),
            'actual_price': actual_values,
            'predicted_price': predicted_values,
            'absolute_error': np.abs(actual_values - predicted_values),
            'percentage_error': np.abs((actual_values - predicted_values) / actual_values) * 100
        })
        
        stats_df.to_csv('prediction_statistics.csv', index=False)
        print("\nDetailed statistics saved to prediction_statistics.csv")
    else:
        print("\nNo overlap period found between actual and predicted prices.")
        print("This might occur if the prediction period starts after all actual data points.")
        print(f"Last actual timestamp: {last_actual_prices['timestamp'].max()}")
        print(f"First prediction timestamp: {predictions_df['timestamp'].min()}")
    
    return comparison_df

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()