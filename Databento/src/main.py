import torch
import numpy as np
import pandas as pd
from pathlib import Path
import glob
from datetime import datetime, timedelta
import warnings
import gc
from tqdm import tqdm
from config.config import Config
from data.data_loader import process_csv_file
from data.dataset import TimeSeriesDataset
from models.lstm import PricePredictionLSTM
from utils.preprocessing import add_technical_features
from utils.visualization import create_price_plot
from training.trainer import train_model
from sklearn.preprocessing import MinMaxScaler  # For data_scaler
from torch.utils.data import DataLoader        # For DataLoader
import torch.nn as nn                          # For nn.MSELoss()
import torch.optim as optim                    # For optim.Adam()

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
    full_df = full_df.resample('1min').last().ffill()    
    full_df = full_df.sort_index()

    print("\nFull dataset shape:", full_df.shape)
    print(f"Full date range: {full_df.index.min()} to {full_df.index.max()}")
    
    print("Resampling to minute intervals...")
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
        # Optional: Check for overlap
        if not (train_df.index.max() < test_df.index.min()):
            warnings.warn("Warning: Overlap detected between training and test data!")
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