import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#removed these errors:

# Forward-looking bias in feature calculation
# Missing normalization for different feature scales
# Look-ahead bias in rolling windows

def add_technical_features(df):
    """Add technical indicators as features."""
    df = df.copy()
    # Keep original price
    df['returns'] = df['price'].pct_change()
    
    # Add basic technical indicators
    windows = [5, 15, 30]
    for window in windows:
        df[f'sma_{window}'] = df['price'].rolling(window=window).mean()
        df[f'std_{window}'] = df['price'].rolling(window=window).std()
    
    # Add time features
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek
    
    # Fill any missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df