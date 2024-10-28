import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler



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
    
    df = df.ffill().bfill()    
    return df

