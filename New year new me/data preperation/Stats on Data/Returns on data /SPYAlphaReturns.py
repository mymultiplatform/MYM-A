import pandas as pd
import os
from pathlib import Path
import numpy as np
from scipy import stats
from datetime import datetime, time, timedelta

def analyze_microstructure(df):
    # Convert timestamps
    df['ts_event'] = pd.to_datetime(df['ts_event'], unit='ns')
    
    # Calculate basic metrics
    df['price_change'] = df['price'].diff()
    df['time_delta'] = df['ts_event'].diff().dt.total_seconds() * 1_000_000  # microseconds
    
    # Order flow classification
    # Assuming side values: 1 = bid, 2 = ask, other = exchange/neither
    df['bid_volume'] = np.where(df['side'] == 1, df['size'], 0)
    df['ask_volume'] = np.where(df['side'] == 2, df['size'], 0)
    df['exchange_volume'] = np.where((df['side'] != 1) & (df['side'] != 2), df['size'], 0)
    
    # Calculate returns and volatility
    df['returns'] = df['price'].pct_change()
    df['rolling_vol'] = df['returns'].rolling(window=100).std()
    
    results = []
    window_size = 1000  # Analyze patterns every 1000 ticks
    
    for i in range(0, len(df) - window_size, window_size):
        window = df.iloc[i:i+window_size]
        
        # Order flow metrics
        total_bid_volume = window['bid_volume'].sum()
        total_ask_volume = window['ask_volume'].sum()
        total_exchange_volume = window['exchange_volume'].sum()
        
        # Order imbalance
        order_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) if (total_bid_volume + total_ask_volume) > 0 else 0
        
        # Time-weighted metrics
        avg_trade_size = window['size'].mean()
        max_trade_size = window['size'].max()
        
        # Price impact calculation
        price_impact = (window['price_change'].abs() / (window['size'] * window['rolling_vol'])).mean()
        
        # VWAP calculation
        vwap = (window['price'] * window['size']).sum() / window['size'].sum()
        
        results.append({
            'timestamp': window['ts_event'].iloc[0],
            'end_timestamp': window['ts_event'].iloc[-1],
            'order_imbalance': order_imbalance,
            'total_volume': window['size'].sum(),
            'bid_volume': total_bid_volume,
            'ask_volume': total_ask_volume,
            'exchange_volume': total_exchange_volume,
            'avg_trade_size': avg_trade_size,
            'max_trade_size': max_trade_size,
            'avg_time_between_trades': window['time_delta'].mean(),
            'price_volatility': window['price'].std(),
            'trade_count': len(window),
            'price_impact': price_impact,
            'vwap': vwap
        })
    
    return pd.DataFrame(results)

def identify_potential_signals(df):
    signals = pd.DataFrame()
    
    # Volume-based signals
    signals['large_imbalance'] = abs(df['order_imbalance']) > df['order_imbalance'].rolling(10).std() * 2
    
    # Time-based signals
    signals['unusual_timing'] = df['avg_time_between_trades'] > df['avg_time_between_trades'].rolling(10).mean() * 2
    
    # Size-based signals
    signals['large_trades'] = df['max_trade_size'] > df['avg_trade_size'] * 3
    
    # Volume surge signals
    df['volume_ma'] = df['total_volume'].rolling(5).mean()
    signals['volume_surge'] = df['total_volume'] > df['volume_ma'] * 1.5
    
    # Price impact signals
    signals['high_impact'] = df['price_impact'] > df['price_impact'].rolling(10).mean() * 2
    
    # VWAP signals
    df['vwap_diff'] = (df['vwap'] - df['vwap'].shift(1)) / df['vwap'].shift(1)
    signals['vwap_deviation'] = abs(df['vwap_diff']) > df['vwap_diff'].rolling(10).std() * 2
    
    # Combine signals
    df['signal_strength'] = signals.sum(axis=1)
    
    return df

def analyze_cross_sectional(combined_analysis):
    # Rank signals across all windows
    combined_analysis['signal_rank'] = combined_analysis.groupby('timestamp')['signal_strength'].rank(pct=True)
    
    # Identify top quantile signals
    top_signals = combined_analysis[combined_analysis['signal_rank'] > 0.8]
    return top_signals


def main():
    input_dir = '/Users/jazzhashzzz/Desktop/data for scripts/data bento data/SPY/SPY'
    
    all_analysis = []
    
    for filename in os.listdir(input_dir):
        # Only process original trade files, skip analysis files
        if filename.endswith('.trades.csv') and not filename.startswith('hft_analysis'):
            print(f"Processing {filename}")
            file_path = os.path.join(input_dir, filename)
            
            try:
                # First read the header to check actual column names
                df = pd.read_csv(file_path, nrows=0)
                print(f"Available columns: {df.columns.tolist()}")
                
                # Then read the full file with verified columns
                df = pd.read_csv(file_path)
                
                microstructure_analysis = analyze_microstructure(df)
                signals_analysis = identify_potential_signals(microstructure_analysis)
                cross_sectional = analyze_cross_sectional(signals_analysis)
                all_analysis.append(signals_analysis)
                
                output_filename = f"hft_analysis_{filename}"
                signals_analysis.to_csv(os.path.join(input_dir, output_filename), index=False)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    
    if all_analysis:
        combined_analysis = pd.concat(all_analysis)
        
        print("\nMarket Microstructure Analysis")
        print("=============================")
        print(f"Total windows analyzed: {len(combined_analysis)}")
        print(f"Windows with strong signals (3+ indicators): {len(combined_analysis[combined_analysis['signal_strength'] >= 3])}")
        
        # Calculate key statistics
        print("\nKey Statistics:")
        print(f"Average order imbalance: {combined_analysis['order_imbalance'].mean():.4f}")
        print(f"Average trade size: {combined_analysis['avg_trade_size'].mean():.2f}")
        print(f"Median time between trades (microseconds): {combined_analysis['avg_time_between_trades'].median():.2f}")
        print(f"Average price impact: {combined_analysis['price_impact'].mean():.6f}")
        
        # Statistical significance tests
        for column in ['order_imbalance', 'price_impact', 'vwap_diff']:
            t_stat, p_value = stats.ttest_1samp(combined_analysis[column].dropna(), 0)
            print(f"\n{column} significance test:")
            print(f"t-statistic: {t_stat:.3f}")
            print(f"p-value: {p_value:.4f}")
        
        return combined_analysis
    
    return None

if __name__ == "__main__":
    results = main()