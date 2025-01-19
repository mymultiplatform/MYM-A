import pandas as pd
import os
from pathlib import Path
import numpy as np
from datetime import datetime, time

# Input directory containing OHLC files
input_dir = '/Users/jazzhashzzz/Desktop/data for scripts/data bento data/SPY/OHCL SPY'

def analyze_daily_returns(df):
    # Convert timestamp to datetime if it's not already
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    
    # Create time filter for 11:30 to 15:00 UTC
    df['time'] = df['ts_event'].dt.time
    mask = (df['time'] >= time(11, 30)) & (df['time'] <= time(15, 0))
    trading_period = df[mask].copy()
    
    # Calculate returns for each 15-minute interval
    trading_period['interval_return'] = trading_period['close'].pct_change() * 100
    
    # Group by date
    trading_period['date'] = trading_period['ts_event'].dt.date
    period_stats = []
    
    for date, group in trading_period.groupby('date'):
        # Get returns for each time interval
        for i in range(len(group) - 1):
            current_row = group.iloc[i]
            next_row = group.iloc[i + 1]
            
            interval_return = ((next_row['close'] - current_row['close']) / 
                             current_row['close']) * 100
            
            period_stats.append({
                'date': date,
                'start_time': current_row['ts_event'].strftime('%H:%M'),
                'end_time': next_row['ts_event'].strftime('%H:%M'),
                'interval_return': interval_return,
                'start_price': current_row['close'],
                'end_price': next_row['close'],
                'volume': next_row['volume']
            })
    
    return pd.DataFrame(period_stats)

def main():
    all_stats = []
    
    for filename in os.listdir(input_dir):
        if filename.startswith('OHLC_') and filename.endswith('.csv'):
            print(f"Processing {filename}")
            file_path = os.path.join(input_dir, filename)
            
            try:
                df = pd.read_csv(file_path)
                period_stats = analyze_daily_returns(df)
                all_stats.append(period_stats)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    if all_stats:
        combined_stats = pd.concat(all_stats)
        combined_stats = combined_stats.sort_values(['date', 'start_time'])
        
        # Analyze returns by time period
        time_analysis = combined_stats.groupby(['start_time', 'end_time']).agg({
            'interval_return': ['mean', 'std', 'count'],
            'volume': 'mean'
        }).round(3)
        
        print("\nReturn Analysis by Time Period:")
        print("===============================")
        print(time_analysis)
        
        # Overall statistics
        print("\nOverall Statistics:")
        print("==================")
        print(f"Average Return per 15-min interval: {combined_stats['interval_return'].mean():.3f}%")
        print(f"Return Std Dev: {combined_stats['interval_return'].std():.3f}%")
        print(f"Max Return: {combined_stats['interval_return'].max():.3f}%")
        print(f"Min Return: {combined_stats['interval_return'].min():.3f}%")
        
        # Calculate win rate by time period
        win_rate = combined_stats.groupby(['start_time', 'end_time']).apply(
            lambda x: (x['interval_return'] > 0).mean() * 100
        ).round(2)
        
        print("\nWin Rate by Time Period:")
        print("=======================")
        print(win_rate)
        
        # Save detailed statistics
        output_path = os.path.join(input_dir, 'time_period_analysis.csv')
        combined_stats.to_csv(output_path, index=False)
        
        time_analysis_path = os.path.join(input_dir, 'time_period_summary.csv')
        time_analysis.to_csv(time_analysis_path)
        
        print(f"\nDetailed statistics saved to: {output_path}")
        print(f"Time period summary saved to: {time_analysis_path}")
        
        return combined_stats, time_analysis
    
    else:
        print("No data files were processed successfully.")
        return None, None

if __name__ == "__main__":
    stats_df, time_analysis_df = main()