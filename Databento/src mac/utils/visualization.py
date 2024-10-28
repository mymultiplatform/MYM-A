import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import pandas as pd
from datetime import timedelta

def format_price(x, p):
    """Format price values for the y-axis"""
    return f'${x:,.2f}'

def create_price_plot(comparison_df, last_timestamp, save_path='price_prediction_plot.png'):
    """
    Create and save a plot of actual vs predicted prices with historical window
    matching prediction length
    
    Args:
        comparison_df: DataFrame containing both actual and predicted prices
        last_timestamp: Timestamp where prediction starts
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 8))
    
    # Convert timestamp to datetime if it's not already
    comparison_df['timestamp'] = pd.to_datetime(comparison_df['timestamp'])
    
    # Get prediction length from the data
    pred_length = len(comparison_df[comparison_df['type'] == 'predicted'])
    
    # Calculate display start time to match prediction length
    display_start = pd.to_datetime(last_timestamp) - pd.Timedelta(minutes=pred_length)
    
    # Filter data to show only the desired time window
    plot_df = comparison_df[comparison_df['timestamp'] >= display_start].copy()
    
    # Plot actual prices
    actual_mask = plot_df['type'] == 'actual'
    if any(actual_mask):
        plt.plot(plot_df[actual_mask]['timestamp'],
                plot_df[actual_mask]['price'],
                label='Actual Price',
                color='blue',
                linewidth=2)
    
    # Plot predicted prices
    predicted_mask = plot_df['type'] == 'predicted'
    if any(predicted_mask):
        plt.plot(plot_df[predicted_mask]['timestamp'],
                plot_df[predicted_mask]['price'],
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
    
    # Format x-axis with dynamic interval based on prediction length
    ax = plt.gca()
    interval = max(1, pred_length // 10)  # Show ~10 ticks across the plot
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))
    plt.gcf().autofmt_xdate()
    
    # Format y-axis
    ax.yaxis.set_major_formatter(FuncFormatter(format_price))
    
    # Adjust y-axis limits to add some padding
    prices = plot_df['price']
    y_min, y_max = prices.min(), prices.max()
    y_padding = (y_max - y_min) * 0.1
    plt.ylim(y_min - y_padding, y_max + y_padding)
    
    # Add padding to the layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path