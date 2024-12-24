import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import pandas as pd


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