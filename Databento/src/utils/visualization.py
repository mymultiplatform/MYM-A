import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import pandas as pd

def format_price(x, p):
    """Format price values for the y-axis"""
    return f'${x:,.2f}'

def create_price_plot(comparison_df, last_timestamp, extended_actual_df=None, save_path='price_prediction_plot.png'):
    """
    Create and save a plot of actual vs predicted prices with enhanced visualization
    Args:
        comparison_df: DataFrame with predicted and actual prices
        last_timestamp: Timestamp where prediction starts
        extended_actual_df: Additional actual price data beyond prediction start
        save_path: Path to save the plot
    """
    plt.style.use('bmh')
    plt.figure(figsize=(15, 8))
    
    # Convert timestamp to datetime if it's not already
    comparison_df['timestamp'] = pd.to_datetime(comparison_df['timestamp'])
    
    # Plot actual prices from comparison_df
    actual_mask = comparison_df['type'] == 'actual'
    if any(actual_mask):
        plt.plot(comparison_df[actual_mask]['timestamp'],
                comparison_df[actual_mask]['price'],
                label='Actual Price',
                color='#2E86C1',
                linewidth=2,
                zorder=2)
    
    # Plot extended actual prices if provided
    if extended_actual_df is not None:
        plt.plot(extended_actual_df.index,
                extended_actual_df['price'],
                color='#2E86C1',
                linewidth=2,
                zorder=2)
    
    # Plot predicted prices
    predicted_mask = comparison_df['type'] == 'predicted'
    if any(predicted_mask):
        plt.plot(comparison_df[predicted_mask]['timestamp'],
                comparison_df[predicted_mask]['price'],
                label='Predicted Price',
                color='#E74C3C',
                linewidth=2,
                linestyle='--',
                zorder=3)
    
    # Add vertical line at cutoff point
    plt.axvline(x=last_timestamp,
                color='#7F8C8D',
                linestyle=':',
                label='Prediction Start',
                zorder=1)
    
    # Add shaded background for prediction period
    plt.axvspan(last_timestamp, 
                comparison_df['timestamp'].max(),
                color='#EAECEE',
                alpha=0.3,
                zorder=0)
    
    # Customize the plot
    plt.title('Price Movement Analysis: Actual vs Predicted',
             fontsize=16,
             pad=20,
             fontweight='bold')
    plt.xlabel('Time', fontsize=12, fontweight='bold')
    plt.ylabel('Price ($)', fontsize=12, fontweight='bold')
    
    # Enhanced legend
    plt.legend(fontsize=10,
              loc='upper left',
              bbox_to_anchor=(0.02, 0.98),
              frameon=True,
              fancybox=True,
              shadow=True)
    
    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axis
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()
    
    # Format y-axis
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.2f}'))
    
    # Ensure proper spacing
    plt.margins(x=0.02)
    plt.tight_layout()
    
    # Save the plot with high resolution
    plt.savefig(save_path,
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()
    
    return save_path