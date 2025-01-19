import pandas as pd
import os
from pathlib import Path
import numpy as np
from scipy import stats
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Input directory containing raw tick data
input_dir = '/Users/jazzhashzzz/Desktop/data for scripts/data bento data/SPY/SPY'

import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf
from datetime import datetime, timedelta

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option Greeks using Black-Scholes
    S: Stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free rate
    sigma: Volatility
    """
    
    # Calculate d1 and d2
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Standard normal CDF and PDF
    N = stats.norm.cdf
    n = stats.norm.pdf
    
    if option_type.lower() == 'call':
        # Call option calculations
        price = S*N(d1) - K*np.exp(-r*T)*N(d2)
        delta = N(d1)
        theta = (-S*sigma*n(d1))/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*N(d2)
    else:
        # Put option calculations
        price = K*np.exp(-r*T)*N(-d2) - S*N(-d1)
        delta = N(d1) - 1
        theta = (-S*sigma*n(d1))/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*N(-d2)
    
    # Common Greeks
    gamma = n(d1)/(S*sigma*np.sqrt(T))
    vega = S*np.sqrt(T)*n(d1)
    rho = K*T*np.exp(-r*T)*N(d2) if option_type.lower() == 'call' else -K*T*np.exp(-r*T)*N(-d2)
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

def get_options_data(symbol='SPY', save_csv=True):
    """
    Get options data and calculate Greeks
    """
    # Get stock data
    stock = yf.Ticker(symbol)
    current_price = stock.info['regularMarketPrice']
    risk_free_rate = 0.05  # You can update this
    
    # Get all available expiration dates
    expirations = stock.options
    
    all_options_data = []
    
    for exp_date in expirations:
        # Calculate time to expiration in years
        expiry = datetime.strptime(exp_date, '%Y-%m-%d')
        T = (expiry - datetime.now()).days / 365
        
        # Get options chain
        opt = stock.option_chain(exp_date)
        
        # Process calls
        calls = opt.calls
        calls['option_type'] = 'call'
        
        # Process puts
        puts = opt.puts
        puts['option_type'] = 'put'
        
        # Combine options
        options = pd.concat([calls, puts])
        
        # Calculate Greeks for each option
        for idx, row in options.iterrows():
            greeks = calculate_greeks(
                S=current_price,
                K=row['strike'],
                T=T,
                r=risk_free_rate,
                sigma=row['impliedVolatility'],
                option_type=row['option_type']
            )
            
            option_data = {
                'symbol': symbol,
                'expiration': exp_date,
                'strike': row['strike'],
                'option_type': row['option_type'],
                'bid': row['bid'],
                'ask': row['ask'],
                'implied_volatility': row['impliedVolatility'],
                'volume': row['volume'],
                'open_interest': row['openInterest'],
                'days_to_expiry': T*365,
                'theoretical_price': greeks['price'],
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'theta': greeks['theta'],
                'vega': greeks['vega'],
                'rho': greeks['rho']
            }
            
            all_options_data.append(option_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_options_data)
    
    # Save to CSV
    if save_csv:
        filename = f'{symbol}_options_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    
    return df

# Run the analysis
if __name__ == "__main__":
    # Get options data for SPY
    print("Fetching options data...")
    options_df = get_options_data('SPY')
    
    # Display summary
    print("\nOptions Data Summary:")
    print("=====================")
    print(f"Total options analyzed: {len(options_df)}")
    print("\nSample of the data:")
    print(options_df.head())
    
    # Display some key statistics
    print("\nAverage Greeks by Option Type:")
    print(options_df.groupby('option_type')[['delta', 'gamma', 'theta', 'vega', 'rho']].mean())