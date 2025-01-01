import pandas as pd
import numpy as np
from scipy.optimize import minimize

def calculate_mle_params(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Calculate daily returns using Close prices
    df['Returns'] = df['Close'].pct_change()
    
    # Remove any NaN values
    returns = df['Returns'].dropna()
    
    # Define the negative log likelihood function
    def neg_log_likelihood(params, returns):
        mu, sigma = params
        # Calculate the negative log likelihood for normal distribution
        log_likelihood = np.sum(-0.5 * np.log(2 * np.pi * sigma**2) - 
                              (returns - mu)**2 / (2 * sigma**2))
        return -log_likelihood  # Minimize negative log likelihood
    
    # Initial guess for parameters (mean and standard deviation)
    initial_guess = [np.mean(returns), np.std(returns)]
    
    # Minimize negative log likelihood
    result = minimize(neg_log_likelihood, 
                     initial_guess,
                     args=(returns,),
                     method='Nelder-Mead')
    
    # Get the optimal parameters
    mu_mle, sigma_mle = result.x
    
    return {
        'mu': mu_mle,        # Mean
        'sigma': sigma_mle,  # Standard deviation
        'success': result.success,
        'convergence': result.message
    }

# Example usage:
file_path = '/Users/jazzhashzzz/Desktop/data for scripts/Yfinance Data/SPY_5y.csv'
params = calculate_mle_params(file_path)

if params['success']:
    print("MLE parameters:")
    print(f"Mean: {params['mu']}")
    print(f"Standard Deviation: {params['sigma']}")
    print(f"Convergence: {params['convergence']}")
else:
    print("MLE optimization failed.")