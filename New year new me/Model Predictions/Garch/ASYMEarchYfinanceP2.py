import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize
# Global parameters for time windows and dates
# Global parameters for time windows and dates
WINDOW_SIZE = 22  # Rolling window size for calculations

# Update these dates to match your actual data
TRAINING_START = pd.to_datetime('2024-12-16')
TRAINING_END = pd.to_datetime('2024-12-31')
FORECAST_START = pd.to_datetime('2024-12-16')
FORECAST_END = pd.to_datetime('2024-12-31')
def calculate_returns(prices):
    """Calculate log returns from a numpy array of prices"""
    return np.log(prices[1:]/prices[:-1])

def calculate_ar1_residuals(returns, window=WINDOW_SIZE):
    """
    Calculate AR(1) residuals manually: r_t = μ + φ*r_{t-1} + ε_t
    where ε_t is the residual we want to calculate
    """
    
    rolling_residuals = np.zeros_like(returns)
    
    for t in range(window, len(returns)):
        # Get window of data
        window_data = returns[t-window:t]
        
        # Lag the returns by 1 period
        y = window_data[1:]  # r_t
        X = window_data[:-1]  # r_{t-1}
        
        # Calculate φ using correlation
        mean_X = np.mean(X)
        mean_y = np.mean(y)
        phi = np.sum((X - mean_X) * (y - mean_y)) / np.sum((X - mean_X)**2)
        
        # Calculate intercept (μ)
        mu = mean_y - phi * mean_X
        
        # Calculate expected return for current period
        expected_return = mu + phi * returns[t-1]
        
        # Calculate residual
        rolling_residuals[t] = returns[t] - expected_return
    
    # Handle initial window period
    rolling_residuals[:window] = returns[:window] - np.mean(returns[:window])
    
    return rolling_residuals

def asymmetric_garch_variance(params, residuals):
    """Calculate asymmetric GARCH(1,1) variances with leverage effect"""
    omega, alpha, gamma, beta = params
    T = len(residuals)
    h = np.zeros(T)
    
    # Initialize first variance using the unconditional variance
    # Note: The unconditional variance formula changes with asymmetric term
    unconditional_var = omega / (1 - alpha - beta - gamma/2)
    h[0] = unconditional_var
    
    # Update using previous forecast, residual, and asymmetric term
    for t in range(1, T):
        # Negative return indicator
        I_t = 1 if residuals[t-1] < 0 else 0
        # TARCH specification
        h[t] = omega + alpha * residuals[t-1]**2 + gamma * I_t * residuals[t-1]**2 + beta * h[t-1]
    
    return h

def asymmetric_garch_likelihood(params, residuals):
    """Asymmetric GARCH(1,1) negative log-likelihood function"""
    omega, alpha, gamma, beta = params
    T = len(residuals)
    h = asymmetric_garch_variance(params, residuals)
    
    llh = -0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + residuals**2 / h)
    return -llh

def estimate_asymmetric_garch_parameters(residuals):
    """Estimate asymmetric GARCH parameters using MLE"""
    # Initial parameters: [omega, alpha, gamma, beta]
    initial_params = [0.0001, 0.1, 0.05, 0.8]
    
    # Bounds for parameters
    bounds = (
        (1e-6, None),    # omega > 0
        (1e-6, 0.9999),  # alpha > 0
        (-0.9999, 0.9999), # gamma can be negative or positive
        (1e-6, 0.9999)   # beta > 0
    )
    
    # Constraint: alpha + beta + gamma/2 < 1 for stationarity
    constraints = ({
        'type': 'ineq', 
        'fun': lambda x: 0.9999 - x[1] - x[2]/2 - x[3]
    })
    
    result = minimize(asymmetric_garch_likelihood, 
                     initial_params,
                     args=(residuals,),
                     bounds=bounds,
                     constraints=constraints,
                     method='SLSQP')
    
    return result.x

def forecast_asymmetric_volatility(last_residual, last_variance, omega, alpha, gamma, beta, n_days):
    """Forecast volatility with asymmetric effects"""
    forecasts = np.zeros(n_days)
    unconditional_var = omega / (1 - alpha - beta - gamma/2)
    
    # Consider asymmetric effect in initial forecast
    I_t = 1 if last_residual < 0 else 0
    current_h = omega + alpha * last_residual**2 + gamma * I_t * last_residual**2 + beta * last_variance
    
    forecasts[0] = np.sqrt(current_h)
    for t in range(1, n_days):
        # For future periods, use expected value of indicator function (0.5)
        current_h = omega + (alpha + gamma/2 + beta) * current_h
        # Add mean reversion
        current_h = current_h + (unconditional_var - current_h) * (1 - alpha - gamma/2 - beta)
        forecasts[t] = np.sqrt(current_h)
    
    return forecasts

def prepare_and_forecast(csv_path):
    # Read CSV
    df = pd.read_csv(csv_path,
                    usecols=['Date', 'Open', 'High', 'Low', 'Close', 
                            'Volume', 'Dividends', 'Stock Splits', 'Capital Gains'])
    
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
    
    # Replace date definitions with:
    training_start = TRAINING_START
    training_end = TRAINING_END    
    forecast_start = FORECAST_START  
    forecast_end = FORECAST_END    
    
    # Replace window size reference:
    window_size = WINDOW_SIZE  
# In prepare_and_forecast function, change how actual_data is defined:
    training_data = df[(df['Date'] >= training_start) & (df['Date'] <= training_end)]
    actual_data = df[(df['Date'] >= forecast_start) & (df['Date'] <= forecast_end)]  # Changed from > to >=
    # Calculate initial returns
    training_returns = calculate_returns(training_data['Close'].values)
    
    # Initialize arrays for storing rolling results
    n_training = len(training_returns)
    rolling_forecasts = np.zeros(n_training)
    rolling_parameters = []
    
    # In the rolling window analysis section:
    for i in range(window_size, n_training):
        window_returns = training_returns[i-window_size:i]
        
        try:
            window_residuals = calculate_ar1_residuals(window_returns)
            omega, alpha, gamma, beta = estimate_asymmetric_garch_parameters(window_residuals)
            
            rolling_parameters.append({
                'date': training_data['Date'].iloc[i],
                'omega': omega,
                'alpha': alpha,
                'gamma': gamma,  # Added gamma
                'beta': beta,
                'persistence': alpha + beta + gamma/2,  # Updated persistence calculation
                'long_run_var': omega/(1-alpha-beta-gamma/2)  # Updated long-run variance
            })
            # Calculate variance for this window
            h = asymmetric_garch_variance([omega, alpha, gamma, beta], window_residuals)
            rolling_forecasts[i] = np.sqrt(h[-1])
            
        except:
            # If estimation fails, use simple volatility
            rolling_forecasts[i] = np.std(window_returns)
    
    # Get final window for forecasting
    final_window_returns = training_returns[-window_size:]
    final_residuals = calculate_ar1_residuals(final_window_returns)
    final_omega, final_alpha, final_gamma, final_beta = estimate_asymmetric_garch_parameters(final_residuals)
    
    # Calculate final historical volatilities
    final_h = asymmetric_garch_variance([final_omega, final_alpha, final_gamma, final_beta], final_residuals)
    
    # First define n_forecast_days and forecast_dates
    n_forecast_days = len(actual_data)
    forecast_dates = actual_data['Date']

    # Calculate actual returns and define n_actual_days
    if len(actual_data) > 1:
        actual_returns = calculate_returns(actual_data['Close'].values)
        n_actual_days = len(actual_returns)
        forecast_dates = actual_data['Date'].iloc[:n_actual_days]
        
        # Generate forecast using n_actual_days
        forecast = forecast_asymmetric_volatility(final_residuals[-1], final_h[-1], 
                                                final_omega, final_alpha, final_gamma, final_beta, 
                                                n_actual_days)
        
        # Create results DataFrame
        forecast_results = pd.DataFrame({
            'Date': forecast_dates,
            'Forecasted_Volatility': forecast,
            'Forecasted_Volatility_Percentage': forecast * 100,
            'Actual_Volatility': np.abs(actual_returns),
            'Actual_Volatility_Percentage': np.abs(actual_returns) * 100
        })
    else:
        # Handle case where there's no actual data
        n_actual_days = n_forecast_days
        forecast = forecast_asymmetric_volatility(final_residuals[-1], final_h[-1], 
                                                final_omega, final_alpha, final_gamma, final_beta, 
                                                n_actual_days)
        
        forecast_results = pd.DataFrame({
            'Date': forecast_dates,
            'Forecasted_Volatility': forecast,
            'Forecasted_Volatility_Percentage': forecast * 100
        })

    
    rolling_params_df = pd.DataFrame(rolling_parameters)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Plot 1: Keep the same as your current code
    ax1.plot(forecast_results['Date'], forecast_results['Forecasted_Volatility_Percentage'], 
            label='Forecasted Volatility', marker='o')
    if 'Actual_Volatility' in forecast_results.columns:
        ax1.plot(forecast_results['Date'], forecast_results['Actual_Volatility_Percentage'], 
                label='Actual Volatility', marker='x')
    ax1.set_title('Asymmetric GARCH Volatility Forecast')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Volatility (%)')
    ax1.legend()
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Include gamma in the rolling parameters plot
    ax2.plot(rolling_params_df['date'], rolling_params_df['alpha'], label='Alpha')
    ax2.plot(rolling_params_df['date'], rolling_params_df['gamma'], label='Gamma', linestyle='--')
    ax2.plot(rolling_params_df['date'], rolling_params_df['beta'], label='Beta')
    ax2.plot(rolling_params_df['date'], rolling_params_df['persistence'], label='Persistence')
    ax2.set_title('Rolling Asymmetric GARCH Parameters')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Parameter Value')
    ax2.legend()
    ax2.grid(True)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('garch_analysis.png')
        
    # Calculate performance metrics
    # First, we need to calculate actual rolling volatility
    actual_rolling_vol = np.zeros(len(rolling_forecasts))
    for i in range(window_size, len(training_returns)):
        actual_rolling_vol[i] = np.std(training_returns[i-window_size:i])
    
    # Now calculate forecast error metrics
    forecast_error = rolling_forecasts[window_size:] - actual_rolling_vol[window_size:]
    valid_indices = ~np.isnan(forecast_error)  # Remove any NaN values
    forecast_error = forecast_error[valid_indices]
    
    # Calculate metrics only if we have valid data
    if len(forecast_error) > 0:
        mse = np.mean(forecast_error**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(forecast_error))
        
        print("\nRolling Volatility Forecast Performance:")
        print(f"Mean Squared Error: {mse:.6f}")
        print(f"Root Mean Squared Error: {rmse:.6f}")
        print(f"Mean Absolute Error: {mae:.6f}")
    
    # Create return_dates for the index
    return_dates = training_data['Date'].iloc[1:].reset_index(drop=True)
    
    return {
        'forecasts': forecast_results,
        'rolling_parameters': rolling_params_df,
        'rolling_forecasts': pd.Series(rolling_forecasts, index=return_dates)
    }
# [All your previous code remains exactly the same until the last lines]
def analyze_garch_implementation(df, training_returns, residuals, h_t, params, rolling_params_df):
    """
    Analyzes GARCH implementation with current data and parameters
    
    Parameters:
    - df: Original dataframe with price data
    - training_returns: Calculated returns array
    - residuals: Calculated residuals
    - h_t: Variance series
    - params: Current [omega, alpha, gamma, beta]
    - rolling_params_df: DataFrame with rolling parameters
    """
    
    omega, alpha, gamma, beta = params
    
    analysis = f"""
GARCH Implementation Analysis Using Current Data
=============================================

1. Data Overview
---------------
Time Period: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}
Number of observations: {len(df)}
Current price: ${df['Close'].iloc[-1]:.2f}

2. Returns Analysis
------------------
Recent returns (last 5 days):
{training_returns[-5:]}
Mean return: {np.mean(training_returns):.6f}
Volatility: {np.std(training_returns):.6f}

3. Residuals Analysis
--------------------
Recent residuals (last 5 periods):
{residuals[-5:]}
Mean residual: {np.mean(residuals):.6f}
Standard deviation: {np.std(residuals):.6f}

4. Current GARCH Parameters
--------------------------
ω (omega): {omega:.6f}
α (alpha): {alpha:.6f}
γ (gamma): {gamma:.6f}
β (beta): {beta:.6f}

5. Model Properties
-----------------
Persistence: {alpha + beta + gamma/2:.6f}
Unconditional Variance: {omega/(1 - alpha - beta - gamma/2):.6f}
Half-life of Variance Shocks: {-np.log(2)/np.log(alpha + beta + gamma/2):.2f} days

6. Parameter Evolution
--------------------
Parameter Ranges over Rolling Windows:
Alpha: {rolling_params_df['alpha'].min():.4f} to {rolling_params_df['alpha'].max():.4f}
Beta: {rolling_params_df['beta'].min():.4f} to {rolling_params_df['beta'].max():.4f}
Gamma: {rolling_params_df['gamma'].min():.4f} to {rolling_params_df['gamma'].max():.4f}

7. Variance Analysis
------------------
Current Conditional Variance: {h_t[-1]:.6f}
Volatility (sqrt of variance): {np.sqrt(h_t[-1]):.6f}
Average Conditional Variance: {np.mean(h_t):.6f}

8. Leverage Effect Analysis
-------------------------
Impact of positive shock: {alpha:.6f}
Impact of negative shock: {alpha + gamma:.6f}
Asymmetry ratio: {(alpha + gamma)/alpha:.2f}

9. Model Diagnostics
------------------
Stationarity: {'Yes' if (alpha + beta + gamma/2) < 1 else 'No'}
Leverage Effect Present: {'Yes' if gamma != 0 else 'No'}
Parameter Significance:
- Alpha/Variance ratio: {alpha/np.std(h_t):.4f}
- Beta/Variance ratio: {beta/np.std(h_t):.4f}
- Gamma/Variance ratio: {gamma/np.std(h_t):.4f}

10. Recent Volatility Forecasts
----------------------------
Last 5 volatility forecasts:
{np.sqrt(h_t[-5:])}
"""
    
    return analysis


def run_garch_analysis(csv_path):
    """
    Runs complete GARCH analysis including forecasting and detailed model analysis
    
    Parameters:
    - csv_path: Path to CSV file containing price data
    
    Returns:
    - Dictionary containing all results and analysis
    """
    # Read CSV and prepare data
    df = pd.read_csv(csv_path,
                    usecols=['Date', 'Open', 'High', 'Low', 'Close', 
                            'Volume', 'Dividends', 'Stock Splits', 'Capital Gains'])
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
    
    # Run base analysis (this includes the plotting)
    results = prepare_and_forecast(csv_path)
    
    # Get training data
    training_start = TRAINING_START
    training_end = TRAINING_END
    training_data = df[(df['Date'] >= training_start) & (df['Date'] <= training_end)]
    
    # Calculate returns and get final window data
    training_returns = calculate_returns(training_data['Close'].values)
    final_window_returns = training_returns[-WINDOW_SIZE:]
    final_residuals = calculate_ar1_residuals(final_window_returns)
    
    # Get final parameters and variance series
    final_params = estimate_asymmetric_garch_parameters(final_residuals)
    final_h = asymmetric_garch_variance(final_params, final_residuals)
    
    # Generate detailed analysis
    analysis = analyze_garch_implementation(
        df=training_data,
        training_returns=training_returns,
        residuals=final_residuals,
        h_t=final_h,
        params=final_params,
        rolling_params_df=results['rolling_parameters']
    )
    
    # Print results
    print("\nForecasting Results:")
    print(results['forecasts'].to_string())
    results['forecasts'].to_csv(r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\yfinance\ASYMforecasts_SPY.csv", index=False)
    print("\nRolling Parameters:")
    results['rolling_parameters'].to_csv(r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\yfinance\ASYMrolling_parameters_SPY.csv", index=False)
    print(results['rolling_parameters'].to_string())
    print("\nDetailed GARCH Analysis:")
    print(analysis)
    
    # Show the plots
    plt.show()
    
    # Return comprehensive results
    return {
        **results,  # Include all original results
        'detailed_analysis': analysis,
        'final_parameters': {
            'omega': final_params[0],
            'alpha': final_params[1],
            'gamma': final_params[2],
            'beta': final_params[3]
        },
        'final_variance_series': final_h,
        'final_residuals': final_residuals
    }

# Usage
analysis_results = run_garch_analysis(r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\yfinance\SPY_5y.csv")
#analysis_results = run_garch_analysis('/Users/jazzhashzzz/Desktop/data for scripts/Yfinance Data/SPY_5y.csv')
