import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize

def calculate_returns(prices):
    """Calculate log returns from a numpy array of prices"""
    return np.log(prices[1:]/prices[:-1])

def calculate_ar1_residuals(returns, window=1008):
    """
    Calculate AR(1) residuals manually: r_t = μ + φ*r_{t-1} + ε_t
    where ε_t is the residual we want to calculate
    """
    
    residuals = np.zeros_like(returns)
    
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
        residuals[t] = returns[t] - expected_return
    
    # Handle initial window period
    residuals[:window] = returns[:window] - np.mean(returns[:window])
    
    return residuals

def garch_variance(params, residuals):
    """Calculate GARCH(1,1) variances using only previous forecast and residual"""
    omega, alpha, beta = params
    T = len(residuals)
    h = np.zeros(T)
    
    # Initialize first variance using the unconditional variance
    unconditional_var = omega / (1 - alpha - beta)
    h[0] = unconditional_var
    
    # Corrected GARCH formula indexing
    for t in range(1, T):
        h[t] = omega + alpha * residuals[t-1]**2 + beta * h[t-1]
    
    return h
def garch_likelihood(params, residuals):
    """GARCH(1,1) negative log-likelihood function"""
    omega, alpha, beta = params
    T = len(residuals)
    h = garch_variance(params, residuals)
    
    llh = -0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + residuals**2 / h)
    return -llh

def estimate_garch_parameters(residuals):
    """Estimate GARCH parameters using MLE"""
    initial_params = [0.0001, 0.1, 0.8]
    bounds = ((1e-6, None), (1e-6, 0.9999), (1e-6, 0.9999))
    constraints = ({'type': 'ineq', 'fun': lambda x: 0.9999 - x[1] - x[2]})
    
    result = minimize(garch_likelihood, 
                     initial_params,
                     args=(residuals,),
                     bounds=bounds,
                     constraints=constraints,
                     method='SLSQP')
    
    return result.x

def forecast_volatility(last_residual, last_variance, omega, alpha, beta, n_days):
    """Forecast volatility with mean reversion"""
    forecasts = np.zeros(n_days)
    unconditional_var = omega / (1 - alpha - beta)
    current_h = omega + alpha * last_residual**2 + beta * last_variance
    
    forecasts[0] = np.sqrt(current_h)
    for t in range(1, n_days):
        current_h = omega + (alpha + beta) * current_h
        # Add mean reversion
        current_h = current_h + (unconditional_var - current_h) * (1 - alpha - beta)
        forecasts[t] = np.sqrt(current_h)
    
    return forecasts

def prepare_and_forecast(csv_path):
    # Read CSV
    df = pd.read_csv(csv_path,
                    usecols=['Date', 'Open', 'High', 'Low', 'Close', 
                            'Volume', 'Dividends', 'Stock Splits', 'Capital Gains'])
    
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
    
    # Define periods - adjust these to use historical data instead of future dates
    training_start = pd.to_datetime('2020-01-02')
    training_end = pd.to_datetime('2024-12-13')     
    forecast_start = pd.to_datetime('2024-12-16')   
    forecast_end = pd.to_datetime('2024-12-31')     
    
    # Split data
    training_data = df[(df['Date'] >= training_start) & (df['Date'] <= training_end)]
    actual_data = df[(df['Date'] > training_end) & (df['Date'] <= forecast_end)]
    
    # Calculate initial returns
    training_returns = calculate_returns(training_data['Close'].values)
    
    # Initialize arrays for storing rolling results
    window_size = 1008
    n_training = len(training_returns)
    rolling_forecasts = np.zeros(n_training)
    rolling_parameters = []
    
    # Perform rolling window analysis
    for i in range(window_size, n_training):
        window_returns = training_returns[i-window_size:i]
        
        try:
            # Calculate AR(1) residuals for this window
            window_residuals = calculate_ar1_residuals(window_returns)
            
            # Estimate GARCH parameters for this window
            omega, alpha, beta = estimate_garch_parameters(window_residuals)
            
            # Store parameters
            rolling_parameters.append({
                'date': training_data['Date'].iloc[i],
                'omega': omega,
                'alpha': alpha,
                'beta': beta,
                'persistence': alpha + beta,
                'long_run_var': omega/(1-alpha-beta)
            })
            
            # Calculate variance for this window
            h = garch_variance([omega, alpha, beta], window_residuals)
            rolling_forecasts[i] = np.sqrt(h[-1])
            
        except:
            # If estimation fails, use simple volatility
            rolling_forecasts[i] = np.std(window_returns)
    
    # Get final window for forecasting
    final_window_returns = training_returns[-window_size:]
    final_residuals = calculate_ar1_residuals(final_window_returns)
    final_omega, final_alpha, final_beta = estimate_garch_parameters(final_residuals)
    
    # Calculate final historical volatilities
    final_h = garch_variance([final_omega, final_alpha, final_beta], final_residuals)
    
    # Change back to 5 or 10 days for forecasting
    n_forecast_days = 10  # or 10, depending on what you want

    # Update forecast call
    forecast = forecast_volatility(final_residuals[-1], final_h[-1], 
                                final_omega, final_alpha, final_beta, n_forecast_days)

    # Update forecast dates
    forecast_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='B')[:n_forecast_days]
    # Create results DataFrames
    forecast_results = pd.DataFrame({
        'Date': forecast_dates,
        'Forecasted_Volatility': forecast,
        'Forecasted_Volatility_Percentage': forecast * 100
    })
    
    rolling_params_df = pd.DataFrame(rolling_parameters)
    
    # Calculate actual volatility if available
    if len(actual_data) > 1:
        actual_returns = calculate_returns(actual_data['Close'].values)
        actual_volatilities = []
        
        # Calculate daily volatility as absolute returns
        actual_volatilities = np.abs(actual_returns)
        
        forecast_results['Actual_Volatility'] = actual_volatilities
        forecast_results['Actual_Volatility_Percentage'] = [v * 100 for v in actual_volatilities]
    #forecast plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot 1: Forecasted vs Actual Volatility
    ax1.plot(forecast_results['Date'], forecast_results['Forecasted_Volatility_Percentage'], 
             label='Forecasted Volatility', marker='o')
    if 'Actual_Volatility' in forecast_results.columns:
        ax1.plot(forecast_results['Date'], forecast_results['Actual_Volatility_Percentage'], 
                label='Actual Volatility', marker='x')
    ax1.set_title('GARCH(1,1) Volatility Forecast')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Volatility (%)')
    ax1.legend()
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Rolling Parameters
    ax2.plot(rolling_params_df['date'], rolling_params_df['alpha'], label='Alpha')
    ax2.plot(rolling_params_df['date'], rolling_params_df['beta'], label='Beta')
    ax2.plot(rolling_params_df['date'], rolling_params_df['persistence'], label='Persistence')
    ax2.set_title('Rolling GARCH Parameters')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Parameter Value')
    ax2.legend()
    ax2.grid(True)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('garch_analysis.png')
    
    # Print summary statistics
    print("\nFinal GARCH(1,1) Parameters:")
    print(f"omega: {final_omega:.6f}")
    print(f"alpha: {final_alpha:.6f}")
    print(f"beta: {final_beta:.6f}")
    print(f"alpha + beta: {final_alpha + final_beta:.6f}")
    print(f"Long-run variance: {final_omega/(1-final_alpha-final_beta):.6f}")
    
    return_dates = training_data['Date'].iloc[1:].reset_index(drop=True)
    
    return {
        'forecasts': forecast_results,
        'rolling_parameters': rolling_params_df,
        'rolling_forecasts': pd.Series(rolling_forecasts, index=return_dates)  # Use return_dates instead of training_data['Date']
    }


#results = prepare_and_forecast('/Users/jazzhashzzz/Desktop/data for scripts/Yfinance Data/SPY_5y.csv')
results = prepare_and_forecast(r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\yfinance\SPY_5y.csv")

print("\nForecasting Results:")
print(results['forecasts'].to_string())
results['rolling_parameters'].to_csv(r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\yfinance\rolling_parameters_SPY.csv", index=False)

# Save forecast results
results['forecasts'].to_csv(r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\yfinance\forecast_results_SPY.csv", index=False)

plt.show()

