import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize

def calculate_returns(prices):
    """Calculate log returns from a numpy array of prices"""
    return np.log(prices[1:]/prices[:-1])

def calculate_rolling_residuals(returns, window=22):
    """Calculate rolling AR(1) residuals"""
    from statsmodels.tsa.arima.model import ARIMA
    rolling_residuals = np.zeros_like(returns)
    
    for i in range(window, len(returns)):
        window_returns = returns[i-window:i]
        try:
            model = ARIMA(window_returns, order=(1,0,0))
            results = model.fit()
            # Store only the last residual
            rolling_residuals[i] = results.resid[-1]
        except:
            rolling_residuals[i] = returns[i] - np.mean(window_returns)
    
    # Fill first window periods with simple residuals
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
    
    # Define periods
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
    # Initialize arrays for storing rolling results
    window_size = 22
    n_training = len(training_returns)
    rolling_forecasts = np.zeros(n_training)
    actual_rolling_vol = np.zeros(n_training)
    rolling_parameters = []
    
    # Calculate actual rolling volatility first
    for i in range(window_size, n_training):
        # Calculate actual volatility using past window_size days
        actual_rolling_vol[i] = np.std(training_returns[i-window_size:i])
    
    # Perform rolling window analysis and predictions
    for i in range(window_size, n_training):
        window_returns = training_returns[i-window_size:i]
        
        try:
            # Calculate AR(1) residuals for this window
            window_residuals = calculate_rolling_residuals(window_returns)
            
            # Estimate GARCH parameters for this window
            omega, alpha, gamma, beta = estimate_asymmetric_garch_parameters(window_residuals)
            
            # Store parameters
            rolling_parameters.append({
                'date': training_data['Date'].iloc[i],
                'omega': omega,
                'alpha': alpha,
                'gamma': gamma,
                'beta': beta,
                'persistence': alpha + beta + gamma/2,
                'long_run_var': omega/(1-alpha-beta-gamma/2),
                'actual_volatility': actual_rolling_vol[i]
            })
            
            # Calculate variance forecast for next period
            h = asymmetric_garch_variance([omega, alpha, gamma, beta], window_residuals)
            rolling_forecasts[i] = np.sqrt(h[-1])
            
        except:
            # If estimation fails, use simple volatility
            rolling_forecasts[i] = actual_rolling_vol[i]
    # Get final window for forecasting
    final_window_returns = training_returns[-window_size:]
    final_residuals = calculate_rolling_residuals(final_window_returns)
    final_omega, final_alpha, final_gamma, final_beta = estimate_asymmetric_garch_parameters(final_residuals)
    
    # Calculate final historical volatilities
    final_h = asymmetric_garch_variance([final_omega, final_alpha, final_gamma, final_beta], final_residuals)
    
    # Generate forecasts
    forecast = forecast_asymmetric_volatility(final_residuals[-1], final_h[-1], 
                                final_omega, final_alpha, final_gamma, final_beta, 10)
    
    # Create forecast dates
    forecast_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='B')[:10]
    
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
        actual_volatility = np.std(actual_returns)
        forecast_results['Actual_Volatility'] = actual_volatility
        forecast_results['Actual_Volatility_Percentage'] = actual_volatility * 100
    
        # Update plots to show rolling comparison
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
    
    # Plot 1: Forecasted vs Actual Future Volatility
    ax1.plot(forecast_results['Date'], forecast_results['Forecasted_Volatility_Percentage'], 
             label='Forecasted Volatility', marker='o')
    if 'Actual_Volatility' in forecast_results.columns:
        ax1.plot(forecast_results['Date'], forecast_results['Actual_Volatility_Percentage'], 
                label='Actual Volatility', marker='x')
    ax1.set_title('Future Volatility Forecast')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Volatility (%)')
    ax1.legend()
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Rolling Parameters
    ax2.plot(rolling_params_df['date'], rolling_params_df['alpha'], label='Alpha')
    ax2.plot(rolling_params_df['date'], rolling_params_df['gamma'], label='Gamma')
    ax2.plot(rolling_params_df['date'], rolling_params_df['beta'], label='Beta')
    ax2.plot(rolling_params_df['date'], rolling_params_df['persistence'], label='Persistence')
    ax2.set_title('Rolling GARCH Parameters')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Parameter Value')
    ax2.legend()
    ax2.grid(True)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Rolling Volatility Comparison
    ax3.plot(rolling_params_df['date'], rolling_forecasts[window_size:], 
             label='Predicted Rolling Volatility', alpha=0.7)
    ax3.plot(rolling_params_df['date'], rolling_params_df['actual_volatility'], 
             label='Actual Rolling Volatility', alpha=0.7)
    ax3.set_title('Rolling Volatility Comparison')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Volatility')
    ax3.legend()
    ax3.grid(True)
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('garch_analysis.png')
    
    # Calculate performance metrics
    forecast_error = rolling_forecasts[window_size:] - actual_rolling_vol[window_size:]
    mse = np.mean(forecast_error**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(forecast_error))
    
    print("\nRolling Volatility Forecast Performance:")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Root Mean Squared Error: {rmse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    
    return {
        'forecasts': forecast_results,
        'rolling_parameters': rolling_params_df,
        'rolling_forecasts': pd.Series(rolling_forecasts[window_size:], index=rolling_params_df['date']),
        'actual_volatility': pd.Series(actual_rolling_vol[window_size:], index=rolling_params_df['date']),
        'performance_metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
    }

# [All your previous code remains exactly the same until the last lines]

# Usage
results = prepare_and_forecast('/Users/jazzhashzzz/Desktop/data for scripts/Yfinance Data/SPY_5y.csv')
print("\nForecasting Results:")
print(results['forecasts'].to_string())
print("\nRolling Parameters:")
print(results['rolling_parameters'].to_string())

plt.show()

