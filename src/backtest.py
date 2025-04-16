"""
Backtesting functionality for the statistical arbitrage strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from src.pca_model import perform_pca, get_factor_weights, calculate_factor_returns, get_optimal_components
from src.ou_process import get_tradable_stocks
from src.signals import calculate_all_s_scores, generate_signals, calculate_portfolio_weights

def rolling_backtest(returns_df, config):
    """
    Perform a rolling window backtest of the statistical arbitrage strategy.
    
    Parameters:
    -----------
    returns_df : DataFrame
        Daily returns for all stocks
    config : dict
        Configuration parameters
        
    Returns:
    --------
    DataFrame
        Strategy performance
    """
    # Extract parameters
    pca_window = config['PCA_WINDOW']
    ou_window = config['OU_WINDOW']
    n_components = config['N_COMPONENTS']
    variance_threshold = config['VARIANCE_THRESHOLD']
    kappa_threshold = config['KAPPA_THRESHOLD']
    s_threshold = config['S_THRESHOLD']
    
    # Ensure we have enough data
    if len(returns_df) <= pca_window:
        raise ValueError(f"Not enough data: need at least {pca_window+1} observations")
    
    # Initialize results containers
    strategy_returns = pd.Series(index=returns_df.index[pca_window:], dtype=float)
    portfolio_positions = pd.DataFrame(index=returns_df.index[pca_window:], columns=returns_df.columns)
    K_values = pd.Series(index=returns_df.index[pca_window:], dtype=float)
    tradable_counts = pd.Series(index=returns_df.index[pca_window:], dtype=int)
    
    # Rolling window backtest
    for t in range(pca_window, len(returns_df)):
        current_date = returns_df.index[t]
        
        try:
            # Get training data (PCA window)
            train_returns = returns_df.iloc[t-pca_window:t]
            
            # Perform PCA
            pca, standardized_returns = perform_pca(train_returns, n_components)
            
            # Determine optimal K components
            K = get_optimal_components(pca, variance_threshold)
            K_values.loc[current_date] = K
            
            # Get last ou_window days for OU process fitting
            ou_estimation_returns = train_returns.iloc[-ou_window:]
            
            # Calculate factor returns
            factor_weights = get_factor_weights(pca, train_returns.columns, train_returns.std())
            factor_returns = calculate_factor_returns(ou_estimation_returns, factor_weights)
            
            # Fit regression for each stock to get residuals
            residuals = pd.DataFrame(index=ou_estimation_returns.index, columns=ou_estimation_returns.columns)
            
            for stock in ou_estimation_returns.columns:
                y = ou_estimation_returns[stock]
                X = factor_returns.iloc[:, :K]  # Use only first K factors
                X = sm.add_constant(X)
                
                model = sm.OLS(y, X).fit()
                residuals[stock] = model.resid
            
            # Identify tradable stocks
            ou_params, tradable_stocks = get_tradable_stocks(residuals, kappa_threshold)
            tradable_counts.loc[current_date] = len(tradable_stocks)
            
            # If no tradable stocks, continue to next day
            if not tradable_stocks:
                continue
                
            # Calculate S-scores for tradable stocks
            s_scores = calculate_all_s_scores(residuals, ou_params, tradable_stocks)
            
            # Generate signals for the last day
            last_day_scores = s_scores.iloc[-1:]
            signals = generate_signals(last_day_scores, s_threshold)
            
            # Calculate portfolio weights
            weights = calculate_portfolio_weights(pca, K, returns_df.columns)
            
            # Store positions for this day
            for stock in tradable_stocks:
                if stock in signals.columns:
                    signal = signals.iloc[0][stock]  # Get signal for this stock
                    if signal != 0:
                        portfolio_positions.loc[current_date, stock] = signal * weights[stock]
            
            # If we have the next day's returns, calculate strategy return
            if t+1 <= len(returns_df):
                next_day_returns = returns_df.iloc[t]
                day_return = 0
                
                for stock in tradable_stocks:
                    position = portfolio_positions.loc[current_date, stock]
                    if not np.isnan(position) and position != 0:
                        day_return += position * next_day_returns[stock]
                
                strategy_returns.loc[current_date] = day_return
                
        except Exception as e:
            print(f"Error on {current_date}: {e}")
            continue
    
    # Calculate cumulative returns
    strategy_returns = strategy_returns.fillna(0)
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    results = {
        'daily_returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'positions': portfolio_positions,
        'k_values': K_values,
        'tradable_counts': tradable_counts
    }
    
    return results

def calculate_performance_metrics(strategy_returns):
    """
    Calculate performance metrics for the strategy.
    
    Parameters:
    -----------
    strategy_returns : Series
        Daily strategy returns
        
    Returns:
    --------
    dict
        Performance metrics
    """
    # Remove any NaN values
    returns = strategy_returns.fillna(0)
    
    # Calculate metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    annualized_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    
    # Calculate drawdowns
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Calculate win rate
    win_rate = (returns > 0).sum() / len(returns)
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }
    
    return metrics

def plot_results(results, filename='plots/backtest_results.png'):
    """
    Plot backtest results.
    
    Parameters:
    -----------
    results : dict
        Backtest results
    """
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot cumulative returns
    cumulative_returns = results['cumulative_returns']
    cumulative_returns.plot(ax=axes[0], color='blue')
    axes[0].set_title('Cumulative Strategy Returns')
    axes[0].set_ylabel('Cumulative Return')
    axes[0].grid(True)
    
    # Plot number of tradable stocks
    results['tradable_counts'].plot(ax=axes[1], color='green')
    axes[1].set_title('Number of Tradable Stocks')
    axes[1].set_ylabel('Count')
    axes[1].grid(True)
    
    # Plot K values (number of components)
    results['k_values'].plot(ax=axes[2], color='purple')
    axes[2].set_title('Number of PCA Components (K)')
    axes[2].set_ylabel('K')
    axes[2].grid(True)
    
    import os
    os.makedirs("plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()