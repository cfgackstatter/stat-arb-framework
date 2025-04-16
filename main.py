"""
Main script to run the statistical arbitrage strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime

from config.settings import *
from src.data import fetch_stock_data, calculate_returns
from src.backtest import rolling_backtest, calculate_performance_metrics, plot_results
from utils.helpers import print_pca_variance, plot_pca_variance

def main():
    """
    Run the statistical arbitrage strategy.
    """
    print("Statistical Arbitrage Strategy")
    print("==============================")
    
    # Load configuration
    config = {
        'PCA_WINDOW': PCA_WINDOW,
        'OU_WINDOW': OU_WINDOW,
        'N_COMPONENTS': N_COMPONENTS,
        'VARIANCE_THRESHOLD': VARIANCE_THRESHOLD,
        'KAPPA_THRESHOLD': KAPPA_THRESHOLD,
        'S_THRESHOLD': S_THRESHOLD
    }
    
    # Fetch historical data
    prices_df = fetch_stock_data(SYMBOLS, START_DATE, END_DATE)
    
    if prices_df is None or prices_df.empty:
        print("Failed to fetch data. Exiting.")
        return
    
    # Calculate returns
    returns_df = calculate_returns(prices_df)
    print(f"Data shape: {returns_df.shape}")
    
    # Run backtest
    results = rolling_backtest(returns_df, config)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(results['daily_returns'])
    
    # Print results
    print("\nPerformance Metrics:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    
    # Plot results
    plot_results(results)

if __name__ == "__main__":
    main()