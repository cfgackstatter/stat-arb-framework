"""
Main script to run the statistical arbitrage strategy.

This module serves as the entry point to the statistical arbitrage strategy,
handling configuration loading, data retrieval, backtesting, and result visualization.
"""

import os
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from config.settings import *
from src.data import fetch_stock_data, calculate_returns, save_data_to_cache, load_data_from_cache
from src.backtest import StatArbBacktester
from utils.helpers import print_pca_variance, plot_pca_variance


# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('stat_arb.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.handlers.clear()
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command line arguments for the strategy.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Statistical Arbitrage Strategy')
    parser.add_argument('--start_date', type=str, default=START_DATE, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, default=END_DATE, help='End date in YYYY-MM-DD format')
    parser.add_argument('--pca_window', type=int, default=PCA_WINDOW, help='Window size for PCA calculation')
    parser.add_argument('--ou_window', type=int, default=OU_WINDOW, help='Window size for OU process estimation')
    parser.add_argument('--variance_threshold', type=float, default=VARIANCE_THRESHOLD, help='Variance threshold for selecting factors')
    parser.add_argument('--kappa_threshold', type=float, default=KAPPA_THRESHOLD, help='Mean reversion threshold')
    parser.add_argument('--entry_threshold', type=float, default=SIGNAL_THRESHOLDS['entry'], help='S-score entry threshold')
    parser.add_argument('--long_exit', type=float, default=SIGNAL_THRESHOLDS['long_exit'], help='S-score long exit threshold')
    parser.add_argument('--short_exit', type=float, default=SIGNAL_THRESHOLDS['short_exit'], help='S-score short exit threshold')
    parser.add_argument('--use_cache', action='store_true', help='Use cached data if available')
    
    return parser.parse_args()


def main():
    """
    Run the statistical arbitrage strategy.
    
    This function:
    1. Loads configuration and parses arguments
    2. Fetches historical price data
    3. Calculates returns
    4. Runs the backtest
    5. Calculates and displays performance metrics
    6. Generates visualization plots
    """
    logger.info("Statistical Arbitrage Strategy")
    logger.info("==============================")
    
    # Parse arguments
    args = parse_arguments()
    
    # Create configuration dictionary
    config = {
        'PCA_WINDOW': args.pca_window,
        'OU_WINDOW': args.ou_window,
        'N_COMPONENTS': N_COMPONENTS,
        'VARIANCE_THRESHOLD': args.variance_threshold,
        'KAPPA_THRESHOLD': args.kappa_threshold,
        'SIGNAL_THRESHOLDS': {
            'entry': args.entry_threshold,
            'long_exit': args.long_exit,
            'short_exit': args.short_exit,
        },
    }

    # Create output directories
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Determine data source
    cache_file = f"prices_{args.start_date}_{args.end_date}.pkl"
    prices_df = None
    
    if USE_DATA_CACHE:
        logger.info("Checking for cached data...")
        # Try to load from cache first
        prices_df = load_data_from_cache(cache_file, CACHE_DIR)

        if prices_df is not None and not prices_df.empty:
            logger.info(f"Loaded cached data with shape {prices_df.shape}")
        else:
            logger.info("No cached data found, fetching from API...")
            # Fetch fresh data (bypass in-memory cache)
            prices_df = fetch_stock_data(
                SYMBOLS, 
                args.start_date, 
                args.end_date, 
                use_cache=False  # Bypass lru_cache
            )

            if prices_df is not None and not prices_df.empty:
                # Save to disk cache for future runs
                save_data_to_cache(prices_df, cache_file, CACHE_DIR)
                logger.info(f"Saved new data to cache: {os.path.join(CACHE_DIR, cache_file)}")
            else:
                logger.error("Failed to fetch data from API")
                return
    else:
        logger.info("Cache disabled, fetching fresh data...")
        prices_df = fetch_stock_data(SYMBOLS, args.start_date, args.end_date, use_cache=False)
    
    # Calculate returns with optional winsorization
    returns_df = calculate_returns(prices_df, method='pct_change', winsorize_pct=1.0)
    logger.info(f"Data shape: {returns_df.shape}")

    # Save returns for reference
    returns_df.to_pickle("results/returns.pkl")
    
    # Create backtester instance
    backtester = StatArbBacktester(config)
    
    # Run backtest
    results = backtester.run_backtest(returns_df)
    
    # Calculate performance metrics
    metrics = backtester.calculate_performance_metrics()
    
    # Print results
    logger.info("\nPerformance Metrics:")
    logger.info(f"Total Return: {metrics['total_return']:.2%}")
    logger.info(f"Annualized Return: {metrics['annualized_return']:.2%}")
    logger.info(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
    logger.info(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
    logger.info(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")

    print("\nPerformance Metrics:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    
    # Save metrics to file
    pd.Series(metrics).to_csv("results/performance_metrics.csv")
    
    # Save results for further analysis
    results['daily_returns'].to_pickle("results/daily_returns.pkl")
    results['cumulative_returns'].to_pickle("results/cumulative_returns.pkl")
    results['positions'].to_pickle("results/positions.pkl")
    
    # Generate plots
    backtester.plot_results(filename="plots/backtest_results.png")
    backtester.plot_exposure_analysis(filename="plots/exposure_analysis.png")
    
    logger.info("Strategy execution completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error running strategy: {str(e)}")