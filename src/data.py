"""
Data retrieval and processing for the statistical arbitrage strategy.

This module handles fetching historical price data, calculating returns,
and preprocessing data for the statistical arbitrage model.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from functools import lru_cache
import time

# Configure logger
logger = logging.getLogger(__name__)

# Cache for expensive data operations
@lru_cache(maxsize=8)
def fetch_stock_data_cached(symbols_tuple, start_date, end_date, attempts=3):
    """Cached version of fetch_stock_data to improve performance."""
    symbols = list(symbols_tuple)
    return _fetch_stock_data_impl(symbols, start_date, end_date, attempts)

def _fetch_stock_data_impl(
    symbols: List[str], 
    start_date: str, 
    end_date: Optional[str] = None,
    attempts: int = 3
) -> pd.DataFrame:
    """Implementation of fetch_stock_data with retries."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}")
    
    # Retry logic for API failures
    for attempt in range(attempts):
        try:
            prices_df = yf.download(
                symbols, 
                start=start_date, 
                end=end_date, 
                auto_adjust=True, 
                progress=False
            )['Close']
            
            if prices_df.empty:
                logger.warning("Empty dataframe returned from Yahoo Finance")
                if attempt < attempts - 1:
                    logger.info(f"Retrying ({attempt+1}/{attempts})...")
                    continue
                return pd.DataFrame()
            
            return prices_df
            
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{attempts} failed: {str(e)}")
            if attempt < attempts - 1:
                # Wait longer between each retry
                time.sleep(2 * (attempt + 1))
            else:
                logger.error(f"Failed to fetch data after {attempts} attempts")
                return pd.DataFrame()
    
    return pd.DataFrame()

def fetch_stock_data(
    symbols: List[str], 
    start_date: str, 
    end_date: Optional[str] = None,
    attempts: int = 3,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch historical price data for specified symbols with error handling and caching.
    
    Args:
        symbols: List of stock symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format (default: current date)
        attempts: Number of API attempts before giving up
        use_cache: Whether to use function caching
        
    Returns:
        DataFrame of adjusted close prices for all symbols
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    if use_cache:
        symbols_tuple = tuple(sorted(symbols))  # Convert to hashable type for caching
        prices_df = fetch_stock_data_cached(symbols_tuple, start_date, end_date, attempts)
    else:
        prices_df = _fetch_stock_data_impl(symbols, start_date, end_date, attempts)

    if prices_df.empty:
        logger.error("Failed to fetch any data")
        return pd.DataFrame()
        
    # Data preprocessing
    # Remove columns (stocks) with more than 10% missing values
    initial_columns = prices_df.shape[1]
    prices_df = prices_df.dropna(axis=1, thresh=len(prices_df) * 0.9)
    dropped_columns = initial_columns - prices_df.shape[1]

    if dropped_columns > 0:
        logger.warning(f"Dropped {dropped_columns} stocks with >10% missing values")
    
    # Fill any remaining missing values with forward fill then backward fill
    prices_df = prices_df.ffill().bfill()
    
    logger.info(f"Successfully retrieved data for {prices_df.shape[1]} symbols")
    logger.info(f"Data period: {prices_df.index[0]} to {prices_df.index[-1]} ({len(prices_df)} days)")

    return prices_df


def calculate_returns(
    prices_df: pd.DataFrame, 
    method: str = 'pct_change',
    winsorize_pct: Optional[float] = None
) -> pd.DataFrame:
    """
    Calculate daily returns from price data with optional winsorization.
    
    Args:
        prices_df: DataFrame of price data
        method: Method for calculating returns ('pct_change' or 'log')
        winsorize_pct: Percentile for winsorizing outliers (None to disable)
        
    Returns:
        DataFrame of daily returns
    """
    if prices_df.empty:
        return pd.DataFrame()
    
    # Calculate returns using specified method
    if method == 'log':
        returns_df = np.log(prices_df / prices_df.shift(1))
    else:  # Default to pct_change
        returns_df = prices_df.pct_change()

    # Drop rows with NaN values
    returns_df = returns_df.dropna()

    # Optionally winsorize to handle outliers
    if winsorize_pct is not None and 0 < winsorize_pct < 50:
        lower = returns_df.quantile(winsorize_pct/100)
        upper = returns_df.quantile(1 - winsorize_pct/100)
        
        for col in returns_df.columns:
            returns_df[col] = returns_df[col].clip(lower=lower[col], upper=upper[col])
        
        logger.info(f"Winsorized returns at {winsorize_pct}% level")

    return prices_df.pct_change().dropna()


def save_data_to_cache(data: pd.DataFrame, filename: str, cache_dir: str = 'data_cache'):
    """
    Save data to disk cache.
    
    Args:
        data: DataFrame to save
        filename: Name of the file
        cache_dir: Directory for caching
    """
    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, filename)
    
    try:
        data.to_pickle(filepath)
        logger.debug(f"Saved data to cache: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save data to cache: {str(e)}")


def load_data_from_cache(filename: str, cache_dir: str = 'data_cache') -> Optional[pd.DataFrame]:
    """
    Load data from disk cache.
    
    Args:
        filename: Name of the file
        cache_dir: Directory for caching
        
    Returns:
        DataFrame if cache exists, None otherwise
    """
    filepath = os.path.join(cache_dir, filename)
    
    if not os.path.exists(filepath):
        return None
    
    try:
        data = pd.read_pickle(filepath)
        logger.debug(f"Loaded data from cache: {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from cache: {str(e)}")
        return None