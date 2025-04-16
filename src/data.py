"""
Data retrieval and processing for the statistical arbitrage strategy.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime

def fetch_stock_data(symbols, start_date, end_date=None):
    """
    Fetch historical price data for specified symbols.
    
    Parameters:
    -----------
    symbols : list
        List of stock symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, uses current date.
        
    Returns:
    --------
    DataFrame
        Adjusted close prices for all symbols
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    print(f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}")
    
    prices_df = yf.download(symbols, start=start_date, end=end_date, auto_adjust=True)['Close']
    
    # Remove columns (stocks) with more than 10% missing values
    prices_df = prices_df.dropna(axis=1, thresh=len(prices_df) * 0.9)
    
    # Fill any remaining missing values with forward fill then backward fill
    prices_df = prices_df.ffill().bfill()
    
    print(f"Successfully retrieved data for {prices_df.shape[1]} symbols")
    return prices_df

def calculate_returns(prices_df):
    """
    Calculate daily returns from price data.
    
    Parameters:
    -----------
    prices_df : DataFrame
        DataFrame of price data
        
    Returns:
    --------
    DataFrame
        Daily returns
    """
    return prices_df.pct_change().dropna()