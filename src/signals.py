"""
Signal generation and portfolio construction.
"""

import pandas as pd
import numpy as np
from src.ou_process import calculate_s_score

def calculate_all_s_scores(residuals, ou_params, tradable_stocks):
    """
    Calculate S-scores for all tradable stocks.
    
    Parameters:
    -----------
    residuals : DataFrame
        Residuals for each stock
    ou_params : dict
        Dictionary of OU parameters
    tradable_stocks : list
        List of tradable stock symbols
        
    Returns:
    --------
    DataFrame
        S-scores for tradable stocks
    """
    s_scores = pd.DataFrame(index=residuals.index)
    
    for stock in tradable_stocks:
        m = ou_params[stock]['m']
        sigma_eq = ou_params[stock]['sigma_eq']
        
        s_scores[stock] = calculate_s_score(residuals[stock], m, sigma_eq)
    
    return s_scores

def generate_signals(s_scores, s_threshold):
    """
    Generate trading signals based on S-scores.
    
    Parameters:
    -----------
    s_scores : DataFrame
        S-scores for tradable stocks
    s_threshold : float
        Threshold for generating signals
        
    Returns:
    --------
    DataFrame
        Trading signals (-1 for short, 0 for no position, 1 for long)
    """
    signals = pd.DataFrame(0, index=s_scores.index, columns=s_scores.columns)
    
    # Long signal when S-score is below negative threshold
    signals[s_scores < -s_threshold] = 1
    
    # Short signal when S-score is above positive threshold
    signals[s_scores > s_threshold] = -1
    
    return signals

def calculate_portfolio_weights(pca, K, stocks):
    """
    Calculate portfolio weights based on PCA loadings.
    
    Parameters:
    -----------
    pca : PCA object
        Fitted PCA object
    K : int
        Number of components kept for systematic factors
    stocks : list
        List of stock symbols
        
    Returns:
    --------
    Series
        Portfolio weights for each stock
    """
    # Sum loadings from components after K
    portfolio_weights = np.zeros(len(stocks))
    
    for i in range(K, pca.n_components_):
        portfolio_weights += pca.components_[i]
    
    # Normalize weights to sum to 1
    portfolio_weights = portfolio_weights / np.sum(np.abs(portfolio_weights))
    
    return pd.Series(portfolio_weights, index=stocks)