"""
PCA implementation for statistical arbitrage.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def perform_pca(returns_df, n_components):
    """
    Perform PCA on standardized returns.
    
    Parameters:
    -----------
    returns_df : DataFrame
        DataFrame of returns
    n_components : int
        Number of components to extract
        
    Returns:
    --------
    pca : PCA object
        Fitted PCA object
    standardized_returns : DataFrame
        Standardized returns
    """
    # Standardize returns
    standardized_returns = (returns_df - returns_df.mean()) / returns_df.std()
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(standardized_returns)
    
    return pca, standardized_returns

def get_factor_weights(pca, columns, std_devs):
    """
    Get factor weights from PCA components.
    
    Parameters:
    -----------
    pca : PCA object
        Fitted PCA object
    columns : list
        Column names
    std_devs : Series
        Standard deviations of returns
        
    Returns:
    --------
    DataFrame
        Factor weights
    """
    # Divide by standard deviation to get the factor weights
    return pd.DataFrame(pca.components_, columns=columns) / std_devs

def calculate_factor_returns(returns_df, factor_weights):
    """
    Calculate factor returns by projecting returns onto factor weights.
    
    Parameters:
    -----------
    returns_df : DataFrame
        DataFrame of returns
    factor_weights : DataFrame
        Factor weights from PCA
        
    Returns:
    --------
    DataFrame
        Factor returns
    """
    # Matrix multiplication of returns and factor weights
    factor_returns = pd.DataFrame(
        np.dot(returns_df, factor_weights.transpose()),
        index=returns_df.index
    )
    
    return factor_returns

def get_optimal_components(pca, variance_threshold):
    """
    Determine optimal number of components based on explained variance.
    
    Parameters:
    -----------
    pca : PCA object
        Fitted PCA object
    variance_threshold : float
        Threshold of explained variance
        
    Returns:
    --------
    int
        Number of components to keep
    """
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    k = np.argmax(cumulative_variance >= variance_threshold) + 1
    return k