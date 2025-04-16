"""
Ornstein-Uhlenbeck process fitting for residuals.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

def fit_ou_process(series):
    """
    Fit a discrete Ornstein-Uhlenbeck process to time series.
    
    Parameters:
    -----------
    series : Series
        Time series to fit
        
    Returns:
    --------
    kappa : float
        Mean reversion speed (annualized)
    m : float
        Long-term mean
    sigma_eq : float
        Equilibrium volatility
    """
    # Prepare data for regression (y_t = a + b*y_{t-1} + ε_t)
    y = series.values[1:]
    x = series.values[:-1]
    n = len(y)

    valid = False
        
    # Add constant to x for intercept
    X = sm.add_constant(x)
    
    # Fit OLS model
    model = sm.OLS(y, X).fit()
    
    # Extract parameters
    a, b = model.params  # a is intercept, b is slope

    # Check validity
    if 0 < b < 1:# and model.pvalues[1] < 0.05:
        valid = True
    
    if valid:
        # Calculate OU parameters from regression coefficients
        kappa = -np.log(b) * 252  # Annualized mean reversion speed
        m = a / (1 - b)       # Long-term mean
        sigma_eq = np.sqrt(np.sum(model.resid**2) / (n - 1) / (1 - b**2))
        
        return kappa, m, sigma_eq
    else:
        return None, None, None
    
def calculate_s_score(series, m, sigma_eq):
    """
    Calculate S-score based on OU parameters.
    
    Parameters:
    -----------
    series : Series
        Time series
    kappa : float
        Mean reversion speed
    m : float
        Long-term mean
    sigma_eq : float
        Equilibrium volatility
        
    Returns:
    --------
    Series
        S-scores
    """
    # S-score is the normalized deviation from mean
    s_score = (series - m) / sigma_eq
    
    return s_score

def get_tradable_stocks(residuals, kappa_threshold):
    """
    Identify stocks with sufficient mean reversion.
    
    Parameters:
    -----------
    residuals : DataFrame
        Residuals for each stock
    kappa_threshold : float
        Mean reversion speed threshold
        
    Returns:
    --------
    dict
        Dictionary of OU parameters for tradable stocks
    list
        List of tradable stock symbols
    """
    ou_params = {}
    tradable_stocks = []
    
    for stock in residuals.columns:
        kappa, m, sigma_eq = fit_ou_process(residuals[stock])
        
        if kappa is not None and kappa > kappa_threshold:
            ou_params[stock] = {'kappa': kappa, 'm': m, 'sigma_eq': sigma_eq}
            tradable_stocks.append(stock)
        
    return ou_params, tradable_stocks