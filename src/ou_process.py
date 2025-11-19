"""
Ornstein-Uhlenbeck process fitting and analysis for mean-reverting time series.

This module provides functions to fit the Ornstein-Uhlenbeck process to residual time series,
identify mean-reverting assets, and calculate related parameters.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Configure logger
logger = logging.getLogger(__name__)


def fit_ou_process(
        series: pd.Series, min_samples: int = 20, confidence_level: float = 0.05
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Fit a discrete Ornstein-Uhlenbeck process to a cumulative residual series.

    Following Avellaneda & Lee (2010), this fits an AR(1) model to the cumulative
    residuals to estimate OU parameters.
    
    Args:
        series: Time series of residuals
        min_samples: Minimum number of samples required for fitting
        confidence_level: P-value threshold for statistical significance
        
    Returns:
        Tuple containing:
            kappa: Mean reversion speed (annualized)
            m: Long-term mean
            sigma_eq: Equilibrium volatility
    """
    # Create auxiliary process X_k = sum_{j=1}^k epsilon_j as in Avellaneda-Lee
    cum_series = series.cumsum()

    # Prepare data for regression (y_t = a + b*y_{t-1} + ε_t)
    y = cum_series.values[1:]
    x = cum_series.values[:-1]
    n = len(y)

    if n < min_samples:
        logger.warning(f"Insufficient data points ({n}) for OU process fitting")
        return None, None, None

    try:    
        # Add constant to x for intercept
        X = sm.add_constant(x)
    
        # Fit OLS model
        model = sm.OLS(y, X).fit()
    
        # Extract parameters
        a, b = model.params  # a is intercept, b is slope

        # Check validity conditions:
        # 1. 0 < b < 1 for stationarity
        # 2. Slope coefficient must be statistically significant
        if not (0 < b < 1):
            logger.debug(f"Non-stationary process: AR coefficient = {b}")
            return None, None, None
    
        if model.pvalues[1] > confidence_level:
            logger.debug(f"Statistically insignificant AR coefficient: p-value = {model.pvalues[1]:.4f}")
            return None, None, None
        
        # Calculate OU parameters from regression coefficients
        dt = 1/252  # Daily time interval (annualized)
        kappa = -np.log(b) / dt  # Annualized mean reversion speed
        m = a / (1 - b)       # Long-term mean

        # Calculate equilibrium volatility
        # Following Avellaneda-Lee method for sigma calculation
        residual_variance = np.sum(model.resid**2) / (n - 1)
        sigma_eq = np.sqrt(residual_variance / (1 - b**2))  # Equilibrium standard error
        
        return kappa, m, sigma_eq
    
    except Exception as e:
        logger.error(f"Error fitting OU process: {str(e)}")
        return None, None, None
    

def get_tradable_stocks(
    residuals: pd.DataFrame,
    kappa_threshold: float,
    min_samples: int = 20
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """
    Identify stocks with sufficient mean reversion characteristics.
    
    Args:
        residuals: DataFrame of residuals for each stock
        kappa_threshold: Mean reversion speed threshold (higher = faster mean reversion)
        min_samples: Minimum samples required for OU fitting
        
    Returns:
        Tuple containing:
            Dictionary of OU parameters for tradable stocks
            List of tradable stock symbols
    """
    ou_params = {}
    tradable_stocks = []
    
    for stock in residuals.columns:
        # Skip columns with too many NaN values
        if residuals[stock].isna().sum() > len(residuals) * 0.2:
            logger.debug(f"Skipping {stock} due to excessive missing values")
            continue

        # Fit OU process to cumulative residuals
        kappa, m, sigma_eq = fit_ou_process(residuals[stock], min_samples)
        
        if kappa is not None and kappa > kappa_threshold:
            ou_params[stock] = {'kappa': kappa, 'm': m, 'sigma_eq': sigma_eq}
            tradable_stocks.append(stock)
            logger.debug(f"{stock} is tradable: kappa={kappa:.2f}, half-life={np.log(2)/kappa*252:.1f} days")

    logger.info(f"Found {len(tradable_stocks)} tradable stocks out of {len(residuals.columns)}")   
    return ou_params, tradable_stocks


def test_stationarity(series: pd.Series, significance: float = 0.05) -> Tuple[bool, float]:
    """
    Test time series for stationarity using Augmented Dickey-Fuller test.
    
    Args:
        series: Time series to test
        significance: Significance level (default: 0.05)
        
    Returns:
        Tuple containing:
            Boolean indicating stationarity
            P-value from the test
    """
    if len(series) < 20:
        return False, 1.0
    
    try:
        result = adfuller(series.dropna())
        p_value = result[1]
        
        # Explicitly convert to Python scalar types for type safety
        p_value_float = float(p_value)
        is_stationary = bool(p_value_float < significance)
        
        return is_stationary, p_value_float
    
    except Exception as e:
        logger.warning(f"Error in stationarity test: {str(e)}")
        return False, 1.0