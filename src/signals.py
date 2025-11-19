"""
Signal generation and portfolio construction.

This module handles the generation of trading signals from S-scores
and the construction of portfolios based on statistical arbitrage principles.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Configure logger
logger = logging.getLogger(__name__)


def _safe_float_conversion(value: Any) -> float:
    """
    Safely convert a pandas Scalar or any numeric type to Python float.
    
    Args:
        value: Value to convert (could be pandas Scalar, numpy type, or Python numeric)
    
    Returns:
        Python float value
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    # For pandas scalars, try conversion
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
    

def calculate_all_s_scores(
    residuals: pd.DataFrame,
    ou_params: Dict[str, Dict[str, float]],
    tradable_stocks: List[str],
    demean: bool = True
) -> pd.DataFrame:
    """
    Calculate S-scores for all tradable stocks with optional demeaning.
    
    Args:
        residuals: DataFrame of residuals for each stock
        ou_params: Dictionary of OU parameters
        tradable_stocks: List of tradable stock symbols
        demean: Whether to demean the long-term means across stocks
    
    Returns:
        DataFrame of S-scores for tradable stocks
    """
    if not tradable_stocks:
        logger.warning("No tradable stocks provided for S-score calculation")
        return pd.DataFrame()
    
    # Demean m across tradable stocks if requested
    if demean and tradable_stocks:
        m_vec = np.array([ou_params[stock]['m'] for stock in tradable_stocks])
        m_mean = float(np.nanmean(m_vec))  # Convert to Python float for type safety
        logger.debug(f"Demeaning long-term means, average: {m_mean:.6f}")
        for stock in tradable_stocks:
            ou_params[stock]['m'] -= m_mean
    
    # Using cumulative sum of residuals to match OU process fitting
    cum_residuals = residuals[tradable_stocks].cumsum()
    
    # Calculate S-score = (X - m) / sigma_eq
    # Use pandas broadcasting operations for type safety
    m_series = pd.Series({stock: ou_params[stock]['m'] for stock in tradable_stocks})
    sigma_eq_series = pd.Series({stock: ou_params[stock]['sigma_eq'] for stock in tradable_stocks})
    
    s_scores = cum_residuals.sub(m_series, axis=1).div(sigma_eq_series, axis=1)
    
    return s_scores


def generate_signals(
    s_scores: pd.DataFrame,
    entry_threshold: float,
    prev_signals: Optional[pd.DataFrame] = None,
    exit_thresholds: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Generate trading signals based on S-scores with exit thresholds.
    
    Args:
        s_scores: DataFrame of S-scores for tradable stocks
        entry_threshold: Threshold for generating entry signals
        prev_signals: Previous day's signals (optional)
        exit_thresholds: Dict with 'long_exit' and 'short_exit' thresholds
    
    Returns:
        DataFrame of trading signals (-1 for short, 0 for no position, 1 for long)
    """
    # Initialize signals dataframe with zeros
    signals = pd.DataFrame(0, index=s_scores.index, columns=s_scores.columns)
    
    # Generate entry signals
    signals[s_scores < -entry_threshold] = 1   # Long
    signals[s_scores > entry_threshold] = -1    # Short
    
    # Apply signal continuity and exit conditions based on previous signals
    if prev_signals is not None and not prev_signals.empty and exit_thresholds is not None:
        for col in s_scores.columns:
            if col in prev_signals.columns:
                # Get previous signal using .at for label-based scalar access
                # Use safe conversion to handle pandas Scalar types
                prev_signal_raw = prev_signals.at[prev_signals.index[0], col]
                prev_signal = _safe_float_conversion(prev_signal_raw)
                
                # Get current signal value using .at
                current_signal_raw = signals.at[signals.index[0], col]
                current_signal = _safe_float_conversion(current_signal_raw)
                
                # If no new signal but had a position previously
                if current_signal == 0 and prev_signal != 0:
                    # Check exit conditions
                    s_score_raw = s_scores.at[s_scores.index[0], col]
                    s_score = _safe_float_conversion(s_score_raw)
                    
                    # For long positions
                    if prev_signal > 0:
                        if s_score <= exit_thresholds['long_exit']:
                            # Continue long position
                            signals.at[signals.index[0], col] = prev_signal
                        # Otherwise leave as 0 (exit position)
                    
                    # For short positions
                    elif prev_signal < 0:
                        if s_score >= exit_thresholds['short_exit']:
                            # Continue short position
                            signals.at[signals.index[0], col] = prev_signal
                        # Otherwise leave as 0 (exit position)
    
    return signals


def construct_beta_matrix(
    factor_models: Dict[str, Any],
    factor_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Construct beta matrix from regression models.
    
    Args:
        factor_models: Dictionary of OLS models {stock: model}
        factor_names: Optional list of factor names for columns
    
    Returns:
        DataFrame of beta coefficients (stocks x factors)
    """
    if not factor_models:
        logger.warning("No factor models provided for beta matrix construction")
        return pd.DataFrame()
    
    # Extract beta coefficients (excluding intercept) for each stock
    beta_data = {}
    for stock, model in factor_models.items():
        try:
            # Extract coefficients excluding intercept
            betas = model.params.drop('const')
            beta_data[stock] = betas
        except Exception as e:
            logger.warning(f"Error extracting betas for {stock}: {str(e)}")
    
    # Convert to DataFrame
    beta_matrix = pd.DataFrame(beta_data).T
    
    # Set factor names if provided
    if factor_names is not None and len(factor_names) == beta_matrix.shape[1]:
        beta_matrix.columns = factor_names
    
    return beta_matrix