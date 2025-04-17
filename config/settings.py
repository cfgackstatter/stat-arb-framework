"""
Configuration settings for the statistical arbitrage strategy.

This module defines all configuration parameters used across the strategy,
including data sources, model parameters, and backtest settings.
"""

from typing import List, Optional, Dict, Any
import os
from datetime import datetime, timedelta

# Data parameters
SYMBOLS: List[str] = [
    'SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'AGG', 'LQD', 'HYG', 'EMB', 'GLD', 'SLV',
    'VNQ', 'VWO', 'VEA', 'TIP', 'XLF', 'XLK', 'XLE', 'XLY', 'XLP', 'XLI', 'XLB',
    'XLU', 'XLC', 'XBI', 'EWJ', 'EWZ', 'EWA', 'FXI', 'GDX', 'TLT'
]

# Time period settings
START_DATE: str = '2024-01-01'
END_DATE: Optional[str] = '2025-03-31'  # Set to None to use current date

# Model parameters
PCA_WINDOW: int = 252       # 1 year of trading days for PCA calculation
OU_WINDOW: int = 60         # Days for OU process estimation
N_COMPONENTS: int = None      # Number of principal components to extract
VARIANCE_THRESHOLD: float = 0.85  # Variance threshold for selecting factors
KAPPA_THRESHOLD: float = 8.4     # Mean reversion threshold (252/30)

# Signal and exit thresholds
SIGNAL_THRESHOLDS: Dict[str, float] = {
    'entry': 1.25,   # Enter position when |s-score| exceeds this value
    'long_exit': -0.5,  # Exit long position when s-score rises above this
    'short_exit': 0.75   # Exit short position when s-score falls below this
}

# Data processing settings
WINSORIZE_PCT: float = 1.0  # Winsorize returns at this percentile to handle outliers
USE_LOG_RETURNS: bool = False  # Whether to use log returns instead of percentage returns

# Cache settings
USE_DATA_CACHE: bool = True  # Whether to cache data to disk
CACHE_DIR: str = 'data_cache'  # Directory for cached data

# Output settings
RESULTS_DIR: str = 'results'  # Directory for results
PLOTS_DIR: str = 'plots'      # Directory for plots

# Create necessary directories
for directory in [CACHE_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Environment settings
DEBUG_MODE: bool = False  # Enable detailed logging
RANDOM_SEED: int = 42     # Random seed for reproducibility

# Strategy customization
ALLOW_SHORTING: bool = True  # Whether shorting is allowed
MAX_POSITION_SIZE: float = 0.1  # Maximum position size as fraction of portfolio
MAX_LEVERAGE: float = 2.0    # Maximum allowable leverage

# Default configuration dictionary
DEFAULT_CONFIG: Dict[str, Any] = {
    'PCA_WINDOW': PCA_WINDOW,
    'OU_WINDOW': OU_WINDOW, 
    'N_COMPONENTS': N_COMPONENTS,
    'VARIANCE_THRESHOLD': VARIANCE_THRESHOLD,
    'KAPPA_THRESHOLD': KAPPA_THRESHOLD,
    'EXIT_THRESHOLDS': {
        'entry': SIGNAL_THRESHOLDS['entry'],
        'long_exit': SIGNAL_THRESHOLDS['long_exit'],
        'short_exit': SIGNAL_THRESHOLDS['short_exit']
    },
    'WINSORIZE_PCT': WINSORIZE_PCT,
    'USE_LOG_RETURNS': USE_LOG_RETURNS
}