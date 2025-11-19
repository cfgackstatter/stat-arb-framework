"""
Backtesting functionality for the statistical arbitrage strategy.

This module handles backtesting, performance evaluation, and visualization
of the statistical arbitrage strategy.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime, timedelta

from src.pca_model import perform_pca, get_factor_weights, calculate_factor_returns, get_optimal_components
from src.ou_process import get_tradable_stocks
from src.signals import calculate_all_s_scores, generate_signals, construct_beta_matrix

# Configure logger
logger = logging.getLogger(__name__)


class StatArbBacktester:
    """
    Statistical arbitrage strategy backtester with rolling window approach.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the backtester with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.pca_window = config['PCA_WINDOW']
        self.ou_window = config['OU_WINDOW']
        self.n_components = config['N_COMPONENTS']
        self.variance_threshold = config['VARIANCE_THRESHOLD']
        self.kappa_threshold = config['KAPPA_THRESHOLD']
        self.previous_signals = None

        # Signal thresholds
        self.signal_thresholds = config.get(
            'SIGNAL_THRESHOLDS',
            {
                'entry': 1.25,
                'long_exit': -0.5,
                'short_exit': 0.75
            }
        )

        # Derived parameters for easier access
        self.entry_threshold = self.signal_thresholds['entry']
        self.exit_thresholds = {
            'long_exit': self.signal_thresholds['long_exit'],
            'short_exit': self.signal_thresholds['short_exit']
        }
        
        # Storage for results
        self.results = None
        
        # Initialize logger
        self.logger = logger
    
    def run_backtest(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the statistical arbitrage strategy backtest.
        
        Args:
            returns_df: DataFrame of daily returns for all assets
            
        Returns:
            Dictionary containing backtest results
        """
        # Validate input data
        if returns_df is None or returns_df.empty:
            self.logger.error("Empty or invalid returns DataFrame provided")
            raise ValueError("Returns DataFrame cannot be empty")
        
        if len(returns_df) <= self.pca_window:
            self.logger.error(f"Insufficient data for backtest: {len(returns_df)} days available, {self.pca_window} required")
            raise ValueError(f"Not enough data for backtest: need at least {self.pca_window+1} observations")
        
        # Initialize results containers
        strategy_returns = pd.Series(index=returns_df.index[self.pca_window:], dtype=float)
        portfolio_positions = pd.DataFrame(0.0, index=returns_df.index[self.pca_window:], columns=returns_df.columns)
        k_values = pd.Series(index=returns_df.index[self.pca_window:], dtype=float)
        tradable_counts = pd.Series(index=returns_df.index[self.pca_window:], dtype=int)

        # Store asset universe
        assets = returns_df.columns.tolist()
        self.logger.info(f"Starting backtest with {len(assets)} assets and {len(returns_df)} days of data")
        
        # Use sequential processing to run backtest
        results = self._run_backtest_internal(returns_df)

        self.results = results
        return results
    
    def _run_backtest_internal(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest sequentially, processing one day at a time.
        
        Args:
            returns_df: DataFrame of daily returns
            
        Returns:
            Dictionary of backtest results
        """
        # Initialize results containers
        strategy_returns = pd.Series(index=returns_df.index[self.pca_window:], dtype=float)
        portfolio_positions = pd.DataFrame(0.0, index=returns_df.index[self.pca_window:], columns=returns_df.columns)
        k_values = pd.Series(index=returns_df.index[self.pca_window:], dtype=float)
        tradable_counts = pd.Series(index=returns_df.index[self.pca_window:], dtype=int)
        
        start_time = time.time()

        # Rolling window backtest
        for t in range(self.pca_window, len(returns_df)):
            current_date = returns_df.index[t]
            self.logger.info(f"Processing date: {current_date}")
            
            try:
                # Process this day
                day_result = self._process_single_day(returns_df, t)
                
                # Store results for this day
                if day_result:
                    k_values.loc[current_date] = day_result['k']
                    tradable_counts.loc[current_date] = day_result['tradable_count']
                    portfolio_positions.loc[current_date] = day_result['positions']
                    
                    # If we have next day's returns, calculate strategy return
                    if t+1 < len(returns_df):
                        next_day_returns = returns_df.iloc[t+1]
                        day_return = day_result['positions'] @ next_day_returns
                        strategy_returns.loc[current_date] = day_return
                        self.logger.debug(f"Portfolio return for {current_date}: {day_return:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error on {current_date}: {str(e)}")
                continue
                
        end_time = time.time()
        self.logger.info(f"Backtest completed in {end_time - start_time:.2f} seconds")
        
        # Calculate cumulative returns
        strategy_returns = strategy_returns.fillna(0)
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        results = {
            'daily_returns': strategy_returns,
            'cumulative_returns': cumulative_returns,
            'positions': portfolio_positions,
            'k_values': k_values,
            'tradable_counts': tradable_counts
        }
        
        return results
    
    def _process_single_day(self, returns_df: pd.DataFrame, t: int) -> Optional[Dict[str, Any]]:
        """
        Process a single day in the backtest.
        
        Args:
            returns_df: DataFrame of daily returns
            t: Index of the current day
            
        Returns:
            Dictionary with day's processing results or None if error/no tradable stocks
        """
        current_date = returns_df.index[t]
        
        try:
            # Get training data (PCA window)
            train_returns = returns_df.iloc[t-self.pca_window:t]
            
            # Perform PCA
            pca, standardized_returns = perform_pca(train_returns, self.n_components)
            
            # Determine optimal K components
            K = get_optimal_components(pca, self.variance_threshold)
            
            # Get last ou_window days for OU process fitting
            ou_estimation_returns = train_returns.iloc[-self.ou_window:]
            
            # Calculate factor returns
            factor_weights = get_factor_weights(pca, train_returns.columns, train_returns.std())
            factor_returns = calculate_factor_returns(ou_estimation_returns, factor_weights)
            
            # Fit regression for each stock to get residuals
            residuals = pd.DataFrame(index=ou_estimation_returns.index, columns=ou_estimation_returns.columns)
            factor_models = {}
            
            for stock in ou_estimation_returns.columns:
                y = ou_estimation_returns[stock]
                X = factor_returns.iloc[:, :K]  # Use only first K factors
                X = sm.add_constant(X)
                
                model = sm.OLS(y, X).fit()
                factor_models[stock] = model
                residuals[stock] = model.resid
            
            # Identify tradable stocks
            ou_params, tradable_stocks = get_tradable_stocks(residuals, self.kappa_threshold)
            
            # If no tradable stocks, return None
            if not tradable_stocks:
                return {
                    'k': K,
                    'tradable_count': 0,
                    'positions': pd.Series(0, index=returns_df.columns)
                }
            
            # Calculate S-scores for tradable stocks
            s_scores = calculate_all_s_scores(residuals, ou_params, tradable_stocks)
            
            # Generate signals for the last day
            last_day_scores = s_scores.iloc[-1:] if not s_scores.empty else pd.DataFrame()

            signals = generate_signals(
                last_day_scores,
                entry_threshold=self.entry_threshold,
                prev_signals=self.previous_signals,
                exit_thresholds=self.exit_thresholds
            )

            # Save current signals for the next iteration
            self.previous_signals = signals
            
            # Construct beta matrix from regression models
            beta_matrix = construct_beta_matrix(factor_models)
            
            # Initialize positions
            positions = pd.Series(0.0, index=returns_df.columns)
            
            # Process signals and calculate positions
            for stock in tradable_stocks:
                signal = 0
                if not signals.empty and stock in signals.columns:
                    signal = signals.iloc[0][stock]
                
                if signal != 0:
                    # Get betas for first K factors
                    stock_betas = beta_matrix.loc[stock].iloc[:K].values
                    
                    # 1. Long $1 of the stock with signal
                    positions[stock] += signal * 1
                    
                    # 2. Calculate and apply hedge positions
                    factor_weights_subset = factor_weights.iloc[:K]
                    hedge = signal * stock_betas @ factor_weights_subset
                    
                    for i, col in enumerate(factor_weights_subset.columns):
                        if i < len(stock_betas) and col in positions.index:
                            positions[col] -= hedge.iloc[i]
            
            # Return results for this day
            return {
                'k': K,
                'tradable_count': len(tradable_stocks),
                'positions': positions,
                'tradable_stocks': tradable_stocks
            }
            
        except Exception as e:
            self.logger.error(f"Error processing day {current_date}: {str(e)}")
            return None
        
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for the backtest results.
        
        Returns:
            Dictionary of performance metrics
        """
        if self.results is None:
            self.logger.error("No backtest results available for performance calculation")
            return {}
        
        returns = self.results['daily_returns'].fillna(0)
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        annualized_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Additional metrics
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
        profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else float('inf')
        
        # Calculate calmar ratio
        calmar_ratio = -annualized_return / max_drawdown if max_drawdown != 0 else float('inf')
        
        # Monthly returns analysis
        if isinstance(returns.index[0], pd.Timestamp):
            monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
            best_month = monthly_returns.max()
            worst_month = monthly_returns.min()
            avg_monthly_return = monthly_returns.mean()
            monthly_win_rate = (monthly_returns > 0).mean()
        else:
            best_month = worst_month = avg_monthly_return = monthly_win_rate = np.nan
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'best_month': best_month,
            'worst_month': worst_month,
            'avg_monthly_return': avg_monthly_return,
            'monthly_win_rate': monthly_win_rate
        }
        
        return metrics

    def plot_results(self, filename: str = 'plots/backtest_results.png') -> None:
        """
        Plot backtest results and save to file.
        
        Args:
            filename: Path to save the plot
        """
        if self.results is None:
            self.logger.error("No backtest results available for plotting")
            return
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot cumulative returns
        cumulative_returns = self.results['cumulative_returns']
        cumulative_returns.plot(ax=axes[0], color='blue', linewidth=2)
        axes[0].set_title('Cumulative Strategy Returns', fontsize=14)
        axes[0].set_ylabel('Cumulative Return', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Add horizontal line at y=1 (initial capital)
        axes[0].axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        # Highlight drawdowns
        if len(cumulative_returns) > 0:
            rolling_max = cumulative_returns.cummax()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            axes[0].fill_between(
                cumulative_returns.index, 
                cumulative_returns, 
                rolling_max, 
                where=cumulative_returns < rolling_max,
                color='red', 
                alpha=0.3,
                interpolate=True
            )
        
        # Plot number of tradable stocks
        self.results['tradable_counts'].plot(ax=axes[1], color='green', linewidth=2)
        axes[1].set_title('Number of Tradable Stocks', fontsize=14)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Plot K values (number of components)
        self.results['k_values'].plot(ax=axes[2], color='purple', linewidth=2)
        axes[2].set_title('Number of PCA Components (K)', fontsize=14)
        axes[2].set_ylabel('K', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        # Improve layout and save
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"Results plot saved to {filename}")
        
    def plot_exposure_analysis(self, filename: str = 'plots/exposure_analysis.png') -> None:
        """
        Plot portfolio exposure analysis.
        
        Args:
            filename: Path to save the plot
        """
        if self.results is None or 'positions' not in self.results:
            self.logger.error("No position data available for exposure analysis")
            return
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        positions = self.results['positions']
        
        # Calculate exposures over time
        long_exposure = positions.clip(lower=0).sum(axis=1)
        short_exposure = positions.clip(upper=0).sum(axis=1)
        net_exposure = positions.sum(axis=1)
        gross_exposure = long_exposure.abs() + short_exposure.abs()
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        # Plot exposures
        long_exposure.plot(ax=axes[0], color='green', label='Long Exposure')
        short_exposure.plot(ax=axes[0], color='red', label='Short Exposure')
        net_exposure.plot(ax=axes[0], color='blue', label='Net Exposure')
        gross_exposure.plot(ax=axes[0], color='purple', label='Gross Exposure')
        
        axes[0].set_title('Portfolio Exposures Over Time', fontsize=14)
        axes[0].set_ylabel('Exposure', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot asset-level exposures (top N assets by average absolute exposure)
        avg_abs_exposure = positions.abs().mean().sort_values(ascending=False)
        top_assets = avg_abs_exposure.head(10).index
        
        positions[top_assets].plot(ax=axes[1], linewidth=1.5)
        axes[1].set_title('Top Asset Exposures Over Time', fontsize=14)
        axes[1].set_ylabel('Position Size', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Improve layout and save
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"Exposure analysis plot saved to {filename}")


def rolling_backtest(returns_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform a rolling window backtest of the statistical arbitrage strategy.
    
    This is a convenience function that creates a StatArbBacktester instance
    and runs the backtest.
    
    Args:
        returns_df: DataFrame of daily returns for all stocks
        config: Configuration parameters
        
    Returns:
        Dictionary containing backtest results
    """
    backtester = StatArbBacktester(config)
    return backtester.run_backtest(returns_df)


def calculate_performance_metrics(strategy_returns: pd.Series) -> Dict[str, float]:
    """
    Calculate performance metrics for the strategy.
    
    This is a convenience function for backward compatibility.
    
    Args:
        strategy_returns: Series of daily strategy returns
        
    Returns:
        Dictionary of performance metrics
    """
    # Remove any NaN values
    returns = strategy_returns.fillna(0)
    
    # Calculate metrics
    prod_result: float = (1 + returns).prod()  # type: ignore[assignment]
    total_return = prod_result - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    annualized_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    
    # Calculate drawdowns
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Additional metrics
    win_rate = (returns > 0).mean()
    avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
    avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
    profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else float('inf')
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor
    }
    
    return metrics