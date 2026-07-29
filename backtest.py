"""
Walk-forward backtest of the Avellaneda–Lee PCA strategy.

Each day t:
  - Estimate PCA / residuals / S-scores on data strictly before t
  - Form hedged positions
  - Earn the day-t return, minus proportional trading costs
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config
from model import (
    eigenportfolio_weights,
    fit_ou,
    fit_pca,
    n_components_for_variance,
    residual_regressions,
    s_scores,
)
from portfolio import build_positions, generate_signals

logger = logging.getLogger(__name__)


def process_day(
    returns: np.ndarray,
    t: int,
    cfg: Config,
    prev_signals: np.ndarray | None,
) -> tuple[np.ndarray, int, int, np.ndarray]:
    """
    Run the full A&L pipeline for a single day.

    Uses returns[t − pca_window : t] (exclusive of day t) so there is no
    look-ahead. Returns (positions, K, n_tradable, signals).
    """
    train = returns[t - cfg.pca_window : t]
    components, evr, std = fit_pca(train, cfg.n_components)
    k = n_components_for_variance(evr, cfg.variance_threshold)

    # Residuals / OU are estimated on the most recent ou_window days
    ou_slice = train[-cfg.ou_window :]
    weights = eigenportfolio_weights(components, std)       # (n_comp, N)
    factors = ou_slice @ weights.T                          # (ou_window, n_comp)
    residuals, betas = residual_regressions(ou_slice, factors, k)

    tradable, _, m, sigma_eq = fit_ou(residuals, cfg.kappa_threshold)
    n_assets = returns.shape[1]

    if not tradable.any():
        flat = np.zeros(n_assets)
        return flat, k, 0, flat

    scores = s_scores(residuals, tradable, m, sigma_eq)
    signals = generate_signals(
        scores,
        cfg.entry_threshold,
        cfg.long_exit,
        cfg.short_exit,
        prev_signals,
    )
    positions = build_positions(signals, betas[:, :k], weights[:k], cfg.target_gross)
    return positions, k, int(tradable.sum()), signals


def run_backtest(returns: pd.DataFrame, cfg: Config) -> dict:
    """
    Walk-forward backtest with a warm-up period.

    Prices/returns may begin before ``cfg.start_date``. Estimation always uses
    the trailing ``pca_window`` (and ``ou_window`` inside it). Reported strategy
    returns start on the first session on/after ``cfg.start_date`` that has a
    full warm-up — so a 10y ``start_date`` yields ~10y of P&L, not 10y − 1y.
    """
    if len(returns) <= cfg.pca_window:
        raise ValueError(f'Need more than {cfg.pca_window} days of data')

    ret = returns.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(returns.index)
    n_days, n_assets = ret.shape
    eval_start = pd.Timestamp(cfg.start_date)

    # First index with full PCA history and on/after the evaluation start
    start = cfg.pca_window
    while start < n_days and dates[start] < eval_start:
        start += 1
    if start >= n_days:
        raise ValueError(
            f'No dates on/after start_date={cfg.start_date} with '
            f'{cfg.pca_window} days of warm-up'
        )

    cost = cfg.cost_bps * 1e-4
    n_out = n_days - start

    daily = np.zeros(n_out)
    pos_hist = np.zeros((n_out, n_assets))
    k_hist = np.zeros(n_out)
    tradable_hist = np.zeros(n_out, dtype=int)

    prev_signals: np.ndarray | None = None
    prev_pos = np.zeros(n_assets)

    logger.info(
        'Backtest: %d assets | warm-up → %s | report %s → %s (%d days)',
        n_assets,
        dates[cfg.pca_window - 1].date(),
        dates[start].date(),
        dates[-1].date(),
        n_out,
    )

    for i, t in enumerate(range(start, n_days)):
        try:
            pos, k, n_trad, signals = process_day(ret, t, cfg, prev_signals)
            prev_signals = signals
            k_hist[i] = k
            tradable_hist[i] = n_trad
        except Exception:
            logger.exception('Failed on %s', dates[t])
            pos = np.zeros(n_assets)
            prev_signals = np.zeros(n_assets)

        turnover = np.abs(pos - prev_pos).sum()
        daily[i] = float(pos @ ret[t]) - cost * turnover
        pos_hist[i] = pos
        prev_pos = pos

    idx = dates[start:]
    daily_s = pd.Series(daily, index=idx, name='return')
    return {
        'daily_returns': daily_s,
        'cumulative_returns': (1 + daily_s).cumprod(),
        'positions': pd.DataFrame(pos_hist, index=idx, columns=returns.columns),
        'k_values': pd.Series(k_hist, index=idx),
        'tradable_counts': pd.Series(tradable_hist, index=idx),
    }


def performance_metrics(returns: pd.Series) -> dict[str, float]:
    """Basic performance statistics from a daily return series."""
    r = returns.fillna(0.0)
    total = float((1 + r).prod() - 1)
    ann_ret = (1 + total) ** (252 / len(r)) - 1
    ann_vol = float(r.std() * np.sqrt(252))
    cum = (1 + r).cumprod()
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    return {
        'total_return': total,
        'annualized_return': ann_ret,
        'annualized_volatility': ann_vol,
        'sharpe_ratio': ann_ret / ann_vol if ann_vol else 0.0,
        'max_drawdown': max_dd,
        'win_rate': float((r > 0).mean()),
    }


def plot_results(results: dict, path: Path) -> None:
    """Equity curve, tradable count, and number of PCA factors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1, 1]})

    cum = results['cumulative_returns']
    cum.plot(ax=axes[0], lw=2)
    axes[0].set_yscale('log')
    axes[0].axhline(1.0, color='k', ls='--', alpha=0.4)
    axes[0].fill_between(
        cum.index, cum, cum.cummax(),
        where=cum < cum.cummax(), color='C3', alpha=0.25,
    )
    axes[0].set_title('Cumulative Returns (log scale)')
    axes[0].grid(True, alpha=0.3, which='both')

    results['tradable_counts'].plot(ax=axes[1], color='C2', lw=2)
    axes[1].set_title('Number of Tradable Assets')
    axes[1].grid(True, alpha=0.3)

    results['k_values'].plot(ax=axes[2], color='C4', lw=2)
    axes[2].set_title('PCA Components Retained (K)')
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_exposure(results: dict, path: Path) -> None:
    """Long / short / net / gross exposure and top individual weights."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pos = results['positions']

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    pos.clip(lower=0).sum(axis=1).plot(ax=axes[0], label='Long')
    pos.clip(upper=0).sum(axis=1).plot(ax=axes[0], label='Short')
    pos.sum(axis=1).plot(ax=axes[0], label='Net')
    pos.abs().sum(axis=1).plot(ax=axes[0], label='Gross')
    axes[0].set_title('Portfolio Exposures')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    top = pos.abs().mean().nlargest(8).index
    pos[top].plot(ax=axes[1], lw=1.2)
    axes[1].set_title('Largest Average Absolute Positions')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
