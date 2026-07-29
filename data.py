"""Market data: download prices and compute returns."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def load_prices(symbols: list[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance.

    Returns a DataFrame (dates × symbols). Columns with >10% missing data
    are dropped; remaining gaps are forward/back-filled.
    """
    logger.info('Downloading %d symbols (%s → %s)', len(symbols), start, end or 'today')
    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError('No data returned from yfinance')

    # yfinance returns MultiIndex columns when multiple tickers are requested
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw['Close']
    elif 'Close' in raw.columns:
        prices = raw['Close']
    else:
        prices = raw

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=symbols[0])

    min_obs = int(len(prices) * 0.9)
    prices = prices.dropna(axis=1, thresh=min_obs).ffill().bfill()
    logger.info(
        'Loaded %d assets, %s → %s',
        prices.shape[1],
        prices.index[0].date(),
        prices.index[-1].date(),
    )
    return prices


def calculate_returns(
    prices: pd.DataFrame,
    winsorize_pct: Optional[float] = 1.0,
) -> pd.DataFrame:
    """Daily simple returns, optionally winsorized per column."""
    returns = prices.pct_change().dropna()
    if winsorize_pct and 0 < winsorize_pct < 50:
        lower = returns.quantile(winsorize_pct / 100)
        upper = returns.quantile(1 - winsorize_pct / 100)
        returns = returns.clip(lower=lower, upper=upper, axis=1)
    return returns
