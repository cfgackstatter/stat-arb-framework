"""
Configuration for the Avellaneda & Lee (2010) PCA statistical arbitrage strategy.

Reference: Avellaneda, M. & Lee, J.-H. (2010). Statistical arbitrage in the
U.S. equities market. Quantitative Finance, 10(7), 761–782.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd


# Current Dow Jones Industrial Average constituents (as of 2026).
# A correlated equity panel is a better match for A&L residual mean-reversion
# than a mixed ETF book spanning bonds, commodities, and countries.
DEFAULT_SYMBOLS = [
    'AAPL', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS',
    'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
    'MRK', 'MSFT', 'NKE', 'NVDA', 'PG', 'SHW', 'TRV', 'UNH', 'V', 'WMT',
]


def _default_start_date(years: int = 10) -> str:
    """First calendar day of the *reported* strategy-return window."""
    return (date.today() - timedelta(days=365 * years)).isoformat()


@dataclass
class Config:
    """Strategy and backtest parameters (paper defaults unless noted)."""

    symbols: list[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))

    # Evaluation window: strategy P&L is reported from start_date → end_date.
    # Price history is fetched earlier (see data_start) so the PCA/OU windows
    # are warm before the first reported return — standard walk-forward practice.
    start_date: str = field(default_factory=_default_start_date)
    end_date: Optional[str] = None  # None → today

    # --- Factor model (A&L §2.1, §5.4) ---
    pca_window: int = 252              # days for the correlation matrix
    ou_window: int = 60                # days for residual / OU estimation
    n_components: Optional[int] = None # None → choose via variance_threshold
    variance_threshold: float = 0.55   # keep factors explaining ≥ 55% of variance

    # --- OU tradability (A&L Appendix) ---
    kappa_threshold: float = 8.4       # require κ > 252/30 (half-life ≲ 30 days)

    # --- S-score trading rules (A&L eq. 16) ---
    entry_threshold: float = 1.25      # open long if s < −1.25; short if s > +1.25
    long_exit: float = -0.50           # close long when s > −0.50
    short_exit: float = 0.75           # close short when s < +0.75

    # --- Portfolio / costs ---
    target_gross: float = 2.0          # |long| + |short| notional (paper “1+1”)
    cost_bps: float = 5.0              # one-way proportional cost (~paper slippage)

    winsorize_pct: float = 1.0
    output_dir: Path = field(default_factory=lambda: Path('output'))

    def data_start(self) -> str:
        """
        Download start date = evaluation start minus pca_window trading days.

        Example: 10y of strategy returns with a 252-day PCA window needs about
        11 years of prices (warmup is not counted in reported performance).
        """
        return (
            pd.Timestamp(self.start_date) - pd.offsets.BDay(self.pca_window)
        ).strftime('%Y-%m-%d')


def with_cli_overrides(cfg: Config, args) -> Config:
    """Apply argparse overrides onto a Config copy."""
    return replace(
        cfg,
        start_date=args.start_date,
        end_date=args.end_date,
        pca_window=args.pca_window,
        ou_window=args.ou_window,
        variance_threshold=args.variance_threshold,
        kappa_threshold=args.kappa_threshold,
        entry_threshold=args.entry_threshold,
        long_exit=args.long_exit,
        short_exit=args.short_exit,
        cost_bps=args.cost_bps,
    )
