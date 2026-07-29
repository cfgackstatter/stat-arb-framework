"""
Trading signals and market-neutral position construction (A&L §4).

Mean-reversion rule (eq. 16):
  open long  if s < −s_bo ,  close long  if s > −s_sc
  open short if s > +s_so ,  close short if s < +s_bc

Defaults: s_bo = s_so = 1.25, s_sc = 0.50, s_bc = 0.75.
"""

from __future__ import annotations

import numpy as np


def generate_signals(
    scores: np.ndarray,
    entry: float,
    long_exit: float,
    short_exit: float,
    prev: np.ndarray | None = None,
) -> np.ndarray:
    """
    Map S-scores to position signs in {−1, 0, +1}.

    Parameters
    ----------
    scores : array, shape (N,)
        Latest S-scores (NaN → not tradable → flat).
    entry : float
        Absolute entry threshold (s_bo / s_so).
    long_exit, short_exit : float
        Exit levels (−s_sc and +s_bc in the paper).
    prev : array or None
        Previous day's signals (for hold / exit logic).

    Returns
    -------
    signals : array, shape (N,)
        +1 = long, −1 = short, 0 = flat.
    """
    signals = np.zeros(len(scores))
    valid = ~np.isnan(scores)
    s = np.where(valid, scores, 0.0)

    # Entry
    signals[(s < -entry) & valid] = 1.0   # residual too low  → buy
    signals[(s > entry) & valid] = -1.0   # residual too high → sell

    if prev is None:
        return signals

    # Hold prior position until the exit threshold is crossed
    hold_long = valid & (signals == 0) & (prev > 0) & (s <= long_exit)
    hold_short = valid & (signals == 0) & (prev < 0) & (s >= short_exit)
    signals = np.where(hold_long | hold_short, prev, signals)
    return signals


def build_positions(
    signals: np.ndarray,
    betas: np.ndarray,
    weights: np.ndarray,
    target_gross: float = 2.0,
) -> np.ndarray:
    """
    Bang-bang book: ±$1 of each signaled asset, hedged with eigenportfolios.

    Long asset i  ⇒  +$1 of i and −β_{ij} dollars of factor j (A&L §4).
    In vector form for the whole book:

        pos = signals − Wᵀ (Bᵀ signals)

    Then scale so that gross exposure (|long| + |short|) equals target_gross.
    """
    # betas: (N, k), weights: (k, N), signals: (N,)
    pos = signals - weights.T @ (betas.T @ signals)

    gross = np.abs(pos).sum()
    if gross > 1e-12 and target_gross > 0:
        pos = pos * (target_gross / gross)
    return pos
