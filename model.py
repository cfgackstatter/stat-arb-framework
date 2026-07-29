"""
PCA factor model, residual extraction, and Ornstein–Uhlenbeck S-scores.

Implements the PCA branch of Avellaneda & Lee (2010):
  1. Eigenportfolios from the return correlation matrix
  2. Idiosyncratic residuals via OLS on those factors
  3. OU / AR(1) fit on cumulative residuals → S-scores
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def fit_pca(returns: np.ndarray, n_components: Optional[int] = None):
    """
    PCA on the sample correlation matrix of returns (A&L §2.1).

    Parameters
    ----------
    returns : array, shape (T, N)
        Asset returns over the estimation window.
    n_components : int or None
        If set, keep only this many leading components.

    Returns
    -------
    components : array, shape (n_comp, N)
        Eigenvectors (rows), ordered by explained variance.
    explained_var_ratio : array, shape (n_comp,)
        Fraction of total variance explained by each component.
    std : array, shape (N,)
        Per-asset volatility used to form eigenportfolio weights.
    """
    std = returns.std(axis=0, ddof=1)
    std = np.where(std < 1e-12, 1.0, std)

    # Standardize → correlation matrix → eigendecomposition
    z = (returns - returns.mean(axis=0)) / std
    corr = (z.T @ z) / max(z.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(corr)

    # eigh returns ascending order; flip to descending
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0.0, None)
    evecs = evecs[:, order]
    explained_var_ratio = evals / evals.sum()
    components = evecs.T  # row j = j-th eigenvector

    if n_components is not None:
        components = components[:n_components]
        explained_var_ratio = explained_var_ratio[:n_components]

    return components, explained_var_ratio, std


def n_components_for_variance(explained_var_ratio: np.ndarray, threshold: float) -> int:
    """Fewest components whose cumulative variance ≥ threshold (A&L §5.4)."""
    return int(np.argmax(np.cumsum(explained_var_ratio) >= threshold) + 1)


def eigenportfolio_weights(components: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Eigenportfolio holdings Q_i^{(j)} = v_i^{(j)} / σ_i (A&L §2.1).

    Parameters
    ----------
    components : array, shape (n_comp, N)
    std : array, shape (N,)

    Returns
    -------
    weights : array, shape (n_comp, N)
    """
    return components / std


def residual_regressions(
    returns: np.ndarray,
    factors: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Regress each asset on the first k factor returns (+ intercept).

    Parameters
    ----------
    returns : array, shape (T, N)
    factors : array, shape (T, n_comp)
    k : int
        Number of leading factors to use.

    Returns
    -------
    residuals : array, shape (T, N)
    betas : array, shape (N, k)
        Factor loadings (intercept excluded).
    """
    t = returns.shape[0]
    x = np.column_stack([np.ones(t), factors[:, :k]])  # (T, k+1)
    coef, _, _, _ = np.linalg.lstsq(x, returns, rcond=None)  # (k+1, N)
    residuals = returns - x @ coef
    betas = coef[1:].T  # (N, k)
    return residuals, betas


def fit_ou(
    residuals: np.ndarray,
    kappa_threshold: float,
    min_samples: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a discrete OU process to cumulative residuals via AR(1) (A&L Appendix).

        X_{k+1} = a + b X_k + ζ_{k+1}
        κ = −log(b) · 252,   m = a / (1 − b),   σ_eq = sqrt(Var(ζ) / (1 − b²))

    An asset is tradable when 0 < b < exp(−κ_min / 252), i.e. κ̂ > kappa_threshold.

    Parameters
    ----------
    residuals : array, shape (T, N)
        Idiosyncratic residual returns.
    kappa_threshold : float
        Minimum annualized mean-reversion speed (paper: 8.4).

    Returns
    -------
    tradable : bool array, shape (N,)
    kappa, m, sigma_eq : float arrays, shape (N,)
        NaN where the model is rejected.
    """
    t, n = residuals.shape
    kappa = np.full(n, np.nan)
    m = np.full(n, np.nan)
    sigma_eq = np.full(n, np.nan)
    tradable = np.zeros(n, dtype=bool)

    # Skip series with too many missing values
    ok = np.isnan(residuals).mean(axis=0) <= 0.2
    if t < min_samples + 1 or not ok.any():
        return tradable, kappa, m, sigma_eq

    # Auxiliary process X_k = Σ_{j=1..k} ε_j
    filled = np.where(np.isnan(residuals), 0.0, residuals)
    cum = np.cumsum(filled, axis=0)
    y, x = cum[1:], cum[:-1]
    n_obs = y.shape[0]

    # OLS AR(1) with intercept, vectorized across assets
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    var_x = ((x - x_mean) ** 2).sum(axis=0)
    cov_xy = ((x - x_mean) * (y - y_mean)).sum(axis=0)

    valid = ok & (var_x > 1e-18)
    b = np.zeros(n)
    a = np.zeros(n)
    b[valid] = cov_xy[valid] / var_x[valid]
    a[valid] = y_mean[valid] - b[valid] * x_mean[valid]

    ar_resid = y - (a + b * x)
    ss_res = (ar_resid ** 2).sum(axis=0)

    # Reject near-unit-root processes (mean-reversion too slow)
    b_max = float(np.exp(-kappa_threshold / 252.0))
    good = valid & (b > 0.0) & (b < b_max)

    kappa[good] = -np.log(b[good]) * 252.0
    m[good] = a[good] / (1.0 - b[good])
    sigma_eq[good] = np.sqrt(ss_res[good] / (n_obs - 1) / (1.0 - b[good] ** 2))
    tradable = good & (kappa > kappa_threshold)

    return tradable, kappa, m, sigma_eq


def s_scores(
    residuals: np.ndarray,
    tradable: np.ndarray,
    m: np.ndarray,
    sigma_eq: np.ndarray,
) -> np.ndarray:
    """
    Latest S-score for each asset (A&L eq. 15, with centered means eq. 18):

        s = (X − m̄) / σ_eq ,   m̄_i = m_i − mean_j(m_j)

    Non-tradable assets receive NaN (must be flat).
    """
    scores = np.full(residuals.shape[1], np.nan)
    if not tradable.any():
        return scores

    # X at the end of the window (= 0 with an intercept, so s ≈ −m̄ / σ_eq)
    x = np.cumsum(residuals[:, tradable], axis=0)[-1]
    m_centered = m[tradable] - np.nanmean(m[tradable])
    scores[tradable] = (x - m_centered) / sigma_eq[tradable]
    return scores
