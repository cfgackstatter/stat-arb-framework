# Statistical Arbitrage — Avellaneda & Lee (2010), PCA version

A clean implementation of the **PCA-based** statistical arbitrage strategy from:

> Avellaneda, M. & Lee, J.-H. (2010). *Statistical arbitrage in the U.S. equities market.*
> Quantitative Finance, 10(7), 761–782.
> [PDF](https://math.nyu.edu/inmemoriam/avellaneda/AvellanedaLeeStatArb20090616.pdf)

## Strategy (paper)

1. **PCA factors** — Estimate the return correlation matrix on a 252-day window. Form eigenportfolios with weights \(Q_i = v_i / \sigma_i\). Keep the fewest components that explain **55%** of variance.
2. **Residuals** — Over the last 60 days, regress each asset on those factor returns (with intercept). The residual is the idiosyncratic return.
3. **Ornstein–Uhlenbeck** — Cumulate residuals and fit an AR(1). Trade only if mean-reversion speed \(\hat\kappa > 252/30 = 8.4\).
4. **S-score** — \(s = (X - \bar m) / \sigma_{eq}\) with cross-sectionally centered means.
5. **Signals (mean-reversion)**
   - Open **long** if \(s < -1.25\); close long if \(s > -0.50\)
   - Open **short** if \(s > +1.25\); close short if \(s < +0.75\)
6. **Portfolio** — ±$1 of each signaled name, hedged with the eigenportfolios (beta-neutral), scaled to target gross leverage. Proportional trading costs applied on turnover.

## Project layout

| File | Role |
|------|------|
| `main.py` | CLI entry point |
| `config.py` | Parameters (paper defaults) |
| `data.py` | Price download and returns |
| `model.py` | PCA, residuals, OU, S-scores |
| `portfolio.py` | Signals and hedged positions |
| `backtest.py` | Walk-forward backtest, metrics, plots |

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

By default this reports **~10 years of strategy returns**. Prices are downloaded
starting one PCA window (~252 trading days) earlier so the model is warm on day
one of that window — standard walk-forward practice. Example: evaluation
2016-07 → 2026-07 uses price history from roughly mid-2015.

Optional overrides:

```bash
python main.py --start_date 2018-01-01 --end_date 2024-12-31 --cost_bps 10
```

### Outputs (`output/`)

- `metrics.csv` — total return, Sharpe, max drawdown, …
- `equity.png` — cumulative returns, tradable count, K over time
- `exposure.png` — long / short / net / gross exposure

## Default parameters

| Parameter | Default | Paper reference |
|-----------|---------|-----------------|
| PCA window | 252 days | §2.1 |
| OU / residual window | 60 days | Appendix |
| Variance explained | 55% | §5.4 |
| κ threshold | 8.4 | Appendix |
| Entry \|s\| | 1.25 | eq. 16 |
| Long / short exit | −0.50 / +0.75 | eq. 16 |
| Gross leverage | 2.0 | “1+1” |
| Cost | 5 bp one-way | §5 |

Edit `config.py` or pass CLI flags to change these.

## Universe

Default symbols are the **Dow 30** (large-cap U.S. equities). That keeps the book inside one asset class with meaningful common factors — closer in spirit to the paper than a mixed ETF set (bonds, gold, country funds, etc.). Edit `DEFAULT_SYMBOLS` in `config.py` to change it.

## Notes

- This is a 30-name demo, not the hundreds of stocks in the original A&L backtests; results will differ.
- The list is the index membership as of 2026 (not point-in-time historically).
- Estimation uses only data before day \(t\); P&L uses the return on day \(t\) (no look-ahead).
- ``start_date`` is the first day of *reported* P&L; warm-up history before that is fetched automatically via ``Config.data_start()``.
- This is a research implementation, not investment advice.
