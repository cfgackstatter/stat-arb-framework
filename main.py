"""
Entry point: run the Avellaneda & Lee (2010) PCA statistical arbitrage backtest.

Usage
-----
    python main.py
    python main.py --start_date 2018-01-01 --cost_bps 10
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from backtest import performance_metrics, plot_exposure, plot_results, run_backtest
from config import Config, with_cli_overrides
from data import calculate_returns, load_prices

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('stat_arb.log'),
        logging.StreamHandler(),
    ],
)
logging.getLogger().handlers[1].setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args(cfg: Config) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Avellaneda–Lee PCA statistical arbitrage backtest',
    )
    parser.add_argument('--start_date', default=cfg.start_date)
    parser.add_argument('--end_date', default=cfg.end_date)
    parser.add_argument('--pca_window', type=int, default=cfg.pca_window)
    parser.add_argument('--ou_window', type=int, default=cfg.ou_window)
    parser.add_argument('--variance_threshold', type=float, default=cfg.variance_threshold)
    parser.add_argument('--kappa_threshold', type=float, default=cfg.kappa_threshold)
    parser.add_argument('--entry_threshold', type=float, default=cfg.entry_threshold)
    parser.add_argument('--long_exit', type=float, default=cfg.long_exit)
    parser.add_argument('--short_exit', type=float, default=cfg.short_exit)
    parser.add_argument('--cost_bps', type=float, default=cfg.cost_bps)
    return parser.parse_args()


def main() -> None:
    cfg = with_cli_overrides(Config(), parse_args(Config()))
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch pca_window trading days before start_date so the first reported
    # return already has a full estimation history (standard warm-up).
    data_start = cfg.data_start()
    print(f'Data: {data_start} → {cfg.end_date or "today"} (warm-up before {cfg.start_date})')
    print(f'Report strategy returns: {cfg.start_date} → {cfg.end_date or "today"}')

    prices = load_prices(cfg.symbols, data_start, cfg.end_date)
    returns = calculate_returns(prices, winsorize_pct=cfg.winsorize_pct)

    results = run_backtest(returns, cfg)
    metrics = performance_metrics(results['daily_returns'])

    print('\nPerformance Metrics')
    print('-------------------')
    for key, value in metrics.items():
        if any(token in key for token in ('return', 'drawdown', 'volatility', 'rate')):
            print(f'  {key}: {value:.2%}')
        else:
            print(f'  {key}: {value:.2f}')

    pd.Series(metrics).to_csv(cfg.output_dir / 'metrics.csv')
    plot_results(results, cfg.output_dir / 'equity.png')
    plot_exposure(results, cfg.output_dir / 'exposure.png')
    logger.info('Results written to %s/', cfg.output_dir)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logger.exception('Backtest failed')
        raise
