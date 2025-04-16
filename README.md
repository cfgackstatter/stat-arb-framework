# Statistical Arbitrage Strategy with PCA

## Overview
This project implements a statistical arbitrage strategy based on Principal Component Analysis (PCA) and Ornstein-Uhlenbeck processes, as described in the Avellaneda-Lee paper "Statistical Arbitrage in the U.S. Equities Market".

The strategy:
1. Decomposes stock returns into systematic factors using PCA
2. Identifies residual components that exhibit mean-reversion
3. Calculates S-scores to generate trading signals
4. Constructs a market-neutral portfolio based on these signals

## Project Structure
- `src/`: Core functionality
  - `data.py`: Data retrieval and processing
  - `pca_model.py`: PCA implementation
  - `ou_process.py`: Ornstein-Uhlenbeck process fitting
  - `signals.py`: Signal generation and portfolio construction
  - `backtest.py`: Backtesting functionality
- `config/`: Configuration
  - `settings.py`: Parameters and settings
- `utils/`: Utility functions
  - `helpers.py`: Helper functions
- `main.py`: Main script to run the strategy

## Installation

### Clone the repository

```console
git clone https://github.com/yourusername/stat-arb-strategy.git
cd stat-arb-strategy
```

### Install dependencies

```console
pip install -r requirements.txt
```

## Usage

### Run the strategy with default settings

```console
python main.py
```

## Configuration
Edit `config/settings.py` to modify:
- Stock universe
- Date range
- PCA window size
- OU window size
- Number of principal components
- Variance threshold
- Mean reversion threshold
- S-score threshold

## Requirements
See `requirements.txt` for the complete list of dependencies.

## License
This project is licensed under the MIT License - see the LICENSE file for details.