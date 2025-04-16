"""
Configuration settings for the statistical arbitrage strategy.
"""

# Data parameters
SYMBOLS = ['MMM', 'ABT', 'ANF', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD',
           'AKAM', 'AA', 'ALXN', 'ATI', 'ALL', 'MO', 'AMZN', 'AEE',
           'AEP', 'AXP', 'AIG', 'AMT', 'AMP', 'ABC', 'AMGN', 'APH', 'ADI', 'AON',
           'APA', 'AIV', 'AAPL', 'AMAT', 'ADM', 'AIZ', 'T', 'ADSK', 'ADP', 'AN',
           'AZO', 'AVB', 'AVY', 'BLL', 'BAC', 'BK', 'BAX', 'BDX', 'BBBY', 'BIG',
           'BIIB', 'BLK', 'HRB', 'BA', 'BWA', 'BXP', 'BSX', 'BMY', 'CHRW', 'COG',
           'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CAT', 'CNP', 'CERN', 'CF', 'SCHW',
           'CVX', 'CMG', 'CB', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CTXS', 'CLF',
           'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'COP',
           'CNX', 'ED', 'STZ', 'GLW', 'COST', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI',
           'DHR', 'DRI', 'DVA', 'DE', 'XRAY', 'DVN', 'DFS', 'DISCA',
           'DLTR', 'D', 'RRD', 'DOV', 'DTE', 'DD', 'DUK', 'ETFC', 'EMN',
           'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ETR', 'EOG', 'EQT',
           'EFX', 'EQR', 'EL', 'EXC', 'EXPE', 'EXPD', 'XOM', 'FFIV', 'FAST', 'FDX',
           'FIS', 'FITB', 'FHN', 'FSLR', 'FE', 'FISV', 'FLIR', 'FLS', 'FLR', 'FMC',
           'FTI', 'F', 'FOSL', 'BEN', 'FCX', 'GME', 'GPS', 'GD', 'GE', 'GIS',
           'GPC', 'GNW', 'GILD', 'GS', 'GT', 'GOOG', 'GWW', 'HAL', 'HOG', 'HIG',
           'HAS', 'HP', 'HES', 'HPQ', 'HD', 'HON', 'HRL', 'HST', 'HUM', 'HBAN',
           'ITW']
START_DATE = '2024-01-01'
END_DATE = '2025-01-10'  # Set to None to use current date

# Strategy parameters
PCA_WINDOW = 252       # 1 year of trading days for PCA calculation
OU_WINDOW = 60         # Days for OU process estimation
N_COMPONENTS = None      # Number of principal components
VARIANCE_THRESHOLD = 0.55  # Variance threshold for selecting factors
KAPPA_THRESHOLD = 8.4      # Mean reversion threshold (252/30)
S_THRESHOLD = 1.25     # S-score threshold for trading signals