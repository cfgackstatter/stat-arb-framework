"""
Helper functions for the statistical arbitrage strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def print_pca_variance(pca):
    """
    Print explained variance for PCA components.
    
    Parameters:
    -----------
    pca : PCA object
        Fitted PCA object
    """
    explained_variance = pca.explained_variance_ratio_
    cum_explained_variance = np.cumsum(explained_variance)
    
    print("Explained variance by component:")
    for i, var in enumerate(explained_variance[:10]):
        print(f"Component {i+1}: {var:.4f} ({cum_explained_variance[i]:.4f} cumulative)")
    
    print(f"Total explained variance: {sum(explained_variance):.4f}")

def plot_pca_variance(pca, filename='plots/pca_variance.png'):
    """
    Plot explained variance for PCA components.
    
    Parameters:
    -----------
    pca : PCA object
        Fitted PCA object
    """
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    explained_variance = pca.explained_variance_ratio_
    cum_explained_variance = np.cumsum(explained_variance)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual')
    plt.step(range(1, len(cum_explained_variance) + 1), cum_explained_variance, where='mid', label='Cumulative')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()