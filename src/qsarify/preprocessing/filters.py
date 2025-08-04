"""
Data preprocessing filters for QSARify.

This module provides functions for filtering descriptor columns based on
variance and correlation, helping to reduce dimensionality and improve model performance.
"""
import pandas as pd
import numpy as np
from typing import Tuple

def remove_near_zero_variance(
    X: pd.DataFrame,
    threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Removes columns from a DataFrame that have near-zero variance.

    Columns with variance below the specified threshold are considered
    near-zero variance and are removed. This helps in removing features
    that provide little to no information to the model.

    Args:
        X: The input DataFrame of descriptor variables.
        threshold: The variance threshold below which columns will be removed.
                   A common value is 0.01, meaning if 99% of the values are the same.

    Returns:
        A new DataFrame with near-zero variance columns removed.
    """
    variances = X.var()
    cols_to_keep = variances[variances >= threshold].index
    return X[cols_to_keep]

def remove_highly_correlated_columns(
    X: pd.DataFrame,
    threshold: float = 0.95,
) -> Tuple[pd.DataFrame, list]:
    """
    Removes one column from each pair of highly correlated columns from a DataFrame.

    This function calculates the Pearson correlation matrix and identifies pairs of
    columns with a correlation coefficient (absolute value) above the specified
    threshold. For each such pair, one of the columns (specifically, the second
    column encountered in the iteration) is marked for removal. This helps in
    reducing multicollinearity and improving model stability while retaining
    at least one representative from highly correlated groups.

    Args:
        X: The input DataFrame of descriptor variables.
        threshold: The absolute Pearson correlation coefficient threshold.
                   If the absolute correlation between two columns is above
                   this threshold, one of them will be removed.

    Returns:
        A tuple containing:
        - A new DataFrame with highly correlated columns removed.
        - A list of column names that were removed.
    """
    corr_matrix = X.corr().abs()
    print(f"[DEBUG] Correlation Matrix:\n{corr_matrix}")
    # Select upper triangle of correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    print(f"[DEBUG] Upper Triangle Matrix:\n{upper_tri}")

    # Find features with correlation greater than threshold and mark one for removal
    to_drop = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr_matrix.iloc[i, j] > threshold:
                to_drop.append(cols[j])
    print(f"[DEBUG] Columns to drop: {to_drop}")

    # Drop features 
    X_filtered = X.drop(columns=to_drop)

    return X_filtered, to_drop
