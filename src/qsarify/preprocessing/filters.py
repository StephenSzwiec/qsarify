"""
Data preprocessing filters for QSARify.

This module provides functions for filtering descriptor columns based on
variance and correlation, helping to reduce dimensionality and improve model performance.
"""
import pandas as pd
import numpy as np

def rm_lowVar(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """
    Remove columns with low variance from the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame with descriptor columns.
    threshold (float): Variance threshold below which columns will be removed.

    Returns:
    pd.DataFrame: DataFrame with low variance columns removed.
    """
    return df.drop(df.columns[df.var() <= threshold], axis=1)

def rm_highCorr(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Remove highly correlated columns from the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame with descriptor columns.
    threshold (float): Correlation threshold above which one of the correlated columns will be removed.

    Returns:
    pd.DataFrame: DataFrame with highly correlated columns removed.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if upper.iloc[i, j] > threshold:
                to_drop.append(upper.columns[j])
    return df.drop(columns=set(to_drop), axis=1)
