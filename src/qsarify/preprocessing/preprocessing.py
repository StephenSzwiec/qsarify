"""
Data preprocessing filters for QSARify.

This module provides functions for filtering descriptor columns based on
variance and correlation, helping to reduce dimensionality and improve model performance.
It also provides data splitting and scaling utilities, leveraging scikit-learn
for robust and standard implementations.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def rm_lowVar(X: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """
    Remove columns with low variance from the DataFrame.

    Parameters:
    X (pd.DataFrame): Input DataFrame with descriptor columns.
    threshold (float): Variance threshold below which columns will be removed.

    Returns:
    pd.DataFrame: DataFrame with low variance columns removed.
    """
    return X.drop(X.columns[X.var() <= threshold], axis=1)

def rm_highCorr(X: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Remove highly correlated columns from the DataFrame.

    Parameters:
    X (pd.DataFrame): Input DataFrame with descriptor columns.
    threshold (float): Correlation threshold above which one of the correlated columns will be removed.

    Returns:
    pd.DataFrame: DataFrame with highly correlated columns removed.
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = {
        column
        for column in upper.columns
        if any(upper[column] > threshold)
    }
    return X.drop(columns=to_drop)

def sorted_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple:
    """
    Perform a sorted split of the DataFrame into training and testing sets.
    This is a form of stratified splitting for regression, ensuring the test set
    samples are spread across the range of the response variable.

    Parameters:
    X (pd.DataFrame): Input DataFrame with descriptor columns.
    y (pd.Series): Target variable.
    test_size (float): Proportion of the dataset to include in the test split.

    Returns:
    tuple: Tuple containing training and testing sets (X_train, X_test, y_train, y_test).
    """
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    if n_test == 0 and n_samples > 0 and test_size > 0:
        n_test = 1 # ensure at least one sample in test set if possible

    if n_test == 0:
        return X, X.iloc[0:0], y, y.iloc[0:0]

    sorted_indices = y.sort_values().index
    
    # Ensure we don't try to select more items than available
    if n_test > len(sorted_indices):
        n_test = len(sorted_indices)

    # Take every nth element to spread the test samples
    every_nth = max(1, len(sorted_indices) // n_test)
    test_indices = sorted_indices[::every_nth][:n_test]
    
    train_indices = sorted_indices.difference(test_indices)
    
    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]
    
    return X_train, X_test, y_train, y_test

def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale the training and testing DataFrames using StandardScaler.

    Parameters:
    X_train (pd.DataFrame): Training DataFrame with descriptor columns.
    X_test (pd.DataFrame): Testing DataFrame with descriptor columns.

    Returns:
    tuple: Tuple containing scaled training and testing DataFrames and the fitted scaler
           (X_train_scaled, X_test_scaled, scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled, scaler

def preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    split_strategy: str = 'sorted',
    test_size: float = 0.2,
    var_threshold: float = 0.1,
    corr_threshold: float = 0.9,
    random_state: int | None = 42
) -> tuple:
    """
    Preprocess the data by removing low variance and highly correlated columns,
    splitting into training and testing sets, and scaling the data.

    Parameters:
    X (pd.DataFrame): Input DataFrame with descriptor columns.
    y (pd.Series): Target variable.
    split_strategy (str): Type of split to perform ('sorted' or 'random').
    test_size (float): Proportion of the dataset to include in the test split.
    var_threshold (float): Variance threshold for removing low variance columns.
    corr_threshold (float): Correlation threshold for removing highly correlated columns.
    random_state (int | None): Random seed for reproducibility in random splitting.

    Returns:
    tuple: Tuple containing processed training and testing sets and the scaler
           (X_train_scaled, X_test_scaled, y_train, y_test, scaler).
    """
    X_filtered = rm_lowVar(X, threshold=var_threshold)
    X_filtered = rm_highCorr(X_filtered, threshold=corr_threshold)
    
    if split_strategy == 'sorted':
        X_train, X_test, y_train, y_test = sorted_split(X_filtered, y, test_size=test_size)
    elif split_strategy == 'random':
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y, test_size=test_size, random_state=random_state
        )
    else:
        raise ValueError("Invalid split_strategy. Choose 'sorted' or 'random'.")
    
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
