"""
Data preprocessing filters for QSARify.

This module provides functions for filtering descriptor columns based on
variance and correlation, helping to reduce dimensionality and improve model performance.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler 

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
    to_drop = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if upper.iloc[i, j] > threshold:
                to_drop.append(upper.columns[j])
    return X.drop(columns=set(to_drop), axis=1)

def sorted_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple:
    """
    Perform a sorted split of the DataFrame into training and testing sets.

    Parameters:
    X (pd.DataFrame): Input DataFrame with descriptor columns.
    y (pd.Series): Target variable.
    test_size (float): Proportion of the dataset to include in the test split.

    Returns:
    tuple: Tuple containing training and testing sets (X_train, X_test, y_train, y_test).
    """
    n_test = int(len(X) * test_size)
    sorted_indices = y.sort_values().index
    test_indices = sorted_indices[::len(sorted_indices)//n_test][:n_test]
    train_indices = sorted_indices.difference(test_indices)
    
    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]
    
    return X_train, X_test, y_train, y_test 

def random_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Perform a random split of the DataFrame into training and testing sets.

    Parameters:
    X (pd.DataFrame): Input DataFrame with descriptor columns.
    y (pd.Series): Target variable.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility in testing.

    Returns:
    tuple: Tuple containing training and testing sets (X_train, X_test, y_train, y_test).
    """
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(X))
    n_test = int(len(X) * test_size)
    test_indices = shuffled_indices[:n_test]
    train_indices = shuffled_indices[n_test:]
    
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    
    return X_train, X_test, y_train, y_test

def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Scale the training and testing DataFrames using Min-Max scaling.

    Parameters:
    X_train (pd.DataFrame): Training DataFrame with descriptor columns.
    X_test (pd.DataFrame): Testing DataFrame with descriptor columns.

    Returns:
    tuple: Tuple containing scaled training and testing DataFrames (X_train_scaled, X_test_scaled).
    """
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled

def preprocess(X: pd.DataFrame, y: pd.Series, split: str = 'sorted', test_size: float = 0.2, var_threshold: float = 0.1, corr_threshold: float = 0.9, _deterministic: bool = False) -> tuple:
    """
    Preprocess the data by removing low variance and highly correlated columns,
    splitting into training and testing sets, and scaling the data.

    Parameters:
    X (pd.DataFrame): Input DataFrame with descriptor columns.
    y (pd.Series): Target variable.
    split (str): Type of split to perform ('sorted' or 'random').
    test_size (float): Proportion of the dataset to include in the test split.
    var_threshold (float): Variance threshold for removing low variance columns.
    corr_threshold (float): Correlation threshold for removing highly correlated columns.
    random_state (int): Random seed for reproducibility in random splitting.

    Returns:
    tuple: Tuple containing processed training and testing sets (X_train_scaled, X_test_scaled, y_train, y_test).
    """
    X_filtered = rm_lowVar(X, threshold=var_threshold)
    X_filtered = rm_highCorr(X_filtered, threshold=corr_threshold)
    
    if split == 'sorted':
        X_train, X_test, y_train, y_test = sorted_split(X_filtered, y, test_size=test_size)
    elif split == 'random':
        X_train, X_test, y_train, y_test = random_split(X_filtered, y, test_size=test_size, random_state=42 if _deterministic else None)
    else:
        raise ValueError("Invalid split method. Choose 'sorted' or 'random'.")
    
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test
