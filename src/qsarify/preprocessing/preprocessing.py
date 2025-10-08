"""
Data preprocessing filters for QSARify.

This module provides comprehensive preprocessing functionality for QSAR modeling,
including handling of constant/near-constant features, correlation filtering,
train-test splitting, and normalization. It maintains separation of concerns
by focusing purely on data transformation without modeling logic.
"""
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from ..exceptions import PreprocessingError


@dataclass
class PreprocessingConfig:
    """Configuration for comprehensive data preprocessing pipeline."""
    
    # Feature filtering
    constant_threshold: float = 0.01  # Std dev threshold for near-constant features
    correlation_threshold: float = 0.95  # Pearson correlation threshold
    
    # Train-test split
    test_size: float = 0.2
    random_state: int = 42
    split_strategy: str = "sorted"  # "sorted", "random", "stratified"
    
    # Scaling
    apply_scaling: bool = True
    scaling_method: str = "standard"  # "standard", "minmax", "robust"
    
    # Validation
    min_samples: int = 10  # Minimum samples required after preprocessing
    min_features: int = 1   # Minimum features required after preprocessing
    
    # Verbose output
    verbose: bool = True

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

def rm_highCorr(X: pd.DataFrame, threshold: float = 0.9, method: str = "variance") -> pd.DataFrame:
    """
    Remove highly correlated columns from the DataFrame.
    Enhanced to choose which feature to keep based on specified method.

    Parameters:
    X (pd.DataFrame): Input DataFrame with descriptor columns.
    threshold (float): Correlation threshold above which one of the correlated columns will be removed.
    method (str): Method for choosing which feature to keep ("variance", "mean_abs").

    Returns:
    pd.DataFrame: DataFrame with highly correlated columns removed.
    """
    if len(X.columns) <= 1:
        return X
        
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    for column in upper.columns:
        if column in to_drop:
            continue
            
        correlated_features = upper[column][upper[column] > threshold].index.tolist()
        if correlated_features:
            # Add current column to the group
            feature_group = [column] + [f for f in correlated_features if f not in to_drop]
            
            if len(feature_group) > 1:
                # Keep feature with highest variance or mean absolute value
                if method == "variance":
                    keeper = max(feature_group, key=lambda f: X[f].var())
                elif method == "mean_abs":
                    keeper = max(feature_group, key=lambda f: X[f].abs().mean())
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Mark others for removal
                for f in feature_group:
                    if f != keeper:
                        to_drop.add(f)
    
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

def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame, 
               method: str = "standard") -> tuple[pd.DataFrame, pd.DataFrame, Union[StandardScaler, MinMaxScaler, RobustScaler]]:
    """
    Scale the training and testing DataFrames using specified scaling method.

    Parameters:
    X_train (pd.DataFrame): Training DataFrame with descriptor columns.
    X_test (pd.DataFrame): Testing DataFrame with descriptor columns.
    method (str): Scaling method ("standard", "minmax", "robust").

    Returns:
    tuple: Tuple containing scaled training and testing DataFrames and the fitted scaler
           (X_train_scaled, X_test_scaled, scaler).
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}. Choose from 'standard', 'minmax', 'robust'")
    
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled, scaler


def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                    n_bins: int = 5, random_state: int = 42) -> tuple:
    """
    Perform stratified split for regression by binning the target variable.
    
    Parameters:
    X (pd.DataFrame): Input DataFrame with descriptor columns.
    y (pd.Series): Target variable.
    test_size (float): Proportion of the dataset to include in the test split.
    n_bins (int): Number of bins for stratification.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    tuple: Tuple containing training and testing sets (X_train, X_test, y_train, y_test).
    """
    try:
        # Create bins for stratification
        y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y_binned
        )
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        warnings.warn(f"Stratified split failed ({e}), falling back to random split")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def validate_data(X: pd.DataFrame, y: pd.Series, config: PreprocessingConfig) -> None:
    """
    Validate input data meets requirements.
    
    Parameters:
    X (pd.DataFrame): Feature matrix.
    y (pd.Series): Target variable.
    config (PreprocessingConfig): Preprocessing configuration.
    
    Raises:
    PreprocessingError: If data doesn't meet requirements.
    """
    if len(X) < config.min_samples:
        raise PreprocessingError(f"Dataset has {len(X)} samples, but minimum {config.min_samples} required")
    
    if len(X.columns) < config.min_features:
        raise PreprocessingError(f"Dataset has {len(X.columns)} features, but minimum {config.min_features} required")
    
    if X.isna().any().any():
        raise PreprocessingError("Dataset contains missing values. Please handle missing data before preprocessing.")
    
    if y.isna().any():
        raise PreprocessingError("Target variable contains missing values. Please handle missing data before preprocessing.")
    
    if not np.isfinite(X.values).all():
        raise PreprocessingError("Dataset contains non-finite values (inf, -inf)")
    
    if not np.isfinite(y.values).all():
        raise PreprocessingError("Target variable contains non-finite values (inf, -inf)")


def comprehensive_preprocess(X: pd.DataFrame, y: pd.Series, 
                           config: Optional[PreprocessingConfig] = None) -> tuple:
    """
    Comprehensive preprocessing pipeline with enhanced functionality.
    
    Parameters:
    X (pd.DataFrame): Input DataFrame with descriptor columns.
    y (pd.Series): Target variable.
    config (PreprocessingConfig, optional): Configuration object. Uses defaults if None.
    
    Returns:
    tuple: Tuple containing processed data and metadata
           (X_train_scaled, X_test_scaled, y_train, y_test, scaler, preprocessing_info).
    """
    if config is None:
        config = PreprocessingConfig()
    
    # Convert to pandas if necessary
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y, name='target')
    
    preprocessing_info = {
        'original_shape': X.shape,
        'removed_features': [],
        'config': config
    }
    
    if config.verbose:
        print(f"Starting comprehensive preprocessing with {len(X)} samples and {len(X.columns)} features")
    
    # Validate input data
    validate_data(X, y, config)
    
    # Remove low variance features (enhanced)
    initial_features = set(X.columns)
    X_filtered = rm_lowVar(X.copy(), threshold=config.constant_threshold)
    removed_low_var = initial_features - set(X_filtered.columns)
    preprocessing_info['removed_features'].extend(list(removed_low_var))
    
    if config.verbose and removed_low_var:
        print(f"Removed {len(removed_low_var)} low variance features (threshold: {config.constant_threshold})")
    
    # Remove highly correlated features (enhanced)
    pre_corr_features = set(X_filtered.columns)
    X_filtered = rm_highCorr(X_filtered, threshold=config.correlation_threshold, method="variance")
    removed_corr = pre_corr_features - set(X_filtered.columns)
    preprocessing_info['removed_features'].extend(list(removed_corr))
    
    if config.verbose and removed_corr:
        print(f"Removed {len(removed_corr)} highly correlated features (threshold: {config.correlation_threshold})")
    
    # Final feature count check
    if len(X_filtered.columns) < config.min_features:
        warnings.warn(
            f"After filtering, only {len(X_filtered.columns)} features remain, "
            f"but minimum {config.min_features} required. Consider relaxing thresholds."
        )
    
    preprocessing_info['final_features'] = list(X_filtered.columns)
    preprocessing_info['n_removed_features'] = len(preprocessing_info['removed_features'])
    preprocessing_info['n_final_features'] = len(X_filtered.columns)
    
    if config.verbose:
        print(f"After feature filtering: {len(X_filtered.columns)} features remain")
    
    # Create train-test split
    if config.split_strategy == 'sorted':
        X_train, X_test, y_train, y_test = sorted_split(X_filtered, y, test_size=config.test_size)
    elif config.split_strategy == 'random':
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y, test_size=config.test_size, random_state=config.random_state
        )
    elif config.split_strategy == 'stratified':
        X_train, X_test, y_train, y_test = stratified_split(
            X_filtered, y, test_size=config.test_size, random_state=config.random_state
        )
    else:
        raise PreprocessingError(f"Unknown split strategy: {config.split_strategy}")
    
    if config.verbose:
        print(f"Train-test split ({config.split_strategy}): {len(X_train)} train, {len(X_test)} test samples")
    
    # Apply scaling
    if config.apply_scaling:
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test, method=config.scaling_method)
        if config.verbose:
            print(f"Applied {config.scaling_method} scaling to features")
    else:
        X_train_scaled, X_test_scaled, scaler = X_train, X_test, None
        if config.verbose:
            print("No scaling applied")
    
    preprocessing_info['scaler'] = scaler
    preprocessing_info['train_shape'] = X_train_scaled.shape
    preprocessing_info['test_shape'] = X_test_scaled.shape
    
    if config.verbose:
        print("Comprehensive preprocessing completed successfully")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, preprocessing_info

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
