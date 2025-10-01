"""
Unit tests for the data preprocessing preprocessing in QSARify.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import qsarify.preprocessing.preprocessing as preprocessing 

@pytest.fixture
def synthetic_data():
    """Create a synthetic DataFrame for testing."""
    return pd.DataFrame({
        "const_col": [1, 1, 1, 1, 1],
        "low_var_col": [0.05, 0.04, 0.05, 0.06, 0.05],
        "high_var_col": [1, 10, 5, 6, 3],
        "corr_a": [1, 2, 3, 4, 5],
        "corr_b": [2, 4, 6, 8, 10],  # perfectly correlated with corr_a
        "unique_col": [7, 6, 2, 9, 11]
    })

@pytest.fixture
def synthetic_data_with_target():
    """
    Create a synthetic DataFrame and target Series for splitting and scaling tests.
    """
    X = pd.DataFrame({
        'feature1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'feature2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    y = pd.Series([100, 10, 90, 20, 80, 30, 70, 40, 60, 50], name='target')
    return X, y

def test_rm_lowVar_removes_columns_below_threshold(synthetic_data):
    """Test that preprocessing.rm_lowVar removes columns with variance <= threshold."""
    result = preprocessing.rm_lowVar(synthetic_data, threshold=0.01)
    assert "const_col" not in result.columns
    assert "low_var_col" not in result.columns
    assert "high_var_col" in result.columns
    assert len(result.columns) == 4 

def test_rm_highCorr_removes_one_of_each_correlated_pair(synthetic_data):
    """Test that preprocessing.rm_highCorr removes one column of highly correlated pairs."""
    result = preprocessing.rm_highCorr(synthetic_data, threshold=0.95)
    assert len(result.columns) == 5  # One of corr_a or corr_b should be removed
    assert not ("corr_a" in result.columns and "corr_b" in result.columns)

def test_rm_highCorr_does_not_remove_below_threshold(synthetic_data):
    """Test that preprocessing.rm_highCorr keeps columns if correlation is below threshold."""
    df = synthetic_data.copy()
    df["corr_c"] = [1, 2, 3, 4, 6]  # imperfect correlation with corr_a
    result = preprocessing.rm_highCorr(df, threshold=0.99)
    assert "corr_a" in result.columns
    assert "corr_b" not in result.columns  # still removed
    assert "corr_c" in result.columns  # should remain

def test_sorted_split(synthetic_data_with_target):
    """
    Test that sorted_split correctly splits data based on sorted target variable.
    """
    X, y = synthetic_data_with_target
    X_train, X_test, y_train, y_test = preprocessing.sorted_split(X, y, test_size=0.2)

    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2

    # Check if the test set contains the most extreme values from the sorted target
    # For test_size=0.2 and 10 samples, n_test = 2. The indices should be 0 and 5 (10/2 = 5, so every 5th element)
    # Original sorted y: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # Test set should pick y values at index 0 and 5 of the sorted series, which are 10 and 60.
    # The original indices for these values are 1 and 8.
    assert y_test.iloc[0] == 10
    assert y_test.iloc[1] == 60
    assert y_test.index.tolist() == [1, 8]

def test_random_split(synthetic_data_with_target):
    """
    Test that random_split correctly splits data randomly and is reproducible.
    """
    X, y = synthetic_data_with_target
    X_train1, X_test1, y_train1, y_test1 = preprocessing.random_split(X, y, test_size=0.3, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = preprocessing.random_split(X, y, test_size=0.3, random_state=42)

    assert len(X_train1) == 7
    assert len(X_test1) == 3
    assert y_test1.name == 'target'

    # Check reproducibility
    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    pd.testing.assert_series_equal(y_test1, y_test2)

def test_scale_data(synthetic_data_with_target):
    """
    Test that scale_data correctly scales the data using MinMaxScaler.
    """
    X, y = synthetic_data_with_target
    X_train, X_test, _, _ = preprocessing.random_split(X, y, test_size=0.3, random_state=42)
    X_train_scaled, X_test_scaled = preprocessing.scale_data(X_train, X_test)

    scaler = MinMaxScaler()
    expected_X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    expected_X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    pd.testing.assert_frame_equal(X_train_scaled, expected_X_train_scaled)
    pd.testing.assert_frame_equal(X_test_scaled, expected_X_test_scaled)

def test_preprocess_sorted_split(synthetic_data_with_target):
    """
    Test the full preprocess pipeline with sorted split.
    """
    X, y = synthetic_data_with_target
    # Add some columns that will be removed by preprocessing
    X['const_col'] = 1
    X['low_var_col'] = [0.05, 0.04, 0.05, 0.06, 0.05, 0.05, 0.04, 0.05, 0.06, 0.05]
    X['corr_a'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    X['corr_b'] = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    X_train_scaled, X_test_scaled, y_train, y_test = preprocessing.preprocess(
        X, y, split='sorted', test_size=0.2, var_threshold=0.01, corr_threshold=0.95
    )

    assert 'const_col' not in X_train_scaled.columns
    assert 'low_var_col' not in X_train_scaled.columns
    assert not ('corr_a' in X_train_scaled.columns and 'corr_b' in X_train_scaled.columns)
    assert 'feature1' in X_train_scaled.columns
    assert len(X_train_scaled) == 8
    assert len(X_test_scaled) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2

def test_preprocess_random_split(synthetic_data_with_target):
    """
    Test the full preprocess pipeline with random split.
    """
    X, y = synthetic_data_with_target
    # Add some columns that will be removed by preprocessing
    X['const_col'] = 1
    X['low_var_col'] = [0.05, 0.04, 0.05, 0.06, 0.05, 0.05, 0.04, 0.05, 0.06, 0.05]
    X['corr_a'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    X['corr_b'] = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    X_train_scaled, X_test_scaled, y_train, y_test = preprocessing.preprocess(
        X, y, split='random', test_size=0.3, var_threshold=0.01, corr_threshold=0.95, _deterministic=True
    )

    assert 'const_col' not in X_train_scaled.columns
    assert 'low_var_col' not in X_train_scaled.columns
    assert not ('corr_a' in X_train_scaled.columns and 'corr_b' in X_train_scaled.columns)
    assert 'feature1' in X_train_scaled.columns
    assert len(X_train_scaled) == 7
    assert len(X_test_scaled) == 3
    assert len(y_train) == 7
    assert len(y_test) == 3

def test_preprocess_invalid_split_method(synthetic_data_with_target):
    """
    Test that preprocess raises ValueError for invalid split method.
    """
    X, y = synthetic_data_with_target
    with pytest.raises(ValueError, match="Invalid split method. Choose 'sorted' or 'random'."):
        preprocessing.preprocess(X, y, split='invalid_method')