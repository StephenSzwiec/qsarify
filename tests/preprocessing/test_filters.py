"""
Unit tests for the data preprocessing filters in QSARify.
"""
import pytest
import pandas as pd
import numpy as np
import qsarify.preprocessing.filters as filters 

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

def test_rm_lowVar_removes_columns_below_threshold(synthetic_data):
    """Test that filters.rm_lowVar removes columns with variance <= threshold."""
    result = filters.rm_lowVar(synthetic_data, threshold=0.01)
    print(result.columns)
    assert "const_col" not in result.columns
    assert "high_var_col" in result.columns
    assert "low_var_col" not in result.columns
    assert len(result.columns) == 4 

def test_rm_highCorr_removes_one_of_each_correlated_pair(synthetic_data):
    """Test that filters.rm_highCorr removes one column of highly correlated pairs."""
    result = filters.rm_highCorr(synthetic_data, threshold=0.95)
    assert len(result.columns) == 5  # One of corr_a or corr_b should be removed
    assert not ("corr_a" in result.columns and "corr_b" in result.columns)

def test_rm_highCorr_does_not_remove_below_threshold(synthetic_data):
    """Test that filters.rm_highCorr keeps columns if correlation is below threshold."""
    df = synthetic_data.copy()
    df["corr_c"] = [1, 2, 3, 4, 6]  # imperfect correlation with corr_a
    result = filters.rm_highCorr(df, threshold=0.99)
    assert "corr_a" in result.columns
    assert "corr_b" not in result.columns  # still removed
    assert "corr_c" in result.columns  # should remain

def test_pipeline_combination(synthetic_data):
    """Test chaining all filters together."""
    result = filters.rm_lowVar(synthetic_data, threshold=0.01)
    result = filters.rm_highCorr(result, threshold=0.9)

    # Expect const_col, high_var_col, and one of corr_a/corr_b to be removed
    assert "const_col" not in result.columns
    assert not ("corr_a" in result.columns and "corr_b" in result.columns)
    assert "unique_col" in result.columns
