"""Unit tests for the data preprocessing filter utilities."""
import pytest
import pandas as pd
import numpy as np
from qsarify.preprocessing import filters

@pytest.fixture
def sample_dataframe():
    """Provides a sample DataFrame for testing filter functions."""
    data = {
        'col_nzv_1': [1, 1, 1, 1, 1],  # Near zero variance
        'col_nzv_2': [1, 1, 1, 1, 2],  # Low variance
        'col_high_var': [1, 2, 3, 4, 5],  # High variance
        'col_corr_1': [10, 20, 30, 40, 50],  # Highly correlated with col_corr_2
        'col_corr_2': [10.1, 20.2, 30.3, 40.4, 50.5],  # Highly correlated with col_corr_1
        'col_uncorr': [5, 4, 3, 2, 1],  # Uncorrelated
        'col_mixed': [1, 2, 1, 2, 1] # Mixed, but not near zero
    }
    return pd.DataFrame(data)

def test_remove_near_zero_variance(sample_dataframe):
    """Test remove_near_zero_variance function."""
    df = sample_dataframe.copy()

    # Test with default threshold (0.01)
    # col_nzv_1 (variance 0) should be removed
    # col_nzv_2 (variance 0.2) should be kept (0.2 > 0.01)
    df_filtered = filters.remove_near_zero_variance(df)
    assert 'col_nzv_1' not in df_filtered.columns
    assert 'col_nzv_2' in df_filtered.columns
    assert 'col_high_var' in df_filtered.columns
    assert 'col_corr_1' in df_filtered.columns
    assert 'col_corr_2' in df_filtered.columns
    assert 'col_uncorr' in df_filtered.columns
    assert 'col_mixed' in df_filtered.columns
    assert len(df_filtered.columns) == 6

    # Test with a higher threshold (e.g., 0.5)
    # col_nzv_1, col_nzv_2, col_mixed should be removed
    df_filtered_high_thresh = filters.remove_near_zero_variance(df, threshold=0.5)
    assert 'col_nzv_1' not in df_filtered_high_thresh.columns
    assert 'col_nzv_2' not in df_filtered_high_thresh.columns
    assert 'col_mixed' not in df_filtered_high_thresh.columns
    assert len(df_filtered_high_thresh.columns) == 4

    # Test with all columns having high variance
    df_all_high_var = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
    df_filtered_all_high_var = filters.remove_near_zero_variance(df_all_high_var)
    assert len(df_filtered_all_high_var.columns) == 2

    # Test with all columns having zero variance
    df_all_zero_var = pd.DataFrame({'A': [1,1,1], 'B': [2,2,2]})
    df_filtered_all_zero_var = filters.remove_near_zero_variance(df_all_zero_var)
    assert len(df_filtered_all_zero_var.columns) == 0

def test_remove_highly_correlated_columns(sample_dataframe):
    """Test remove_highly_correlated_columns function."""
    df = sample_dataframe.copy()
    print(f"[DEBUG] Sample DataFrame Correlation Matrix:\n{df.corr().abs()}")

    # Test with default threshold (0.95)
    # col_corr_1 and col_corr_2 are highly correlated (corr ~1.0)
    df_filtered, removed_cols = filters.remove_highly_correlated_columns(df)
    assert ('col_corr_1' in df_filtered.columns and 'col_corr_2' not in df_filtered.columns) or \
           ('col_corr_2' in df_filtered.columns and 'col_corr_1' not in df_filtered.columns)
    assert len(removed_cols) == 1
    assert len(df_filtered.columns) == 6

    # Test with a lower threshold (e.g., 0.5) to remove more columns
    # col_corr_1/2 should be removed, and potentially others depending on correlations
    df_filtered_low_thresh, removed_cols_low_thresh = filters.remove_highly_correlated_columns(df, threshold=0.5)
    assert len(removed_cols_low_thresh) >= 1 # At least one of col_corr_1/2
    assert len(df_filtered_low_thresh.columns) <= 6

    # Test with no highly correlated columns
    df_no_corr = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [3, 2, 1],
        'C': [1, 3, 2]
    })
    df_filtered_no_corr, removed_cols_no_corr = filters.remove_highly_correlated_columns(df_no_corr)
    assert len(removed_cols_no_corr) == 0
    assert len(df_filtered_no_corr.columns) == 3

    # Test with all columns highly correlated
    df_all_corr = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [1.1, 2.1, 3.1],
        'C': [1.2, 2.2, 3.2]
    })
    df_filtered_all_corr, removed_cols_all_corr = filters.remove_highly_correlated_columns(df_all_corr)
    assert len(removed_cols_all_corr) == 2 # Should keep only one column
    assert len(df_filtered_all_corr.columns) == 1
