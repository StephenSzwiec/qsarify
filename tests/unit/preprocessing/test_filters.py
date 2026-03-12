"""Unit tests for qsarify.preprocessing.filters."""

import numpy as np
import pandas as pd

from qsarify.preprocessing.filters import (
    remove_high_correlation,
    remove_near_zero_variance,
)


# ---------------------------------------------------------------------------
# Near-zero variance removal
# ---------------------------------------------------------------------------


def test_remove_constant_column() -> None:
    df = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0]})
    result = remove_near_zero_variance(df, threshold=0.01)
    assert "a" not in result.columns
    assert "b" in result.columns


def test_remove_near_constant_column() -> None:
    # std/range is tiny relative to threshold
    df = pd.DataFrame({"a": [1.000, 1.001, 1.000, 1.001], "b": [1.0, 2.0, 3.0, 4.0]})
    result = remove_near_zero_variance(df, threshold=0.01)
    # column a has very low coefficient of variation → should be removed
    assert "a" not in result.columns
    assert "b" in result.columns


def test_keep_all_columns_when_no_low_variance() -> None:
    df = pd.DataFrame({"a": [1.0, 3.0, 5.0], "b": [2.0, 4.0, 6.0]})
    result = remove_near_zero_variance(df, threshold=0.01)
    assert list(result.columns) == ["a", "b"]


def test_returns_dataframe() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = remove_near_zero_variance(df, threshold=0.01)
    assert isinstance(result, pd.DataFrame)


def test_all_constant_columns_removed() -> None:
    df = pd.DataFrame({"a": [5.0, 5.0, 5.0], "b": [3.0, 3.0, 3.0]})
    result = remove_near_zero_variance(df, threshold=0.01)
    assert result.shape[1] == 0


# ---------------------------------------------------------------------------
# High-correlation removal
# ---------------------------------------------------------------------------


def _correlated_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = rng.standard_normal(100)
    noise = rng.standard_normal(100) * 0.05
    # d1 and d2 are almost perfectly correlated; d3 is independent
    d1 = base
    d2 = base + noise
    d3 = rng.standard_normal(100)
    return pd.DataFrame({"d1": d1, "d2": d2, "d3": d3})


def test_removes_correlated_column() -> None:
    df = _correlated_df()
    result = remove_high_correlation(df, threshold=0.95)
    # d1 and d2 are ~1.0 correlated; one should be removed
    assert result.shape[1] == 2


def test_keeps_independent_columns() -> None:
    df = _correlated_df()
    result = remove_high_correlation(df, threshold=0.95)
    # d3 should always survive
    assert "d3" in result.columns


def test_no_removal_below_threshold() -> None:
    df = _correlated_df()
    result = remove_high_correlation(df, threshold=1.0)
    # threshold=1.0 means only perfect correlation triggers removal
    assert result.shape[1] == 3


def test_returns_dataframe_after_correlation_filter() -> None:
    df = _correlated_df()
    result = remove_high_correlation(df, threshold=0.95)
    assert isinstance(result, pd.DataFrame)


def test_single_column_unchanged() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = remove_high_correlation(df, threshold=0.9)
    assert list(result.columns) == ["a"]
