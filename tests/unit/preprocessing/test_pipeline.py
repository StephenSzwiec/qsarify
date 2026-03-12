"""Unit tests for qsarify.preprocessing.pipeline.preprocessing()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qsarify.preprocessing.pipeline import preprocessing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_xy(n: int = 30, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
            "const": np.ones(n),  # constant → removed by NZV filter
            "with_nan": np.where(  # has NaNs → removed by NaN filter
                rng.random(n) > 0.5, np.nan, rng.standard_normal(n)
            ),
        }
    )
    y = pd.Series(rng.standard_normal(n), name="activity")
    return X, y


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------


def test_returns_four_parts() -> None:
    X, y = _make_xy()
    result = preprocessing(X, y)
    assert len(result) == 4


def test_columns_match_train_test() -> None:
    X, y = _make_xy()
    X_tr, X_te, _, _ = preprocessing(X, y)
    assert list(X_tr.columns) == list(X_te.columns)


def test_train_test_split_totals() -> None:
    X, y = _make_xy(n=30)
    X_tr, X_te, y_tr, y_te = preprocessing(X, y, test_size=0.2)
    assert len(X_tr) + len(X_te) == 30
    assert len(y_tr) + len(y_te) == 30


# ---------------------------------------------------------------------------
# Filtering behaviour
# ---------------------------------------------------------------------------


def test_nan_column_removed() -> None:
    X, y = _make_xy()
    X_tr, X_te, _, _ = preprocessing(X, y)
    assert "with_nan" not in X_tr.columns
    assert "with_nan" not in X_te.columns


def test_constant_column_removed() -> None:
    X, y = _make_xy()
    X_tr, X_te, _, _ = preprocessing(X, y)
    assert "const" not in X_tr.columns
    assert "const" not in X_te.columns


def test_informative_columns_retained() -> None:
    X, y = _make_xy()
    X_tr, X_te, _, _ = preprocessing(X, y)
    assert "a" in X_tr.columns
    assert "b" in X_tr.columns


def test_high_correlation_column_removed() -> None:
    """One of a near-perfectly correlated pair should be dropped."""
    n = 30
    rng = np.random.default_rng(1)
    base = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": base,
            "x2": base + rng.standard_normal(n) * 0.001,  # |r| ≈ 1.0
            "x3": rng.standard_normal(n),  # independent
        }
    )
    y = pd.Series(rng.standard_normal(n), name="y")
    X_tr, _, _, _ = preprocessing(X, y, correlation_threshold=0.95)
    # x1 and x2 are near-identical; one must be removed
    assert not ({"x1", "x2"} <= set(X_tr.columns))
    assert "x3" in X_tr.columns


# ---------------------------------------------------------------------------
# Deep-copy invariant
# ---------------------------------------------------------------------------


def test_input_not_mutated() -> None:
    X, y = _make_xy()
    original_cols = list(X.columns)
    original_vals = X["a"].copy()
    preprocessing(X, y)
    assert list(X.columns) == original_cols
    pd.testing.assert_series_equal(X["a"], original_vals)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalize_changes_values() -> None:
    X, y = _make_xy()
    X_tr_norm, _, _, _ = preprocessing(X, y, normalize=True)
    X_tr_raw, _, _, _ = preprocessing(X, y, normalize=False)
    assert not X_tr_norm.equals(X_tr_raw)


def test_normalize_false_preserves_scale() -> None:
    n = 20
    X = pd.DataFrame({"a": np.arange(1.0, n + 1), "b": np.arange(2.0, n + 2)})
    y = pd.Series(np.arange(float(n)), name="y")
    X_tr, X_te, _, _ = preprocessing(
        X, y, normalize=False, split="random", random_seed=0
    )
    combined = pd.concat([X_tr, X_te])
    assert combined["a"].min() >= 1.0
    assert combined["a"].max() <= float(n)


# ---------------------------------------------------------------------------
# Split strategies
# ---------------------------------------------------------------------------


def test_random_split_reproducible() -> None:
    X, y = _make_xy(n=30)
    r1 = preprocessing(X, y, split="random", random_seed=7)
    r2 = preprocessing(X, y, split="random", random_seed=7)
    assert list(r1[0].index) == list(r2[0].index)


def test_sorted_split_all_rows_covered() -> None:
    X, y = _make_xy(n=30)
    X_tr, X_te, _, _ = preprocessing(X, y, split="sorted")
    assert len(X_tr) + len(X_te) == 30


def test_invalid_split_raises() -> None:
    X, y = _make_xy()
    with pytest.raises(ValueError, match="split"):
        preprocessing(X, y, split="unknown_method")
