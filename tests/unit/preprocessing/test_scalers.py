"""Unit tests for qsarify.preprocessing.scalers."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from qsarify.preprocessing.scalers import (
    StandardScaler,
    MinMaxScaler,
)


N = 20
rng = np.random.default_rng(0)
X_RAW = pd.DataFrame(rng.standard_normal((N, 3)) * 5 + 10, columns=["d1", "d2", "d3"])
y_RAW = pd.Series(rng.standard_normal(N) * 2 + 5, name="y")


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------


def test_standard_scaler_mean_zero() -> None:
    sc = StandardScaler()
    sc.fit(X_RAW)
    X_t = sc.transform(X_RAW)
    assert_allclose(X_t.mean(axis=0), 0.0, atol=1e-10)


def test_standard_scaler_std_one() -> None:
    sc = StandardScaler()
    sc.fit(X_RAW)
    X_t = sc.transform(X_RAW)
    assert_allclose(X_t.std(axis=0, ddof=0), 1.0, atol=1e-10)


def test_standard_scaler_inverse() -> None:
    sc = StandardScaler()
    sc.fit(X_RAW)
    X_t = sc.transform(X_RAW)
    X_inv = sc.inverse_transform(X_t)
    assert_allclose(X_inv.values, X_RAW.values, atol=1e-10)


def test_standard_scaler_not_fitted_raises() -> None:
    sc = StandardScaler()
    with pytest.raises(RuntimeError, match="fit"):
        sc.transform(X_RAW)


def test_standard_scaler_returns_dataframe() -> None:
    sc = StandardScaler()
    sc.fit(X_RAW)
    result = sc.transform(X_RAW)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(X_RAW.columns)


# ---------------------------------------------------------------------------
# MinMaxScaler
# ---------------------------------------------------------------------------


def test_minmax_scaler_range() -> None:
    sc = MinMaxScaler(feature_range=(0.0, 1.0))
    sc.fit(X_RAW)
    X_t = sc.transform(X_RAW)
    assert_allclose(X_t.min(axis=0), 0.0, atol=1e-10)
    assert_allclose(X_t.max(axis=0), 1.0, atol=1e-10)


def test_minmax_scaler_custom_range() -> None:
    sc = MinMaxScaler(feature_range=(-1.0, 1.0))
    sc.fit(X_RAW)
    X_t = sc.transform(X_RAW)
    assert_allclose(X_t.min(axis=0), -1.0, atol=1e-10)
    assert_allclose(X_t.max(axis=0), 1.0, atol=1e-10)


def test_minmax_scaler_inverse() -> None:
    sc = MinMaxScaler()
    sc.fit(X_RAW)
    X_t = sc.transform(X_RAW)
    X_inv = sc.inverse_transform(X_t)
    assert_allclose(X_inv.values, X_RAW.values, atol=1e-10)


def test_minmax_scaler_not_fitted_raises() -> None:
    sc = MinMaxScaler()
    with pytest.raises(RuntimeError, match="fit"):
        sc.transform(X_RAW)


def test_minmax_scaler_returns_dataframe() -> None:
    sc = MinMaxScaler()
    sc.fit(X_RAW)
    result = sc.transform(X_RAW)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(X_RAW.columns)
