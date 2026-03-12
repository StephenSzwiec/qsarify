"""Tests for ResultSet."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qsarify.results.model_result import ModelResult
from qsarify.results.result_set import ResultSet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    model_type: str = "mlr",
    n_features: int = 3,
    r2: float = 0.90,
    q2: float = 0.85,
    rmse: float = 0.10,
) -> ModelResult:
    return ModelResult(
        model_type=model_type,
        n_features=n_features,
        n_train=30,
        r_squared=r2,
        q_squared_loo=q2,
        rmse=rmse,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_empty_result_set():
    rs = ResultSet()
    assert len(rs) == 0


def test_add_and_len():
    rs = ResultSet()
    rs.add(_make_result())
    rs.add(_make_result())
    assert len(rs) == 2


def test_init_with_list():
    results = [_make_result(r2=0.9), _make_result(r2=0.8)]
    rs = ResultSet(results=results)
    assert len(rs) == 2


# ---------------------------------------------------------------------------
# Iteration and indexing
# ---------------------------------------------------------------------------


def test_iter():
    rs = ResultSet(results=[_make_result(r2=v) for v in [0.9, 0.8, 0.7]])
    r2_vals = [r.r_squared for r in rs]
    assert r2_vals == [0.9, 0.8, 0.7]


def test_getitem():
    r = _make_result(r2=0.95)
    rs = ResultSet(results=[r])
    assert rs[0] is r


# ---------------------------------------------------------------------------
# Sort
# ---------------------------------------------------------------------------


def test_sort_descending():
    rs = ResultSet(results=[_make_result(r2=v) for v in [0.7, 0.9, 0.8]])
    sorted_rs = rs.sort("r_squared", ascending=False)
    vals = [r.r_squared for r in sorted_rs]
    assert vals == [0.9, 0.8, 0.7]


def test_sort_ascending():
    rs = ResultSet(results=[_make_result(rmse=v) for v in [0.2, 0.1, 0.3]])
    sorted_rs = rs.sort("rmse", ascending=True)
    vals = [r.rmse for r in sorted_rs]
    assert vals == [0.1, 0.2, 0.3]


def test_sort_does_not_mutate_original():
    rs = ResultSet(results=[_make_result(r2=v) for v in [0.7, 0.9]])
    _ = rs.sort("r_squared")
    assert rs[0].r_squared == 0.7  # original unchanged


def test_sort_none_values_last():
    r_none = ModelResult(model_type="mlr", n_features=2, n_train=20, r_squared=None)
    r_good = _make_result(r2=0.9)
    rs = ResultSet(results=[r_none, r_good])
    sorted_rs = rs.sort("r_squared", ascending=False)
    # Good value should come first
    assert sorted_rs[0].r_squared == 0.9


def test_sort_empty_returns_empty():
    rs = ResultSet()
    assert len(rs.sort("r_squared")) == 0


def test_sort_unknown_field_raises():
    rs = ResultSet(results=[_make_result()])
    with pytest.raises(KeyError):
        rs.sort("nonexistent_field")


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


def test_filter_by_model_type():
    rs = ResultSet(results=[
        _make_result(model_type="mlr"),
        _make_result(model_type="ridge"),
        _make_result(model_type="mlr"),
    ])
    filtered = rs.filter(model_type="mlr")
    assert len(filtered) == 2
    assert all(r.model_type == "mlr" for r in filtered)


def test_filter_by_n_features():
    rs = ResultSet(results=[
        _make_result(n_features=2),
        _make_result(n_features=3),
        _make_result(n_features=2),
    ])
    filtered = rs.filter(n_features=2)
    assert len(filtered) == 2


def test_filter_min_r_squared():
    rs = ResultSet(results=[_make_result(r2=v) for v in [0.6, 0.75, 0.9]])
    filtered = rs.filter(min_r_squared=0.75)
    r2_vals = [r.r_squared for r in filtered]
    assert all(v >= 0.75 for v in r2_vals)
    assert len(filtered) == 2


def test_filter_min_q_squared_loo():
    rs = ResultSet(results=[_make_result(q2=v) for v in [0.5, 0.7, 0.85]])
    filtered = rs.filter(min_q_squared_loo=0.7)
    assert len(filtered) == 2


def test_filter_max_rmse():
    rs = ResultSet(results=[_make_result(rmse=v) for v in [0.05, 0.15, 0.30]])
    filtered = rs.filter(max_rmse=0.15)
    assert len(filtered) == 2


def test_filter_combined():
    rs = ResultSet(results=[
        _make_result(model_type="mlr", r2=0.9, rmse=0.05),
        _make_result(model_type="ridge", r2=0.9, rmse=0.05),
        _make_result(model_type="mlr", r2=0.6, rmse=0.20),
    ])
    filtered = rs.filter(model_type="mlr", min_r_squared=0.8)
    assert len(filtered) == 1


def test_filter_empty_input():
    rs = ResultSet()
    assert len(rs.filter(model_type="mlr")) == 0


# ---------------------------------------------------------------------------
# Best
# ---------------------------------------------------------------------------


def test_best_returns_top_n():
    rs = ResultSet(results=[_make_result(r2=v) for v in [0.7, 0.85, 0.9, 0.6]])
    best = rs.best("r_squared", n=2)
    assert len(best) == 2
    assert best[0].r_squared == 0.9
    assert best[1].r_squared == 0.85


def test_best_default_by_r_squared():
    rs = ResultSet(results=[_make_result(r2=v) for v in [0.7, 0.9]])
    best = rs.best()
    assert len(best) == 1
    assert best[0].r_squared == 0.9


# ---------------------------------------------------------------------------
# to_dataframe
# ---------------------------------------------------------------------------


def test_to_dataframe_shape():
    rs = ResultSet(results=[_make_result() for _ in range(5)])
    df = rs.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5


def test_to_dataframe_columns_include_key_fields():
    rs = ResultSet(results=[_make_result()])
    df = rs.to_dataframe()
    for col in ["model_type", "n_features", "r_squared", "q_squared_loo", "rmse"]:
        assert col in df.columns


def test_to_dataframe_no_array_columns():
    rs = ResultSet(results=[_make_result()])
    df = rs.to_dataframe()
    # No ndarray columns — all values should be scalar or None
    for col in df.columns:
        val = df[col].iloc[0]
        assert not isinstance(val, np.ndarray)


def test_to_dataframe_empty():
    rs = ResultSet()
    df = rs.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_to_dataframe_values_match():
    r = _make_result(model_type="ridge", n_features=4, r2=0.92)
    rs = ResultSet(results=[r])
    df = rs.to_dataframe()
    assert df.iloc[0]["model_type"] == "ridge"
    assert df.iloc[0]["n_features"] == 4
    assert df.iloc[0]["r_squared"] == pytest.approx(0.92)
