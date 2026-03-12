"""Tests for validation procedures (LMO and Y-scrambling).

Y-scrambling tests use a minimum of 100 iterations per user requirement.
A 100-sample dataset with a strong linear signal is used to keep the suite
fast while providing meaningful statistical assurance.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from qsarify.validation.procedures import (
    LMOResult,
    YScramblingResult,
    run_lmo,
    run_y_scrambling,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def linear_data() -> tuple[np.ndarray, np.ndarray]:
    """100 samples with a strong linear relationship (low noise).

    Chosen to be large enough that 100 Y-scrambling iterations run quickly
    while giving stable statistics.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + rng.standard_normal(100) * 0.05
    return X.astype(np.float64), y.astype(np.float64)


# ---------------------------------------------------------------------------
# LMOResult dataclass
# ---------------------------------------------------------------------------


def test_lmo_result_fields():
    arr = np.array([0.8, 0.9])
    r = LMOResult(
        holdout_fraction=0.3,
        n_iterations=2,
        q2_per_iteration=arr,
        r2_per_iteration=arr.copy(),
        rmse_per_iteration=np.array([0.1, 0.1]),
        mean_q2=0.85,
        std_q2=0.05,
        mean_r2=0.85,
        std_r2=0.05,
        mean_rmse=0.1,
        std_rmse=0.0,
    )
    assert r.holdout_fraction == 0.3
    assert r.n_iterations == 2
    assert r.mean_q2 == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# run_lmo — structural tests
# ---------------------------------------------------------------------------


def test_lmo_returns_lmo_result(linear_data):
    X, y = linear_data
    result = run_lmo(
        LinearRegression(),
        X,
        y,
        holdout_fraction=0.20,
        n_iterations=5,
        random_seed=42,
    )
    assert isinstance(result, LMOResult)


def test_lmo_iteration_count(linear_data):
    X, y = linear_data
    result = run_lmo(
        LinearRegression(),
        X,
        y,
        holdout_fraction=0.30,
        n_iterations=8,
        random_seed=1,
    )
    assert result.n_iterations == 8
    assert len(result.q2_per_iteration) == 8
    assert len(result.r2_per_iteration) == 8
    assert len(result.rmse_per_iteration) == 8


def test_lmo_holdout_fraction_stored(linear_data):
    X, y = linear_data
    result = run_lmo(
        LinearRegression(),
        X,
        y,
        holdout_fraction=0.33,
        n_iterations=5,
        random_seed=0,
    )
    assert result.holdout_fraction == pytest.approx(0.33)


def test_lmo_good_model_has_positive_q2(linear_data):
    X, y = linear_data
    result = run_lmo(
        LinearRegression(),
        X,
        y,
        holdout_fraction=0.20,
        n_iterations=10,
        random_seed=42,
    )
    assert result.mean_q2 > 0.5


def test_lmo_stats_consistent(linear_data):
    X, y = linear_data
    result = run_lmo(
        LinearRegression(),
        X,
        y,
        holdout_fraction=0.20,
        n_iterations=10,
        random_seed=7,
    )
    assert result.mean_q2 == pytest.approx(float(np.mean(result.q2_per_iteration)), abs=1e-10)
    assert result.std_q2 == pytest.approx(float(np.std(result.q2_per_iteration)), abs=1e-10)
    assert result.mean_r2 == pytest.approx(float(np.mean(result.r2_per_iteration)), abs=1e-10)
    assert result.mean_rmse == pytest.approx(float(np.mean(result.rmse_per_iteration)), abs=1e-10)


def test_lmo_deterministic_with_seed(linear_data):
    X, y = linear_data
    r1 = run_lmo(LinearRegression(), X, y, n_iterations=5, random_seed=99)
    r2 = run_lmo(LinearRegression(), X, y, n_iterations=5, random_seed=99)
    np.testing.assert_array_almost_equal(r1.q2_per_iteration, r2.q2_per_iteration)


def test_lmo_different_seeds_differ(linear_data):
    X, y = linear_data
    r1 = run_lmo(LinearRegression(), X, y, n_iterations=10, random_seed=1)
    r2 = run_lmo(LinearRegression(), X, y, n_iterations=10, random_seed=2)
    assert not np.allclose(r1.q2_per_iteration, r2.q2_per_iteration)


def test_lmo_works_with_ridge(linear_data):
    X, y = linear_data
    result = run_lmo(Ridge(alpha=0.1), X, y, holdout_fraction=0.20, n_iterations=5, random_seed=0)
    assert result.mean_q2 > 0.0


def test_lmo_all_holdout_fractions(linear_data):
    X, y = linear_data
    for frac in [0.20, 0.30, 0.33]:
        result = run_lmo(LinearRegression(), X, y, holdout_fraction=frac, n_iterations=5, random_seed=0)
        assert result.holdout_fraction == pytest.approx(frac)
        assert len(result.q2_per_iteration) == 5


# ---------------------------------------------------------------------------
# YScramblingResult dataclass
# ---------------------------------------------------------------------------


def test_y_scrambling_result_fields():
    arr = np.array([0.1, 0.15, 0.12])
    r = YScramblingResult(
        n_iterations=3,
        r2_scrambled=arr,
        q2_scrambled=arr.copy(),
        r2_original=0.95,
        q2_original=0.92,
        mean_r2_scrambled=0.123,
        std_r2_scrambled=0.025,
        mean_q2_scrambled=0.123,
        std_q2_scrambled=0.025,
    )
    assert r.n_iterations == 3
    assert r.r2_original == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# run_y_scrambling — structural tests (100-iteration minimum)
# ---------------------------------------------------------------------------


def test_y_scrambling_returns_result(linear_data):
    X, y = linear_data
    result = run_y_scrambling(
        LinearRegression(), X, y,
        r2_original=0.95, q2_original=0.90,
        n_iterations=100, random_seed=0,
    )
    assert isinstance(result, YScramblingResult)


def test_y_scrambling_iteration_count(linear_data):
    X, y = linear_data
    result = run_y_scrambling(
        LinearRegression(), X, y,
        r2_original=0.95, q2_original=0.90,
        n_iterations=100, random_seed=0,
    )
    assert result.n_iterations == 100
    assert len(result.r2_scrambled) == 100
    assert len(result.q2_scrambled) == 100


def test_y_scrambling_r2_much_lower_than_original(linear_data):
    """100-iteration scrambling: mean scrambled R² must be far below original."""
    X, y = linear_data
    est = LinearRegression().fit(X, y)
    y_pred = est.predict(X)
    r2_orig = float(1.0 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

    result = run_y_scrambling(
        LinearRegression(), X, y,
        r2_original=r2_orig, q2_original=0.9,
        n_iterations=100, random_seed=42,
    )
    assert result.mean_r2_scrambled < r2_orig - 0.50


def test_y_scrambling_q2_much_lower_than_original(linear_data):
    """100-iteration scrambling: mean scrambled Q²_LOO must be far below original."""
    X, y = linear_data
    result = run_y_scrambling(
        LinearRegression(), X, y,
        r2_original=0.98, q2_original=0.97,
        n_iterations=100, random_seed=42,
    )
    assert result.mean_q2_scrambled < result.q2_original - 0.50


def test_y_scrambling_original_values_stored(linear_data):
    X, y = linear_data
    result = run_y_scrambling(
        LinearRegression(), X, y,
        r2_original=0.95, q2_original=0.88,
        n_iterations=100, random_seed=0,
    )
    assert result.r2_original == pytest.approx(0.95)
    assert result.q2_original == pytest.approx(0.88)


def test_y_scrambling_stats_consistent(linear_data):
    X, y = linear_data
    result = run_y_scrambling(
        LinearRegression(), X, y,
        r2_original=0.95, q2_original=0.90,
        n_iterations=100, random_seed=0,
    )
    assert result.mean_r2_scrambled == pytest.approx(float(np.mean(result.r2_scrambled)), abs=1e-10)
    assert result.std_r2_scrambled == pytest.approx(float(np.std(result.r2_scrambled)), abs=1e-10)
    assert result.mean_q2_scrambled == pytest.approx(float(np.mean(result.q2_scrambled)), abs=1e-10)
    assert result.std_q2_scrambled == pytest.approx(float(np.std(result.q2_scrambled)), abs=1e-10)


def test_y_scrambling_deterministic_with_seed(linear_data):
    X, y = linear_data
    kw = dict(r2_original=0.9, q2_original=0.85, n_iterations=100, random_seed=7)
    r1 = run_y_scrambling(LinearRegression(), X, y, **kw)
    r2 = run_y_scrambling(LinearRegression(), X, y, **kw)
    np.testing.assert_array_almost_equal(r1.r2_scrambled, r2.r2_scrambled)
    np.testing.assert_array_almost_equal(r1.q2_scrambled, r2.q2_scrambled)


def test_y_scrambling_different_seeds_differ(linear_data):
    X, y = linear_data
    r1 = run_y_scrambling(LinearRegression(), X, y, r2_original=0.9, q2_original=0.85,
                          n_iterations=100, random_seed=1)
    r2 = run_y_scrambling(LinearRegression(), X, y, r2_original=0.9, q2_original=0.85,
                          n_iterations=100, random_seed=2)
    assert not np.allclose(r1.r2_scrambled, r2.r2_scrambled)


def test_y_scrambling_scrambled_r2_all_below_original(linear_data):
    """With 100 iterations on a strong model, all scrambled R² should be below original."""
    X, y = linear_data
    est = LinearRegression().fit(X, y)
    y_pred = est.predict(X)
    r2_orig = float(1.0 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

    result = run_y_scrambling(
        LinearRegression(), X, y,
        r2_original=r2_orig, q2_original=0.9,
        n_iterations=100, random_seed=0,
    )
    # On this strong linear dataset, every scrambled model should be much worse
    assert np.all(result.r2_scrambled < r2_orig - 0.20)
