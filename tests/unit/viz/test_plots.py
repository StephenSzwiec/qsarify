"""Tests for viz/plots.py — diagnostic plot functions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pytest

from qsarify.modeling.clustering import cophenetic_cluster
from qsarify.results.model_result import ModelResult
from qsarify.validation.procedures import YScramblingResult
from qsarify.viz.plots import (
    plot_cluster,
    plot_qq,
    plot_residuals,
    plot_williams,
    plot_y_scrambling,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(n: int = 30, n_features: int = 3, seed: int = 0) -> ModelResult:
    """Build a ModelResult with realistic arrays for plotting tests."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    y_pred = y + rng.standard_normal(n) * 0.2
    residuals = y - y_pred
    s = float(np.std(residuals))
    std_resid = residuals / s if s > 0 else residuals

    p = n_features
    X = rng.standard_normal((n, p))
    XtX_inv = np.linalg.pinv(X.T @ X)
    leverage = np.asarray((X @ XtX_inv * X).sum(axis=1), dtype=np.float64)
    h_star = 3.0 * p / n

    return ModelResult(
        model_type="mlr",
        n_features=p,
        n_train=n,
        r_squared=0.90,
        q_squared_loo=0.85,
        rmse=0.20,
        std_error_estimate=s,
        leverage=leverage,
        std_residuals=std_resid,
        leverage_threshold=h_star,
        y_train=y,
        y_pred_train=y_pred,
    )


def _make_y_scram_result(seed: int = 0) -> YScramblingResult:
    rng = np.random.default_rng(seed)
    r2_scram = rng.uniform(0.0, 0.3, size=100)
    q2_scram = rng.uniform(-0.5, 0.2, size=100)
    return YScramblingResult(
        n_iterations=100,
        r2_scrambled=r2_scram,
        q2_scrambled=q2_scram,
        r2_original=0.92,
        q2_original=0.88,
        mean_r2_scrambled=float(np.mean(r2_scram)),
        std_r2_scrambled=float(np.std(r2_scram)),
        mean_q2_scrambled=float(np.mean(q2_scram)),
        std_q2_scrambled=float(np.std(q2_scram)),
    )


# ---------------------------------------------------------------------------
# plot_residuals
# ---------------------------------------------------------------------------


def test_plot_residuals_returns_figure():
    result = _make_result()
    fig = plot_residuals(result)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_residuals_has_axes():
    result = _make_result()
    fig = plot_residuals(result)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_residuals_x_is_fitted_values():
    """The scatter x-data should equal y_pred_train."""
    result = _make_result()
    fig = plot_residuals(result)
    ax = fig.axes[0]
    # The first collection or line should contain the y_pred values on x-axis
    scatter_x = ax.collections[0].get_offsets()[:, 0]
    np.testing.assert_array_almost_equal(
        np.sort(scatter_x), np.sort(result.y_pred_train)
    )
    plt.close(fig)


def test_plot_residuals_y_is_std_residuals():
    """The scatter y-data should equal std_residuals."""
    result = _make_result()
    fig = plot_residuals(result)
    ax = fig.axes[0]
    scatter_y = ax.collections[0].get_offsets()[:, 1]
    np.testing.assert_array_almost_equal(
        np.sort(scatter_y), np.sort(result.std_residuals)
    )
    plt.close(fig)


def test_plot_residuals_accepts_ax():
    """Plot function should use a provided Axes."""
    result = _make_result()
    fig_ext, ax_ext = plt.subplots()
    fig = plot_residuals(result, ax=ax_ext)
    assert fig is fig_ext
    plt.close(fig)


def test_plot_residuals_save_path(tmp_path):
    result = _make_result()
    out = tmp_path / "residuals.png"
    fig = plot_residuals(result, save_path=str(out))
    assert out.exists()
    plt.close(fig)


def test_plot_residuals_missing_y_pred_raises():
    result = _make_result()
    result.y_pred_train = None
    with pytest.raises(ValueError, match="y_pred_train"):
        plot_residuals(result)


def test_plot_residuals_missing_std_residuals_raises():
    result = _make_result()
    result.std_residuals = None
    with pytest.raises(ValueError, match="std_residuals"):
        plot_residuals(result)


# ---------------------------------------------------------------------------
# plot_qq
# ---------------------------------------------------------------------------


def test_plot_qq_returns_figure():
    result = _make_result()
    fig = plot_qq(result)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_qq_has_axes():
    result = _make_result()
    fig = plot_qq(result)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_qq_accepts_ax():
    result = _make_result()
    fig_ext, ax_ext = plt.subplots()
    fig = plot_qq(result, ax=ax_ext)
    assert fig is fig_ext
    plt.close(fig)


def test_plot_qq_save_path(tmp_path):
    result = _make_result()
    out = tmp_path / "qq.png"
    fig = plot_qq(result, save_path=str(out))
    assert out.exists()
    plt.close(fig)


def test_plot_qq_missing_std_residuals_raises():
    result = _make_result()
    result.std_residuals = None
    with pytest.raises(ValueError, match="std_residuals"):
        plot_qq(result)


# ---------------------------------------------------------------------------
# plot_williams
# ---------------------------------------------------------------------------


def test_plot_williams_returns_figure():
    result = _make_result()
    fig = plot_williams(result)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_williams_has_axes():
    result = _make_result()
    fig = plot_williams(result)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_williams_threshold_line_present():
    """The Williams plot must draw a vertical line at h* = 3p/n."""
    result = _make_result(n=30, n_features=3)
    h_star = result.leverage_threshold
    fig = plot_williams(result)
    ax = fig.axes[0]
    # Check that a vertical line at h_star exists among the axes lines
    vlines_x = [
        line.get_xdata()[0]
        for line in ax.lines
        if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]
    ]
    assert any(abs(x - h_star) < 1e-6 for x in vlines_x), (
        f"Expected vertical line at h*={h_star}, found x-positions: {vlines_x}"
    )
    plt.close(fig)


def test_plot_williams_accepts_ax():
    result = _make_result()
    fig_ext, ax_ext = plt.subplots()
    fig = plot_williams(result, ax=ax_ext)
    assert fig is fig_ext
    plt.close(fig)


def test_plot_williams_save_path(tmp_path):
    result = _make_result()
    out = tmp_path / "williams.png"
    fig = plot_williams(result, save_path=str(out))
    assert out.exists()
    plt.close(fig)


def test_plot_williams_missing_leverage_raises():
    result = _make_result()
    result.leverage = None
    with pytest.raises(ValueError, match="leverage"):
        plot_williams(result)


def test_plot_williams_missing_std_residuals_raises():
    result = _make_result()
    result.std_residuals = None
    with pytest.raises(ValueError, match="std_residuals"):
        plot_williams(result)


def test_plot_williams_x_is_leverage():
    result = _make_result()
    fig = plot_williams(result)
    ax = fig.axes[0]
    scatter_x = ax.collections[0].get_offsets()[:, 0]
    np.testing.assert_array_almost_equal(np.sort(scatter_x), np.sort(result.leverage))
    plt.close(fig)


def test_plot_williams_y_is_std_residuals():
    result = _make_result()
    fig = plot_williams(result)
    ax = fig.axes[0]
    scatter_y = ax.collections[0].get_offsets()[:, 1]
    np.testing.assert_array_almost_equal(
        np.sort(scatter_y), np.sort(result.std_residuals)
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_y_scrambling
# ---------------------------------------------------------------------------


def test_plot_y_scrambling_returns_figure():
    y_scram = _make_y_scram_result()
    result = _make_result()
    fig = plot_y_scrambling(y_scram, result)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_y_scrambling_has_two_axes():
    """Y-scrambling plot shows R² and Q² distributions side by side."""
    y_scram = _make_y_scram_result()
    result = _make_result()
    fig = plot_y_scrambling(y_scram, result)
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_plot_y_scrambling_accepts_axes():
    y_scram = _make_y_scram_result()
    result = _make_result()
    fig_ext, (ax1, ax2) = plt.subplots(1, 2)
    fig = plot_y_scrambling(y_scram, result, axes=(ax1, ax2))
    assert fig is fig_ext
    plt.close(fig)


def test_plot_y_scrambling_save_path(tmp_path):
    y_scram = _make_y_scram_result()
    result = _make_result()
    out = tmp_path / "y_scram.png"
    fig = plot_y_scrambling(y_scram, result, save_path=str(out))
    assert out.exists()
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_cluster
# ---------------------------------------------------------------------------


def _make_cluster_result(p: int = 8, cut_d: float = 0.5, seed: int = 0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(60)
    noise = rng.standard_normal((60, p)) * 0.5
    X = base[:, None] + noise
    return cophenetic_cluster(X, cut_d=cut_d)


def test_plot_cluster_returns_figure() -> None:
    result = _make_cluster_result()
    fig = plot_cluster(result)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_cluster_has_one_panel() -> None:
    result = _make_cluster_result()
    fig = plot_cluster(result)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_plot_cluster_all_singletons_no_raise() -> None:
    """All-singleton partition (no cohesion values) must not raise."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((30, 6))
    result = cophenetic_cluster(X, cut_d=1e-6)
    fig = plot_cluster(result)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_cluster_accepts_ax() -> None:
    result = _make_cluster_result()
    fig_ext, ax_ext = plt.subplots()
    returned = plot_cluster(result, ax=ax_ext)
    assert returned is fig_ext
    plt.close(fig_ext)


def test_plot_cluster_save_path(tmp_path) -> None:
    result = _make_cluster_result()
    out = tmp_path / "cluster.png"
    plot_cluster(result, save_path=str(out))
    assert out.exists() and out.stat().st_size > 0
