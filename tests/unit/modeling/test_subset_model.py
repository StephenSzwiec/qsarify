"""Unit tests for qsarify.modeling.base.SubsetModel (OLS/MLR)."""

import numpy as np
from numpy.testing import assert_allclose

from qsarify.modeling.subset_model import SubsetModel
from qsarify.results.model_result import ModelResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)
N, P = 40, 5

X_ALL = RNG.standard_normal((N, P))
TRUE_COEF = np.array([1.0, -2.0, 0.5])  # true coefficients for descriptors 0,2,4
NOISE = RNG.standard_normal(N) * 0.1
Y = (
    X_ALL[:, 0] * TRUE_COEF[0]
    + X_ALL[:, 2] * TRUE_COEF[1]
    + X_ALL[:, 4] * TRUE_COEF[2]
    + NOISE
)

INDICES = [0, 2, 4]  # select 3 out of 5 descriptors

# Small external test set
X_TEST = RNG.standard_normal((10, P))
Y_TEST = (
    X_TEST[:, 0] * TRUE_COEF[0]
    + X_TEST[:, 2] * TRUE_COEF[1]
    + X_TEST[:, 4] * TRUE_COEF[2]
    + RNG.standard_normal(10) * 0.1
)


# ---------------------------------------------------------------------------
# Basic fit/predict
# ---------------------------------------------------------------------------


def test_fit_returns_self() -> None:
    model = SubsetModel(INDICES)
    result = model.fit(X_ALL, Y)
    assert result is model


def test_predict_shape() -> None:
    model = SubsetModel(INDICES)
    model.fit(X_ALL, Y)
    preds = model.predict(X_ALL)
    assert preds.shape == (N,)


def test_predict_on_test_set_shape() -> None:
    model = SubsetModel(INDICES)
    model.fit(X_ALL, Y)
    preds = model.predict(X_TEST)
    assert preds.shape == (10,)


def test_recovers_true_coefficients() -> None:
    """With low noise, recovered coefficients should be close to truth."""
    model = SubsetModel(INDICES)
    model.fit(X_ALL, Y)
    coef = model.coef_  # non-intercept coefficients
    assert_allclose(coef, TRUE_COEF, atol=0.2)


def test_perfect_fit_r2_is_one() -> None:
    """Exact linear data → R² = 1."""
    X = np.column_stack([np.arange(20, dtype=float), np.ones(20)])
    y = 3.0 * X[:, 0] + 2.0
    model = SubsetModel([0, 1])
    model.fit(X, y)
    r = model.get_results()
    assert r.r_squared is not None
    assert_allclose(r.r_squared, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# get_results → training metrics
# ---------------------------------------------------------------------------


def test_get_results_returns_model_result() -> None:
    model = SubsetModel(INDICES)
    model.fit(X_ALL, Y)
    assert isinstance(model.get_results(), ModelResult)


def test_training_metrics_populated() -> None:
    model = SubsetModel(INDICES)
    model.fit(X_ALL, Y)
    r = model.get_results()
    for attr in (
        "r_squared",
        "r_squared_adj",
        "rmse",
        "mae",
        "mse",
        "rss",
        "tss",
        "mss",
        "std_error_estimate",
        "f_statistic",
        "lof",
        "ccc",
        "q_squared_loo",
        "slope_origin",
        "slope_origin_reverse",
        "r_squared_origin",
        "r_squared_origin_reverse",
        "roy_r_squared_m_mean",
        "roy_r_squared_m_delta",
        "closeness",
        "closeness_reverse",
    ):
        assert getattr(r, attr) is not None, f"{attr} should not be None"


def test_external_metrics_none_without_test() -> None:
    model = SubsetModel(INDICES)
    model.fit(X_ALL, Y)
    r = model.get_results()
    for attr in (
        "q_squared_f1",
        "q_squared_f2",
        "q_squared_f3",
        "r_squared_ext",
        "press_ext",
    ):
        assert getattr(r, attr) is None, f"{attr} should be None without test set"


def test_r_squared_in_unit_interval() -> None:
    model = SubsetModel(INDICES)
    model.fit(X_ALL, Y)
    r = model.get_results()
    assert r.r_squared is not None
    assert 0.0 <= r.r_squared <= 1.0


def test_q2_loo_less_than_r_squared() -> None:
    """Q²_LOO ≤ R² in the absence of overfitting."""
    model = SubsetModel(INDICES)
    model.fit(X_ALL, Y)
    r = model.get_results()
    assert r.q_squared_loo is not None
    assert r.r_squared is not None
    assert r.q_squared_loo <= r.r_squared + 1e-10  # small tolerance


# ---------------------------------------------------------------------------
# Coefficient statistics
# ---------------------------------------------------------------------------


def test_coef_stats_not_none() -> None:
    model = SubsetModel(INDICES)
    model.fit(X_ALL, Y)
    r = model.get_results()
    assert r.coef_std_errors is not None
    assert r.coef_confidence_intervals is not None
    assert r.coef_p_values is not None


def test_coef_stats_shapes() -> None:
    model = SubsetModel(INDICES)
    model.fit(X_ALL, Y)
    r = model.get_results()
    p = len(INDICES)
    assert r.coef_std_errors is not None
    assert r.coef_std_errors.shape == (p,)
    assert r.coef_confidence_intervals is not None
    assert r.coef_confidence_intervals.shape == (p, 2)
    assert r.coef_p_values is not None
    assert r.coef_p_values.shape == (p,)


def test_coef_ci_contains_truth() -> None:
    """True coefficients should lie within 95% CI for low-noise data."""
    model = SubsetModel(INDICES)
    model.fit(X_ALL, Y)
    r = model.get_results()
    assert r.coef_confidence_intervals is not None
    for i, truth in enumerate(TRUE_COEF):
        lo = r.coef_confidence_intervals[i, 0]
        hi = r.coef_confidence_intervals[i, 1]
        assert lo < truth < hi, f"Truth {truth} not in CI [{lo}, {hi}] for coef {i}"


def test_significant_coefs_have_small_pvalues() -> None:
    """True predictors should be statistically significant."""
    model = SubsetModel(INDICES)
    model.fit(X_ALL, Y)
    r = model.get_results()
    assert r.coef_p_values is not None
    # All three true descriptors should be significant at 0.05 level
    assert (r.coef_p_values < 0.05).all()


# ---------------------------------------------------------------------------
# Applicability domain
# ---------------------------------------------------------------------------


def test_leverage_shape() -> None:
    model = SubsetModel(INDICES)
    model.fit(X_ALL, Y)
    r = model.get_results()
    assert r.leverage is not None
    assert r.leverage.shape == (N,)


def test_leverage_non_negative() -> None:
    model = SubsetModel(INDICES)
    model.fit(X_ALL, Y)
    r = model.get_results()
    assert r.leverage is not None
    assert (r.leverage >= -1e-10).all()


def test_leverage_threshold_formula() -> None:
    """h* = 3p / n."""
    model = SubsetModel(INDICES)
    model.fit(X_ALL, Y)
    r = model.get_results()
    expected = 3.0 * len(INDICES) / N
    assert r.leverage_threshold is not None
    assert_allclose(r.leverage_threshold, expected)


def test_std_residuals_shape() -> None:
    model = SubsetModel(INDICES)
    model.fit(X_ALL, Y)
    r = model.get_results()
    assert r.std_residuals is not None
    assert r.std_residuals.shape == (N,)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_set_correctly() -> None:
    model = SubsetModel(INDICES)
    model.fit(X_ALL, Y)
    r = model.get_results()
    assert r.model_type == "mlr"
    assert r.n_features == len(INDICES)
    assert r.n_train == N
    assert r.selected_descriptors == INDICES


# ---------------------------------------------------------------------------
# External test set
# ---------------------------------------------------------------------------


def test_external_metrics_populated_with_test() -> None:
    model = SubsetModel(INDICES)
    model.fit(X_ALL, Y, X_test=X_TEST, y_test=Y_TEST)
    r = model.get_results()
    for attr in (
        "q_squared_f1",
        "q_squared_f2",
        "q_squared_f3",
        "r_squared_ext",
        "press_ext",
    ):
        assert getattr(r, attr) is not None, f"{attr} should be set with test data"


def test_n_test_set_with_test() -> None:
    model = SubsetModel(INDICES)
    model.fit(X_ALL, Y, X_test=X_TEST, y_test=Y_TEST)
    r = model.get_results()
    assert r.n_test == len(Y_TEST)


def test_press_ext_positive() -> None:
    model = SubsetModel(INDICES)
    model.fit(X_ALL, Y, X_test=X_TEST, y_test=Y_TEST)
    r = model.get_results()
    assert r.press_ext is not None
    assert r.press_ext >= 0.0
