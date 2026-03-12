"""Unit tests for qsarify.utils.statistics.

All expected values were derived from manual calculations or cross-checked
against scikit-learn reference implementations where applicable.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qsarify.utils.statistics import (
    LOOResult,
    balanced_accuracy,
    ccc,
    closeness,
    closeness_prime,
    cohen_kappa,
    delta_r_m_squared,
    f1_score_binary,
    f_statistic,
    hat_values,
    k_prime_slope,
    k_slope,
    k_xx,
    k_xy,
    log_loss,
    lof,
    mae,
    mcc,
    mse,
    mss,
    npv,
    ols_hat_loo,
    precision,
    press_loo,
    q2_f1,
    q2_f2,
    q2_f3,
    q2_loo,
    r_bar_m_squared,
    r_squared,
    r_squared_0,
    r_squared_0_prime,
    r_squared_adj,
    r_squared_ext,
    r_squared_m,
    r_squared_m_prime,
    recall,
    regularized_coef_stats,
    rmse,
    rss,
    sklearn_loo,
    specificity,
    std_error,
    tss,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

Y_TRUE = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
Y_PRED = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

# Simple 5-sample external test set
Y_TRUE_EXT = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
Y_PRED_EXT = np.array([2.1, 2.9, 4.2, 4.8, 6.2])
Y_MEAN_TRAIN = 3.0  # mean of training set

# Small descriptor matrix for K_xx / K_xy tests
X_SMALL = np.array(
    [
        [1.0, 0.5, 0.2],
        [2.0, 1.0, 0.4],
        [3.0, 1.5, 0.6],
        [4.0, 2.0, 0.8],
        [5.0, 2.5, 1.0],
    ]
)
Y_SMALL = np.array([1.1, 2.2, 2.9, 4.1, 5.0])

# Classification fixtures
Y_CLS_TRUE = np.array([1, 0, 1, 1, 0, 0, 1, 0])
Y_CLS_PRED = np.array([1, 0, 1, 0, 0, 1, 1, 0])
# TP=3, TN=3, FP=1, FN=1  => accuracy=6/8=0.75


# ---------------------------------------------------------------------------
# Regression — sum-of-squares primitives
# ---------------------------------------------------------------------------


def test_rss() -> None:
    expected = np.sum((Y_TRUE - Y_PRED) ** 2)
    assert_allclose(rss(Y_TRUE, Y_PRED), expected)


def test_tss() -> None:
    expected = np.sum((Y_TRUE - Y_TRUE.mean()) ** 2)
    assert_allclose(tss(Y_TRUE), expected)


def test_mss() -> None:
    expected = np.sum((Y_PRED - Y_TRUE.mean()) ** 2)
    assert_allclose(mss(Y_TRUE, Y_PRED), expected)


def test_mae() -> None:
    expected = np.mean(np.abs(Y_TRUE - Y_PRED))
    assert_allclose(mae(Y_TRUE, Y_PRED), expected)


def test_mse() -> None:
    expected = np.mean((Y_TRUE - Y_PRED) ** 2)
    assert_allclose(mse(Y_TRUE, Y_PRED), expected)


def test_rmse() -> None:
    expected = np.sqrt(np.mean((Y_TRUE - Y_PRED) ** 2))
    assert_allclose(rmse(Y_TRUE, Y_PRED), expected)


def test_r_squared() -> None:
    expected = 1.0 - rss(Y_TRUE, Y_PRED) / tss(Y_TRUE)
    assert_allclose(r_squared(Y_TRUE, Y_PRED), expected)


def test_r_squared_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert_allclose(r_squared(y, y), 1.0)


def test_r_squared_adj() -> None:
    n = len(Y_TRUE)
    p = 2
    r2 = r_squared(Y_TRUE, Y_PRED)
    expected = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    assert_allclose(r_squared_adj(Y_TRUE, Y_PRED, n_features=p), expected)


def test_std_error() -> None:
    n, p = len(Y_TRUE), 2
    expected = np.sqrt(rss(Y_TRUE, Y_PRED) / (n - p - 1))
    assert_allclose(std_error(Y_TRUE, Y_PRED, n_features=p), expected)


def test_f_statistic() -> None:
    n, p = len(Y_TRUE), 2
    msr = mss(Y_TRUE, Y_PRED) / p
    mse_val = rss(Y_TRUE, Y_PRED) / (n - p - 1)
    expected = msr / mse_val
    assert_allclose(f_statistic(Y_TRUE, Y_PRED, n_features=p), expected)


def test_lof() -> None:
    n, p = len(Y_TRUE), 2
    mse_val = mse(Y_TRUE, Y_PRED)
    expected = mse_val / (1 - 2 * p / n) ** 2
    assert_allclose(lof(Y_TRUE, Y_PRED, n_features=p), expected)


def test_ccc() -> None:
    cov_val = np.cov(Y_TRUE, Y_PRED, ddof=0)[0, 1]
    var_y = np.var(Y_TRUE, ddof=0)
    var_yhat = np.var(Y_PRED, ddof=0)
    mean_diff = (Y_TRUE.mean() - Y_PRED.mean()) ** 2
    expected = 2 * cov_val / (var_y + var_yhat + mean_diff)
    assert_allclose(ccc(Y_TRUE, Y_PRED), expected)


# ---------------------------------------------------------------------------
# Cross-validation metrics
# ---------------------------------------------------------------------------


def test_press_loo() -> None:
    y_loo = Y_PRED + 0.05  # pretend these are LOO predictions
    expected = np.sum((Y_TRUE - y_loo) ** 2)
    assert_allclose(press_loo(Y_TRUE, y_loo), expected)


def test_q2_loo() -> None:
    y_loo = Y_PRED + 0.05
    p = press_loo(Y_TRUE, y_loo)
    t = tss(Y_TRUE)
    assert_allclose(q2_loo(Y_TRUE, y_loo), 1.0 - p / t)


# ---------------------------------------------------------------------------
# ols_hat_loo
# ---------------------------------------------------------------------------

# Larger random dataset for OLS LOO tests
_RNG_OLS = np.random.default_rng(99)
_N_OLS, _P_OLS = 40, 3
_X_OLS = _RNG_OLS.standard_normal((_N_OLS, _P_OLS))
_TRUE_COEF_OLS = np.array([2.0, -1.5, 0.8])
_Y_OLS = _X_OLS @ _TRUE_COEF_OLS + _RNG_OLS.standard_normal(_N_OLS) * 0.3


def test_ols_hat_loo_returns_loo_result() -> None:
    result = ols_hat_loo(_X_OLS, _Y_OLS)
    assert isinstance(result, LOOResult)


def test_ols_hat_loo_q2_in_range() -> None:
    result = ols_hat_loo(_X_OLS, _Y_OLS)
    assert -1.0 <= result.q2_loo <= 1.0 + 1e-9


def test_ols_hat_loo_press_nonnegative() -> None:
    result = ols_hat_loo(_X_OLS, _Y_OLS)
    assert result.press >= 0.0


def test_ols_hat_loo_y_pred_shape() -> None:
    result = ols_hat_loo(_X_OLS, _Y_OLS)
    assert result.y_pred.shape == (_N_OLS,)


def test_ols_hat_loo_q2_consistent_with_press() -> None:
    """q2_loo field must equal 1 - press / tss(y)."""
    result = ols_hat_loo(_X_OLS, _Y_OLS)
    expected_q2 = 1.0 - result.press / tss(_Y_OLS)
    assert_allclose(result.q2_loo, expected_q2, rtol=1e-9)


def test_ols_hat_loo_good_signal() -> None:
    """Strong signal data should yield Q²_LOO well above zero."""
    result = ols_hat_loo(_X_OLS, _Y_OLS)
    assert result.q2_loo > 0.5, f"Expected Q²_LOO > 0.5, got {result.q2_loo}"


def test_ols_hat_loo_y_pred_are_ols_predictions() -> None:
    """y_pred should equal the OLS training predictions (not LOO)."""
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression(fit_intercept=True).fit(_X_OLS, _Y_OLS)
    expected_y_pred = lr.predict(_X_OLS)
    result = ols_hat_loo(_X_OLS, _Y_OLS)
    assert_allclose(result.y_pred, expected_y_pred, rtol=1e-6)


# ---------------------------------------------------------------------------
# sklearn_loo
# ---------------------------------------------------------------------------


def test_sklearn_loo_returns_loo_result() -> None:
    from sklearn.linear_model import LinearRegression

    result = sklearn_loo(LinearRegression(), _X_OLS, _Y_OLS)
    assert isinstance(result, LOOResult)


def test_sklearn_loo_q2_in_range() -> None:
    from sklearn.linear_model import LinearRegression

    result = sklearn_loo(LinearRegression(), _X_OLS, _Y_OLS)
    assert -1.0 <= result.q2_loo <= 1.0 + 1e-9


def test_sklearn_loo_matches_ols_hat_loo_for_linear_regression() -> None:
    """sklearn_loo with LinearRegression must agree with ols_hat_loo."""
    from sklearn.linear_model import LinearRegression

    hat = ols_hat_loo(_X_OLS, _Y_OLS)
    sk = sklearn_loo(LinearRegression(), _X_OLS, _Y_OLS)
    assert_allclose(sk.q2_loo, hat.q2_loo, atol=1e-6)
    assert_allclose(sk.press, hat.press, atol=1e-6)
    assert_allclose(sk.y_pred, hat.y_pred, atol=1e-6)


def test_sklearn_loo_works_with_ridge() -> None:
    from sklearn.linear_model import Ridge

    result = sklearn_loo(Ridge(alpha=1.0), _X_OLS, _Y_OLS)
    assert isinstance(result, LOOResult)
    assert result.q2_loo > 0.0


def test_sklearn_loo_press_consistent_with_q2() -> None:
    from sklearn.linear_model import LinearRegression

    result = sklearn_loo(LinearRegression(), _X_OLS, _Y_OLS)
    expected_q2 = 1.0 - result.press / tss(_Y_OLS)
    assert_allclose(result.q2_loo, expected_q2, rtol=1e-9)


def test_loo_result_has_y_pred_loo_field() -> None:
    result = ols_hat_loo(_X_OLS, _Y_OLS)
    assert hasattr(result, "y_pred_loo")
    assert result.y_pred_loo.shape == (_N_OLS,)


def test_ols_hat_loo_y_pred_loo_matches_naive() -> None:
    """LOO predictions via hat-matrix must match naive n-fold refitting."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    naive_loo = np.asarray(
        cross_val_predict(LinearRegression(), _X_OLS, _Y_OLS, cv=LeaveOneOut()),
        dtype=np.float64,
    )
    result = ols_hat_loo(_X_OLS, _Y_OLS)
    assert_allclose(result.y_pred_loo, naive_loo, atol=1e-6)


def test_sklearn_loo_y_pred_loo_shape() -> None:
    from sklearn.linear_model import LinearRegression

    result = sklearn_loo(LinearRegression(), _X_OLS, _Y_OLS)
    assert result.y_pred_loo.shape == (_N_OLS,)


def test_sklearn_loo_y_pred_loo_matches_ols_hat_loo() -> None:
    from sklearn.linear_model import LinearRegression

    hat = ols_hat_loo(_X_OLS, _Y_OLS)
    sk = sklearn_loo(LinearRegression(), _X_OLS, _Y_OLS)
    assert_allclose(sk.y_pred_loo, hat.y_pred_loo, atol=1e-6)


# ---------------------------------------------------------------------------
# hat_values
# ---------------------------------------------------------------------------


_RNG_HV = np.random.default_rng(7)
_X_HV = _RNG_HV.standard_normal((20, 3))


def test_hat_values_shape() -> None:
    h = hat_values(_X_HV)
    assert h.shape == (20,)


def test_hat_values_non_negative() -> None:
    h = hat_values(_X_HV)
    assert (h >= -1e-12).all()


def test_hat_values_sum_equals_rank() -> None:
    """sum(h_i) == rank(X) for full-rank X."""
    h = hat_values(_X_HV)
    rank = np.linalg.matrix_rank(_X_HV)
    assert_allclose(h.sum(), rank, atol=1e-9)


def test_hat_values_empty_columns() -> None:
    """hat_values on a zero-column matrix returns zeros."""
    h = hat_values(np.zeros((5, 0)))
    assert h.shape == (5,)
    assert_allclose(h, 0.0)


def test_hat_values_matches_manual() -> None:
    """Cross-check with the manual formula h = diag(X (X^T X)^{-1} X^T)."""
    X = _X_HV
    XtX_inv = np.linalg.pinv(X.T @ X)
    H = X @ XtX_inv @ X.T
    expected = np.diag(H)
    assert_allclose(hat_values(X), expected, atol=1e-10)


# ---------------------------------------------------------------------------
# regularized_coef_stats
# ---------------------------------------------------------------------------


_RNG_REG = np.random.default_rng(17)
_N_REG, _P_REG = 50, 4
_X_REG = _RNG_REG.standard_normal((_N_REG, _P_REG))
_COEF_REG = np.array([1.5, -0.8, 0.4, -1.2])
_Y_REG = _X_REG @ _COEF_REG + _RNG_REG.standard_normal(_N_REG) * 0.3


def test_ridge_coef_stats_returns_arrays() -> None:
    std_err, ci, p_vals = regularized_coef_stats(_X_REG, _Y_REG, _COEF_REG, 1.0, "ridge")
    assert std_err is not None
    assert ci is not None
    assert p_vals is not None


def test_ridge_coef_stats_shapes() -> None:
    std_err, ci, p_vals = regularized_coef_stats(_X_REG, _Y_REG, _COEF_REG, 1.0, "ridge")
    assert std_err is not None and std_err.shape == (_P_REG,)
    assert ci is not None and ci.shape == (_P_REG, 2)
    assert p_vals is not None and p_vals.shape == (_P_REG,)


def test_ridge_coef_stats_std_err_non_negative() -> None:
    std_err, _, _ = regularized_coef_stats(_X_REG, _Y_REG, _COEF_REG, 1.0, "ridge")
    assert std_err is not None
    assert (std_err >= 0.0).all()


def test_ridge_coef_stats_p_values_in_range() -> None:
    _, _, p_vals = regularized_coef_stats(_X_REG, _Y_REG, _COEF_REG, 1.0, "ridge")
    assert p_vals is not None
    assert ((p_vals >= 0.0) & (p_vals <= 1.0)).all()


def test_lasso_coef_stats_active_set() -> None:
    """Lasso stats should return non-None for non-zero coefficients."""
    std_err, ci, p_vals = regularized_coef_stats(_X_REG, _Y_REG, _COEF_REG, 0.01, "lasso")
    # All coef are non-zero so active set = all; should return arrays
    assert std_err is not None
    assert std_err.shape == (_P_REG,)


def test_lasso_all_zero_returns_none() -> None:
    """All-zero Lasso coef → None stats."""
    zero_coef = np.zeros(_P_REG)
    std_err, ci, p_vals = regularized_coef_stats(_X_REG, _Y_REG, zero_coef, 100.0, "lasso")
    assert std_err is None
    assert ci is None
    assert p_vals is None


# ---------------------------------------------------------------------------
# External validation metrics
# ---------------------------------------------------------------------------


def test_q2_f1() -> None:
    press_ext = np.sum((Y_TRUE_EXT - Y_PRED_EXT) ** 2)
    ss_tr = np.sum((Y_TRUE_EXT - Y_MEAN_TRAIN) ** 2)
    expected = 1.0 - press_ext / ss_tr
    assert_allclose(q2_f1(Y_TRUE_EXT, Y_PRED_EXT, y_mean_train=Y_MEAN_TRAIN), expected)


def test_q2_f2() -> None:
    press_ext = np.sum((Y_TRUE_EXT - Y_PRED_EXT) ** 2)
    ss_ext = tss(Y_TRUE_EXT)
    expected = 1.0 - press_ext / ss_ext
    assert_allclose(q2_f2(Y_TRUE_EXT, Y_PRED_EXT), expected)


def test_q2_f3() -> None:
    n_tr = len(Y_TRUE)
    n_ext = len(Y_TRUE_EXT)
    press_ext = np.sum((Y_TRUE_EXT - Y_PRED_EXT) ** 2)
    tss_all = tss(Y_TRUE_EXT)
    expected = 1.0 - (press_ext * n_tr) / (tss_all * n_ext)
    assert_allclose(q2_f3(Y_TRUE_EXT, Y_PRED_EXT, n_train=n_tr), expected)


def test_r_squared_ext() -> None:
    press_ext = np.sum((Y_TRUE_EXT - Y_PRED_EXT) ** 2)
    ss_ext = tss(Y_TRUE_EXT)
    expected = 1.0 - press_ext / ss_ext
    assert_allclose(r_squared_ext(Y_TRUE_EXT, Y_PRED_EXT), expected)


def test_k_slope() -> None:
    expected = np.dot(Y_TRUE, Y_PRED) / np.dot(Y_PRED, Y_PRED)
    assert_allclose(k_slope(Y_TRUE, Y_PRED), expected)


def test_k_prime_slope() -> None:
    expected = np.dot(Y_TRUE, Y_PRED) / np.dot(Y_TRUE, Y_TRUE)
    assert_allclose(k_prime_slope(Y_TRUE, Y_PRED), expected)


def test_r_squared_0() -> None:
    num = np.sum((Y_TRUE - Y_PRED) ** 2)
    denom = np.sum(Y_TRUE ** 2)
    expected = 1.0 - num / denom
    assert_allclose(r_squared_0(Y_TRUE, Y_PRED), expected)


def test_r_squared_0_prime() -> None:
    num = np.sum((Y_TRUE - Y_PRED) ** 2)
    denom = np.sum(Y_PRED ** 2)
    expected = 1.0 - num / denom
    assert_allclose(r_squared_0_prime(Y_TRUE, Y_PRED), expected)


# ---------------------------------------------------------------------------
# Roy's criteria
# ---------------------------------------------------------------------------


def test_r_squared_m() -> None:
    r2 = r_squared(Y_TRUE_EXT, Y_PRED_EXT)
    r2_0 = r_squared_0(Y_TRUE_EXT, Y_PRED_EXT)
    r2_ext = r_squared_ext(Y_TRUE_EXT, Y_PRED_EXT)
    expected = r2 * (1.0 - np.sqrt(np.abs(r2_ext - r2_0)))
    assert_allclose(r_squared_m(Y_TRUE_EXT, Y_PRED_EXT), expected)


def test_r_squared_m_prime() -> None:
    r2 = r_squared(Y_TRUE_EXT, Y_PRED_EXT)
    r2_0p = r_squared_0_prime(Y_TRUE_EXT, Y_PRED_EXT)
    r2_ext = r_squared_ext(Y_TRUE_EXT, Y_PRED_EXT)
    expected = r2 * (1.0 - np.sqrt(np.abs(r2_ext - r2_0p)))
    assert_allclose(r_squared_m_prime(Y_TRUE_EXT, Y_PRED_EXT), expected)


def test_r_bar_m_squared() -> None:
    rm2 = r_squared_m(Y_TRUE_EXT, Y_PRED_EXT)
    rm2p = r_squared_m_prime(Y_TRUE_EXT, Y_PRED_EXT)
    expected = (rm2 + rm2p) / 2.0
    assert_allclose(r_bar_m_squared(rm2, rm2p), expected)


def test_delta_r_m_squared() -> None:
    rm2 = r_squared_m(Y_TRUE_EXT, Y_PRED_EXT)
    rm2p = r_squared_m_prime(Y_TRUE_EXT, Y_PRED_EXT)
    expected = np.abs(rm2 - rm2p)
    assert_allclose(delta_r_m_squared(rm2, rm2p), expected)


# ---------------------------------------------------------------------------
# Golbraikh-Tropsha closeness criteria
# ---------------------------------------------------------------------------


def test_closeness() -> None:
    r2 = r_squared(Y_TRUE_EXT, Y_PRED_EXT)
    r2_0 = r_squared_0(Y_TRUE_EXT, Y_PRED_EXT)
    expected = np.abs(r2 - r2_0) / r2
    assert_allclose(closeness(Y_TRUE_EXT, Y_PRED_EXT), expected)


def test_closeness_prime() -> None:
    r2 = r_squared(Y_TRUE_EXT, Y_PRED_EXT)
    r2_0p = r_squared_0_prime(Y_TRUE_EXT, Y_PRED_EXT)
    expected = np.abs(r2 - r2_0p) / r2
    assert_allclose(closeness_prime(Y_TRUE_EXT, Y_PRED_EXT), expected)


# ---------------------------------------------------------------------------
# Descriptor correlation metrics (QUIK)
# ---------------------------------------------------------------------------


def _expected_k(matrix: np.ndarray) -> float:
    """Reference implementation of K_xx / K_xy."""
    p = matrix.shape[1]
    cov = np.cov(matrix, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.abs(eigvals)
    total = eigvals.sum()
    if total == 0:
        return 0.0
    fracs = eigvals / total
    deviations = np.abs(fracs - 1.0 / p)
    return float(deviations.sum() / (2.0 * (p - 1) / p))


def test_k_xx_shape() -> None:
    result = k_xx(X_SMALL)
    assert isinstance(result, float)
    assert 0.0 <= result


def test_k_xx_value() -> None:
    expected = _expected_k(X_SMALL)
    assert_allclose(k_xx(X_SMALL), expected, rtol=1e-5)


def test_k_xy_value() -> None:
    X_aug = np.column_stack([X_SMALL, Y_SMALL])
    expected = _expected_k(X_aug)
    assert_allclose(k_xy(X_SMALL, Y_SMALL), expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def test_precision_value() -> None:
    # TP=3, FP=1 → 3/4 = 0.75
    assert_allclose(precision(Y_CLS_TRUE, Y_CLS_PRED), 0.75)


def test_recall_value() -> None:
    # TP=3, FN=1 → 3/4 = 0.75
    assert_allclose(recall(Y_CLS_TRUE, Y_CLS_PRED), 0.75)


def test_specificity_value() -> None:
    # TN=3, FP=1 → 3/4 = 0.75
    assert_allclose(specificity(Y_CLS_TRUE, Y_CLS_PRED), 0.75)


def test_npv_value() -> None:
    # TN=3, FN=1 → 3/4 = 0.75
    assert_allclose(npv(Y_CLS_TRUE, Y_CLS_PRED), 0.75)


def test_balanced_accuracy_value() -> None:
    # (0.75 + 0.75) / 2 = 0.75
    assert_allclose(balanced_accuracy(Y_CLS_TRUE, Y_CLS_PRED), 0.75)


def test_f1_score_binary_value() -> None:
    # 2 * (0.75 * 0.75) / (0.75 + 0.75) = 0.75
    assert_allclose(f1_score_binary(Y_CLS_TRUE, Y_CLS_PRED), 0.75)


def test_mcc_value() -> None:
    # (3*3 - 1*1) / sqrt(4*4*4*4) = 8/16 = 0.5
    assert_allclose(mcc(Y_CLS_TRUE, Y_CLS_PRED), 0.5)


def test_cohen_kappa_value() -> None:
    from sklearn.metrics import cohen_kappa_score

    expected = cohen_kappa_score(Y_CLS_TRUE, Y_CLS_PRED)
    assert_allclose(cohen_kappa(Y_CLS_TRUE, Y_CLS_PRED), expected, rtol=1e-5)


def test_log_loss_value() -> None:
    from sklearn.metrics import log_loss as sk_log_loss

    y_prob = np.array([0.9, 0.1, 0.8, 0.3, 0.1, 0.7, 0.85, 0.2])
    expected = sk_log_loss(Y_CLS_TRUE, y_prob)
    assert_allclose(log_loss(Y_CLS_TRUE, y_prob), expected, rtol=1e-5)
