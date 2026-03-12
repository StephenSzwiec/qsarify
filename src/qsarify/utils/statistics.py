"""Vectorized statistical metrics for QSAR/QSPR model evaluation.

All functions accept NumPy arrays and return scalar floats.  They are
intentionally stateless so they can be reused across the modeling, validation,
and result-display layers.

References
----------
Gramatica, P. et al. (2013). QSARINS: A new software for the development,
analysis, and validation of QSAR MLR models. J. Comput. Chem., 34, 2121–2132.

Roy, K. et al. (2012). On a simple approach for determining applicability
domain of QSAR models. Chemom. Intell. Lab. Syst., 145, 22–29.

Golbraikh, A. & Tropsha, A. (2002). Beware of q2! J. Mol. Graph. Model.,
20, 269–276.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats as scipy_stats

__all__ = [
    # Regression — sum-of-squares primitives
    "rss",
    "tss",
    "mss",
    "mae",
    "mse",
    "rmse",
    "press_loo",
    # Regression — model quality
    "r_squared",
    "r_squared_adj",
    "std_error",
    "f_statistic",
    "lof",
    "ccc",
    # LOO cross-validation
    "q2_loo",
    "LOOResult",
    "ols_hat_loo",
    "sklearn_loo",
    # Applicability domain / leverage
    "hat_values",
    # Regularized model coefficient statistics
    "regularized_coef_stats",
    # External validation
    "q2_f1",
    "q2_f2",
    "q2_f3",
    "r_squared_ext",
    "k_slope",
    "k_prime_slope",
    "r_squared_0",
    "r_squared_0_prime",
    # Roy's criteria
    "r_squared_m",
    "r_squared_m_prime",
    "r_bar_m_squared",
    "delta_r_m_squared",
    # Golbraikh-Tropsha closeness
    "closeness",
    "closeness_prime",
    # Descriptor correlation (QUIK rule)
    "k_xx",
    "k_xy",
    # Classification
    "precision",
    "recall",
    "specificity",
    "npv",
    "balanced_accuracy",
    "f1_score_binary",
    "mcc",
    "cohen_kappa",
    "log_loss",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_float_array(a: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(a, dtype=np.float64)


# ---------------------------------------------------------------------------
# Regression — sum-of-squares primitives
# ---------------------------------------------------------------------------


def rss(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Residual Sum of Squares.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.
    y_pred : array-like of shape (n,)
        Predicted response values.

    Returns
    -------
    float
        :math:`RSS = \\sum_i (y_i - \\hat{y}_i)^2`
    """
    y, yh = _to_float_array(y_true), _to_float_array(y_pred)
    return float(np.sum((y - yh) ** 2))


def tss(y_true: ArrayLike) -> float:
    """Total Sum of Squares.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.

    Returns
    -------
    float
        :math:`TSS = \\sum_i (y_i - \\bar{y})^2`
    """
    y = _to_float_array(y_true)
    return float(np.sum((y - y.mean()) ** 2))


def mss(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Model Sum of Squares (explained variance).

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values (used to compute :math:`\\bar{y}`).
    y_pred : array-like of shape (n,)
        Predicted response values.

    Returns
    -------
    float
        :math:`MSS = \\sum_i (\\hat{y}_i - \\bar{y})^2`
    """
    y, yh = _to_float_array(y_true), _to_float_array(y_pred)
    return float(np.sum((yh - y.mean()) ** 2))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.
    y_pred : array-like of shape (n,)
        Predicted response values.

    Returns
    -------
    float
        :math:`MAE = \\frac{1}{n} \\sum_i |y_i - \\hat{y}_i|`
    """
    y, yh = _to_float_array(y_true), _to_float_array(y_pred)
    return float(np.mean(np.abs(y - yh)))


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Squared Error.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.
    y_pred : array-like of shape (n,)
        Predicted response values.

    Returns
    -------
    float
        :math:`MSE = \\frac{1}{n} \\sum_i (y_i - \\hat{y}_i)^2`
    """
    y, yh = _to_float_array(y_true), _to_float_array(y_pred)
    return float(np.mean((y - yh) ** 2))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.
    y_pred : array-like of shape (n,)
        Predicted response values.

    Returns
    -------
    float
        :math:`RMSE = \\sqrt{MSE}`
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def press_loo(y_true: ArrayLike, y_pred_loo: ArrayLike) -> float:
    """Predicted Residual Error Sum of Squares (LOO).

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.
    y_pred_loo : array-like of shape (n,)
        Leave-one-out cross-validation predictions.

    Returns
    -------
    float
        :math:`PRESS = \\sum_i (y_i - \\hat{y}_{i/i})^2`
    """
    return rss(y_true, y_pred_loo)


# ---------------------------------------------------------------------------
# Regression — model quality
# ---------------------------------------------------------------------------


def r_squared(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of Determination R².

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.
    y_pred : array-like of shape (n,)
        Predicted response values.

    Returns
    -------
    float
        :math:`R^2 = 1 - RSS / TSS`
    """
    return 1.0 - rss(y_true, y_pred) / tss(y_true)


def r_squared_adj(y_true: ArrayLike, y_pred: ArrayLike, n_features: int) -> float:
    """Adjusted R².

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.
    y_pred : array-like of shape (n,)
        Predicted response values.
    n_features : int
        Number of predictor variables *p*.

    Returns
    -------
    float
        :math:`R^2_{adj} = 1 - (1 - R^2)(n-1) / (n - p - 1)`
    """
    y = _to_float_array(y_true)
    n = len(y)
    r2 = r_squared(y_true, y_pred)
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1))


def std_error(y_true: ArrayLike, y_pred: ArrayLike, n_features: int) -> float:
    """Standard Error of the Regression Estimate.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.
    y_pred : array-like of shape (n,)
        Predicted response values.
    n_features : int
        Number of predictor variables *p*.

    Returns
    -------
    float
        :math:`s = \\sqrt{RSS / (n - p - 1)}`
    """
    y = _to_float_array(y_true)
    n = len(y)
    return float(np.sqrt(rss(y_true, y_pred) / (n - n_features - 1)))


def f_statistic(y_true: ArrayLike, y_pred: ArrayLike, n_features: int) -> float:
    """F-Statistic for overall regression significance.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.
    y_pred : array-like of shape (n,)
        Predicted response values.
    n_features : int
        Number of predictor variables *p*.

    Returns
    -------
    float
        :math:`F = (MSS/p) / (RSS/(n-p-1))`
    """
    y = _to_float_array(y_true)
    n = len(y)
    p = n_features
    msr = mss(y_true, y_pred) / p
    mse_val = rss(y_true, y_pred) / (n - p - 1)
    return float(msr / mse_val)


def lof(y_true: ArrayLike, y_pred: ArrayLike, n_features: int) -> float:
    """Friedman's Lack-of-Fit statistic.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.
    y_pred : array-like of shape (n,)
        Predicted response values.
    n_features : int
        Number of predictor variables *p*.

    Returns
    -------
    float
        :math:`LOF = MSE / (1 - 2p/n)^2`
    """
    y = _to_float_array(y_true)
    n = len(y)
    p = n_features
    mse_val = mse(y_true, y_pred)
    return float(mse_val / (1.0 - 2.0 * p / n) ** 2)


def ccc(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Concordance Correlation Coefficient.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.
    y_pred : array-like of shape (n,)
        Predicted response values.

    Returns
    -------
    float
        :math:`CCC = 2 Cov(y, \\hat{y}) / (Var(y) + Var(\\hat{y}) + (\\bar{y} - \\bar{\\hat{y}})^2)`
    """
    y, yh = _to_float_array(y_true), _to_float_array(y_pred)
    cov_val = float(np.cov(y, yh, ddof=0)[0, 1])
    var_y = float(np.var(y, ddof=0))
    var_yh = float(np.var(yh, ddof=0))
    mean_diff_sq = float((y.mean() - yh.mean()) ** 2)
    return float(2.0 * cov_val / (var_y + var_yh + mean_diff_sq))


# ---------------------------------------------------------------------------
# LOO cross-validation
# ---------------------------------------------------------------------------


def q2_loo(y_true: ArrayLike, y_pred_loo: ArrayLike) -> float:
    """Leave-One-Out cross-validated Q².

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.
    y_pred_loo : array-like of shape (n,)
        LOO cross-validation predictions.

    Returns
    -------
    float
        :math:`Q^2_{LOO} = 1 - PRESS / TSS`
    """
    return 1.0 - press_loo(y_true, y_pred_loo) / tss(y_true)


class LOOResult(NamedTuple):
    """Result of a leave-one-out cross-validation computation.

    Attributes
    ----------
    q2_loo : float
        Q²_LOO = 1 − PRESS / TSS.
    y_pred : ndarray of shape (n,)
        Training predictions from the model fitted on the full dataset.
    y_pred_loo : ndarray of shape (n,)
        Leave-one-out cross-validation predictions.
    press : float
        Predicted residual error sum of squares (LOO).
    """

    q2_loo: float
    y_pred: NDArray[np.float64]
    y_pred_loo: NDArray[np.float64]
    press: float


def ols_hat_loo(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> LOOResult:
    """OLS leave-one-out Q² via the hat-matrix shortcut.

    Fits OLS with an intercept once and computes all *n* LOO residuals in
    O(n²) time using the leverage values (diagonal of the hat matrix),
    avoiding *n* separate model fits.

    .. math::

        \\text{LOO residual}_i = \\frac{e_i}{1 - h_i}

    where :math:`e_i = y_i - \\hat{y}_i` is the training residual and
    :math:`h_i` is the *i*-th leverage value.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Descriptor matrix (without intercept column).
    y : ndarray of shape (n,)
        Response vector.

    Returns
    -------
    LOOResult
        Named tuple with ``q2_loo``, ``y_pred`` (full-fit training
        predictions), and ``press``.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n = len(y_arr)
    X_aug = np.column_stack([np.ones(n), X_arr])
    XtX_inv: NDArray[np.float64] = np.linalg.pinv(X_aug.T @ X_aug)
    coef: NDArray[np.float64] = XtX_inv @ (X_aug.T @ y_arr)
    y_pred_arr: NDArray[np.float64] = X_aug @ coef
    residuals: NDArray[np.float64] = y_arr - y_pred_arr
    h: NDArray[np.float64] = (X_aug @ XtX_inv * X_aug).sum(axis=1)
    denom: NDArray[np.float64] = np.clip(1.0 - h, 1e-12, None)
    y_pred_loo_arr: NDArray[np.float64] = y_arr - residuals / denom
    press_val = float(np.sum((residuals / denom) ** 2))
    tss_val = tss(y_arr)
    q2_val = 1.0 - press_val / tss_val if tss_val > 0.0 else 0.0
    return LOOResult(q2_loo=float(q2_val), y_pred=y_pred_arr, y_pred_loo=y_pred_loo_arr, press=press_val)


def sklearn_loo(
    estimator: Any,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    n_jobs: int = 1,
) -> LOOResult:
    """General leave-one-out Q² for any scikit-learn compatible estimator.

    Computes LOO predictions via
    :func:`sklearn.model_selection.cross_val_predict` with
    :class:`sklearn.model_selection.LeaveOneOut`, then fits the estimator
    once on the full training set for training predictions.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Any object implementing ``fit(X, y)`` and ``predict(X)``.
        The estimator is cloned internally; do not pass a pre-fitted instance.
    X : ndarray of shape (n, p)
        Descriptor matrix.
    y : ndarray of shape (n,)
        Response vector.
    n_jobs : int, optional
        Number of parallel jobs for LOO CV.  Default 1.

    Returns
    -------
    LOOResult
        Named tuple with ``q2_loo``, ``y_pred`` (full-fit training
        predictions), ``y_pred_loo``, and ``press``.
    """
    from sklearn.base import clone
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    y_pred_loo_arr: NDArray[np.float64] = np.asarray(
        cross_val_predict(estimator, X_arr, y_arr, cv=LeaveOneOut(), n_jobs=n_jobs),
        dtype=np.float64,
    )
    est_full = clone(estimator)
    est_full.fit(X_arr, y_arr)
    y_pred_arr: NDArray[np.float64] = np.asarray(
        est_full.predict(X_arr), dtype=np.float64
    )
    press_val = float(np.sum((y_arr - y_pred_loo_arr) ** 2))
    tss_val = tss(y_arr)
    q2_val = 1.0 - press_val / tss_val if tss_val > 0.0 else 0.0
    return LOOResult(q2_loo=float(q2_val), y_pred=y_pred_arr, y_pred_loo=y_pred_loo_arr, press=press_val)


# ---------------------------------------------------------------------------
# Applicability domain — leverage
# ---------------------------------------------------------------------------


def hat_values(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Leverage values h_i = x_i (X^T X)^{-1} x_i^T for all rows of X.

    Used to define the applicability domain (Williams plot) for any model
    type.  Computed without an intercept column, consistent with QSARINS.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Descriptor matrix (no intercept column).

    Returns
    -------
    ndarray of shape (n,)
        Leverage values h_i ∈ [0, 1].
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.shape[1] == 0:
        return np.zeros(X_arr.shape[0], dtype=np.float64)
    XtX_inv = np.asarray(np.linalg.pinv(X_arr.T @ X_arr), dtype=np.float64)
    M: NDArray[np.float64] = X_arr @ XtX_inv
    return np.asarray((M * X_arr).sum(axis=1), dtype=np.float64)


# ---------------------------------------------------------------------------
# Regularized model coefficient statistics
# ---------------------------------------------------------------------------


def regularized_coef_stats(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    coef: NDArray[np.float64],
    alpha: float,
    model_type: str,
) -> tuple[
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
]:
    """Approximate coefficient statistics for Ridge or Lasso.

    Uses OLS-like formula with the regularised normal equations:
    ``Var(β) ≈ σ² (X^T X + αI)^{-1}``.

    For Lasso the active-set approximation is applied (coefficients computed
    on the non-zero subset).  This is a standard approximation in QSAR
    software.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Centered descriptor matrix (no intercept column).
    y : ndarray of shape (n,)
        Centred response vector (intercept removed).
    coef : ndarray of shape (p,)
        Fitted regression coefficients (no intercept).
    alpha : float
        Regularisation strength.
    model_type : str
        ``'ridge'`` or ``'lasso'``.

    Returns
    -------
    tuple of (std_errors, confidence_intervals, p_values)
        Each is an ndarray of shape (p,) / (p, 2), or ``None`` if
        computation is not possible (e.g. all-zero Lasso solution).
    """
    n, p = X.shape
    residuals = y - X @ coef
    rss_val = float(residuals @ residuals)
    df_resid = max(1, n - p - 1)
    s2 = rss_val / df_resid

    if model_type == "ridge":
        XtX_reg_inv = np.asarray(np.linalg.pinv(X.T @ X + alpha * np.eye(p)), dtype=np.float64)
        coef_var: NDArray[np.float64] = np.diag(XtX_reg_inv) * s2
        std_err: NDArray[np.float64] = np.sqrt(np.clip(coef_var, 0.0, None))
        df = df_resid
        t_stats: NDArray[np.float64] = coef / np.where(std_err > 0, std_err, 1e-300)
        p_vals: NDArray[np.float64] = 2.0 * scipy_stats.t.sf(np.abs(t_stats), df=df)
        t_crit = float(scipy_stats.t.ppf(0.975, df=df))
        ci: NDArray[np.float64] = np.column_stack(
            [coef - t_crit * std_err, coef + t_crit * std_err]
        )
        return std_err, ci, p_vals

    # Lasso: active-set approximation on non-zero coefficients
    active: NDArray[np.bool_] = np.abs(coef) > 1e-10
    if active.sum() == 0:
        return None, None, None
    X_active: NDArray[np.float64] = X[:, active]
    p_active = int(X_active.shape[1])
    res_active = y - X_active @ coef[active]
    rss_active = float(res_active @ res_active)
    s2_active = rss_active / max(1, n - p_active - 1)
    XtX_active_inv = np.asarray(np.linalg.pinv(X_active.T @ X_active), dtype=np.float64)
    coef_var_active: NDArray[np.float64] = np.diag(XtX_active_inv) * s2_active
    std_err_active: NDArray[np.float64] = np.sqrt(np.clip(coef_var_active, 0.0, None))
    df_active = max(1, n - p_active - 1)
    t_active: NDArray[np.float64] = coef[active] / np.where(std_err_active > 0, std_err_active, 1e-300)
    p_active_vals: NDArray[np.float64] = 2.0 * scipy_stats.t.sf(np.abs(t_active), df=df_active)
    t_crit_active = float(scipy_stats.t.ppf(0.975, df=df_active))
    ci_active: NDArray[np.float64] = np.column_stack(
        [coef[active] - t_crit_active * std_err_active,
         coef[active] + t_crit_active * std_err_active]
    )
    # Expand to full size (zeros / ones for inactive coefficients)
    std_err_full: NDArray[np.float64] = np.zeros(p)
    p_vals_full: NDArray[np.float64] = np.ones(p)
    ci_full: NDArray[np.float64] = np.column_stack([coef, coef])
    std_err_full[active] = std_err_active
    p_vals_full[active] = p_active_vals
    ci_full[active] = ci_active
    return std_err_full, ci_full, p_vals_full


# ---------------------------------------------------------------------------
# External validation metrics
# ---------------------------------------------------------------------------


def q2_f1(y_true_ext: ArrayLike, y_pred_ext: ArrayLike, y_mean_train: float) -> float:
    """External Q²_F1 (Schüürmann et al. definition).

    Parameters
    ----------
    y_true_ext : array-like of shape (n_ext,)
        Observed response values in the external test set.
    y_pred_ext : array-like of shape (n_ext,)
        Predicted response values for the external test set.
    y_mean_train : float
        Mean of the *training* set response values.

    Returns
    -------
    float
        :math:`Q^2_{F1} = 1 - PRESS_{ext} / SS(\\bar{y}_{TR})`
    """
    y_ext = _to_float_array(y_true_ext)
    press_ext = rss(y_true_ext, y_pred_ext)
    ss_tr = float(np.sum((y_ext - y_mean_train) ** 2))
    return float(1.0 - press_ext / ss_tr)


def q2_f2(y_true_ext: ArrayLike, y_pred_ext: ArrayLike) -> float:
    """External Q²_F2.

    Parameters
    ----------
    y_true_ext : array-like of shape (n_ext,)
        Observed response values in the external test set.
    y_pred_ext : array-like of shape (n_ext,)
        Predicted response values for the external test set.

    Returns
    -------
    float
        :math:`Q^2_{F2} = 1 - PRESS_{ext} / SS(\\bar{y}_{ext})`
    """
    press_ext = rss(y_true_ext, y_pred_ext)
    ss_ext = tss(y_true_ext)
    return float(1.0 - press_ext / ss_ext)


def q2_f3(y_true_ext: ArrayLike, y_pred_ext: ArrayLike, n_train: int) -> float:
    """External Q²_F3 (sample-size adjusted).

    Parameters
    ----------
    y_true_ext : array-like of shape (n_ext,)
        Observed response values in the external test set.
    y_pred_ext : array-like of shape (n_ext,)
        Predicted response values for the external test set.
    n_train : int
        Number of training set observations.

    Returns
    -------
    float
        :math:`Q^2_{F3} = 1 - (PRESS_{ext} \\cdot n_{TR}) / (TSS \\cdot n_{ext})`
    """
    y_ext = _to_float_array(y_true_ext)
    n_ext = len(y_ext)
    press_ext = rss(y_true_ext, y_pred_ext)
    tss_val = tss(y_true_ext)
    return float(1.0 - (press_ext * n_train) / (tss_val * n_ext))


def r_squared_ext(y_true_ext: ArrayLike, y_pred_ext: ArrayLike) -> float:
    """External R² on the test set.

    Parameters
    ----------
    y_true_ext : array-like of shape (n_ext,)
        Observed response values in the external test set.
    y_pred_ext : array-like of shape (n_ext,)
        Predicted response values for the external test set.

    Returns
    -------
    float
        :math:`R^2_{ext} = 1 - PRESS_{ext} / SS(\\bar{y}_{ext})`
    """
    return q2_f2(y_true_ext, y_pred_ext)


def k_slope(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Slope *k* through origin with predicted values on x-axis.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.
    y_pred : array-like of shape (n,)
        Predicted response values.

    Returns
    -------
    float
        :math:`k = \\sum y_i \\hat{y}_i / \\sum \\hat{y}_i^2`
    """
    y, yh = _to_float_array(y_true), _to_float_array(y_pred)
    return float(np.dot(y, yh) / np.dot(yh, yh))


def k_prime_slope(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Reverse slope *k'* through origin with observed values on x-axis.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.
    y_pred : array-like of shape (n,)
        Predicted response values.

    Returns
    -------
    float
        :math:`k' = \\sum y_i \\hat{y}_i / \\sum y_i^2`
    """
    y, yh = _to_float_array(y_true), _to_float_array(y_pred)
    return float(np.dot(y, yh) / np.dot(y, y))


def r_squared_0(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """R²₀ through origin (observed as reference).

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.
    y_pred : array-like of shape (n,)
        Predicted response values.

    Returns
    -------
    float
        :math:`R^2_0 = 1 - \\sum(y_i - \\hat{y}_i)^2 / \\sum y_i^2`
    """
    y, yh = _to_float_array(y_true), _to_float_array(y_pred)
    return float(1.0 - np.sum((y - yh) ** 2) / np.sum(y ** 2))


def r_squared_0_prime(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """R'²₀ through origin (predicted as reference).

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.
    y_pred : array-like of shape (n,)
        Predicted response values.

    Returns
    -------
    float
        :math:`R'^2_0 = 1 - \\sum(y_i - \\hat{y}_i)^2 / \\sum \\hat{y}_i^2`
    """
    y, yh = _to_float_array(y_true), _to_float_array(y_pred)
    return float(1.0 - np.sum((y - yh) ** 2) / np.sum(yh ** 2))


# ---------------------------------------------------------------------------
# Roy's criteria
# ---------------------------------------------------------------------------


def r_squared_m(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Roy's r²_m metric.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.
    y_pred : array-like of shape (n,)
        Predicted response values.

    Returns
    -------
    float
        :math:`r^2_m = R^2 (1 - \\sqrt{|R^2_{ext} - R^2_0|})`
    """
    r2 = r_squared(y_true, y_pred)
    r2_ext = r_squared_ext(y_true, y_pred)
    r2_0 = r_squared_0(y_true, y_pred)
    return float(r2 * (1.0 - np.sqrt(np.abs(r2_ext - r2_0))))


def r_squared_m_prime(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Roy's r'²_m metric.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.
    y_pred : array-like of shape (n,)
        Predicted response values.

    Returns
    -------
    float
        :math:`r'^2_m = R^2 (1 - \\sqrt{|R^2_{ext} - R'^2_0|})`
    """
    r2 = r_squared(y_true, y_pred)
    r2_ext = r_squared_ext(y_true, y_pred)
    r2_0p = r_squared_0_prime(y_true, y_pred)
    return float(r2 * (1.0 - np.sqrt(np.abs(r2_ext - r2_0p))))


def r_bar_m_squared(rm2: float, rm2_prime: float) -> float:
    """Roy's averaged r̄²_m.

    Parameters
    ----------
    rm2 : float
        :math:`r^2_m` value.
    rm2_prime : float
        :math:`r'^2_m` value.

    Returns
    -------
    float
        :math:`\\bar{r}^2_m = (r^2_m + r'^2_m) / 2`
    """
    return float((rm2 + rm2_prime) / 2.0)


def delta_r_m_squared(rm2: float, rm2_prime: float) -> float:
    """Roy's Δr²_m spread.

    Parameters
    ----------
    rm2 : float
        :math:`r^2_m` value.
    rm2_prime : float
        :math:`r'^2_m` value.

    Returns
    -------
    float
        :math:`\\Delta r^2_m = |r^2_m - r'^2_m|`
    """
    return float(np.abs(rm2 - rm2_prime))


# ---------------------------------------------------------------------------
# Golbraikh-Tropsha closeness criteria
# ---------------------------------------------------------------------------


def closeness(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Golbraikh-Tropsha closeness criterion *clos*.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.
    y_pred : array-like of shape (n,)
        Predicted response values.

    Returns
    -------
    float
        :math:`clos = |R^2 - R^2_0| / R^2`
    """
    r2 = r_squared(y_true, y_pred)
    r2_0 = r_squared_0(y_true, y_pred)
    return float(np.abs(r2 - r2_0) / r2)


def closeness_prime(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Golbraikh-Tropsha closeness criterion *clos'*.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Observed response values.
    y_pred : array-like of shape (n,)
        Predicted response values.

    Returns
    -------
    float
        :math:`clos' = |R^2 - R'^2_0| / R^2`
    """
    r2 = r_squared(y_true, y_pred)
    r2_0p = r_squared_0_prime(y_true, y_pred)
    return float(np.abs(r2 - r2_0p) / r2)


# ---------------------------------------------------------------------------
# Descriptor correlation metrics (QUIK rule support)
# ---------------------------------------------------------------------------


def _k_statistic(matrix: NDArray[np.float64]) -> float:
    """Compute the K_xx / K_xy correlation statistic for *matrix*.

    Uses the eigenvalue-based formula from Gramatica (2013):

    .. math::

        K = \\frac{\\sum_{i=1}^{p} \\left|
            \\frac{\\lambda_i}{\\sum_j \\lambda_j} - \\frac{1}{p}
            \\right|}{\\frac{2(p-1)}{p}}

    Parameters
    ----------
    matrix : ndarray of shape (n, p)
        Centered (or raw) numeric matrix.

    Returns
    -------
    float
        K statistic in [0, 1].
    """
    p = matrix.shape[1]
    if p < 2:
        return 0.0
    cov = np.cov(matrix, rowvar=False)
    eigvals = np.abs(np.linalg.eigvalsh(cov))
    total = eigvals.sum()
    if total == 0.0:
        return 0.0
    fracs = eigvals / total
    deviations = np.abs(fracs - 1.0 / p)
    return float(deviations.sum() / (2.0 * (p - 1) / p))


def k_xx(X: ArrayLike) -> float:
    """Descriptor inter-correlation statistic K_xx.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Descriptor matrix.

    Returns
    -------
    float
        K_xx in [0, 1]; higher values indicate more collinearity.
    """
    return _k_statistic(_to_float_array(X))


def k_xy(X: ArrayLike, y: ArrayLike) -> float:
    """Augmented descriptor-response correlation statistic K_xy.

    Computes the K statistic on the matrix [X | y].

    Parameters
    ----------
    X : array-like of shape (n, p)
        Descriptor matrix.
    y : array-like of shape (n,)
        Response vector.

    Returns
    -------
    float
        K_xy in [0, 1].
    """
    X_arr = _to_float_array(X)
    y_arr = _to_float_array(y).reshape(-1, 1)
    X_aug = np.hstack([X_arr, y_arr])
    return _k_statistic(X_aug)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def _confusion_counts(
    y_true: NDArray[np.int_], y_pred: NDArray[np.int_]
) -> tuple[int, int, int, int]:
    """Return (TP, TN, FP, FN) for binary classification."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def precision(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Precision (Positive Predictive Value).

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True binary labels (0/1).
    y_pred : array-like of shape (n,)
        Predicted binary labels (0/1).

    Returns
    -------
    float
        :math:`Precision = TP / (TP + FP)`
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    tp, _, fp, _ = _confusion_counts(yt, yp)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Recall (Sensitivity / True Positive Rate).

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True binary labels (0/1).
    y_pred : array-like of shape (n,)
        Predicted binary labels (0/1).

    Returns
    -------
    float
        :math:`Recall = TP / (TP + FN)`
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    tp, _, _, fn = _confusion_counts(yt, yp)
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def specificity(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Specificity (True Negative Rate).

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True binary labels (0/1).
    y_pred : array-like of shape (n,)
        Predicted binary labels (0/1).

    Returns
    -------
    float
        :math:`Specificity = TN / (TN + FP)`
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    _, tn, fp, _ = _confusion_counts(yt, yp)
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def npv(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Negative Predictive Value.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True binary labels (0/1).
    y_pred : array-like of shape (n,)
        Predicted binary labels (0/1).

    Returns
    -------
    float
        :math:`NPV = TN / (TN + FN)`
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    _, tn, _, fn = _confusion_counts(yt, yp)
    return float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0


def balanced_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Balanced Accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True binary labels (0/1).
    y_pred : array-like of shape (n,)
        Predicted binary labels (0/1).

    Returns
    -------
    float
        :math:`BA = (Sensitivity + Specificity) / 2`
    """
    return float((recall(y_true, y_pred) + specificity(y_true, y_pred)) / 2.0)


def f1_score_binary(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """F1 Score for binary classification.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True binary labels (0/1).
    y_pred : array-like of shape (n,)
        Predicted binary labels (0/1).

    Returns
    -------
    float
        :math:`F1 = 2 \\cdot Precision \\cdot Recall / (Precision + Recall)`
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    denom = prec + rec
    return float(2.0 * prec * rec / denom) if denom > 0 else 0.0


def mcc(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Matthews Correlation Coefficient.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True binary labels (0/1).
    y_pred : array-like of shape (n,)
        Predicted binary labels (0/1).

    Returns
    -------
    float
        MCC in [-1, 1].
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    tp, tn, fp, fn = _confusion_counts(yt, yp)
    num = tp * tn - fp * fn
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float(num / denom) if denom > 0 else 0.0


def cohen_kappa(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Cohen's Kappa coefficient.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True binary labels (0/1).
    y_pred : array-like of shape (n,)
        Predicted binary labels (0/1).

    Returns
    -------
    float
        Kappa in [-1, 1].
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    n = len(yt)
    tp, tn, fp, fn = _confusion_counts(yt, yp)
    p_o = (tp + tn) / n
    p_yes = ((tp + fn) / n) * ((tp + fp) / n)
    p_no = ((tn + fp) / n) * ((tn + fn) / n)
    p_e = p_yes + p_no
    return float((p_o - p_e) / (1.0 - p_e)) if (1.0 - p_e) != 0 else 0.0


def log_loss(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Binary Log Loss (Cross-Entropy).

    Parameters
    ----------
    y_true : array-like of shape (n,)
        True binary labels (0/1).
    y_prob : array-like of shape (n,)
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        :math:`\\text{Log Loss} = -\\frac{1}{n} \\sum_i [y_i \\log p_i + (1-y_i) \\log(1-p_i)]`
    """
    y = _to_float_array(y_true)
    p = np.clip(_to_float_array(y_prob), 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
