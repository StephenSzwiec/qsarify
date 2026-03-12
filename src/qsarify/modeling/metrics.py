"""Shared metric assembly for all QSARify model types.

:func:`compute_model_result` is the **single point** where training
arrays, LOO predictions, and optional external-test data are turned into a
fully populated :class:`~qsarify.results.model_result.ModelResult`.  Both
:class:`~qsarify.modeling.subset_model.SubsetModel` and the sklearn
:class:`~qsarify.modeling.models.FullModel` wrappers call this function,
ensuring that metric computation is uniform across the entire library.

All heavy numerical work is delegated to :mod:`qsarify.utils.statistics`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from qsarify.results.model_result import ModelResult
from qsarify.utils import statistics as stat

__all__ = ["compute_model_result"]


def compute_model_result(
    *,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    y_pred_loo: NDArray[np.float64],
    model_type: str,
    hyperparameters: dict[str, object],
    # Optional external test set
    y_test: NDArray[np.float64] | None = None,
    y_pred_test: NDArray[np.float64] | None = None,
    # Optional pre-computed applicability-domain leverage
    leverage: NDArray[np.float64] | None = None,
    # Linear-model-only coefficient statistics
    coef_std_errors: NDArray[np.float64] | None = None,
    coef_confidence_intervals: NDArray[np.float64] | None = None,
    coef_p_values: NDArray[np.float64] | None = None,
    # SubsetModel metadata
    selected_descriptors: list[int] | None = None,
    cluster_assignments: dict[int, int] | None = None,
) -> ModelResult:
    """Assemble a :class:`ModelResult` from training arrays.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Training descriptor matrix (no intercept column).
    y : ndarray of shape (n,)
        Training response vector.
    y_pred : ndarray of shape (n,)
        Full-fit training predictions.
    y_pred_loo : ndarray of shape (n,)
        LOO cross-validation predictions.
    model_type : str
        One of ``'mlr'``, ``'ridge'``, ``'lasso'``, ``'svr'``, ``'rf'``,
        ``'gbr'``.
    hyperparameters : dict
        Model hyperparameters to record.
    y_test : ndarray of shape (n_test,) or None
        External test set observed values.
    y_pred_test : ndarray of shape (n_test,) or None
        External test set predictions.
    leverage : ndarray of shape (n,) or None
        Pre-computed applicability-domain leverage values.  When ``None``,
        computed from *X* via ``stat.hat_values(X)``.  Callers that already
        have leverage (e.g. :class:`SubsetModel` via the OLS hat matrix)
        should pass it to avoid redundant computation.
    coef_std_errors, coef_confidence_intervals, coef_p_values
        Coefficient-level statistics for linear models.  Set to ``None``
        for non-linear model types.
    selected_descriptors : list[int] or None
        Column indices into the full descriptor matrix (SubsetModel only).
    cluster_assignments : dict[int, int] or None
        Descriptor-index → cluster-ID mapping (SubsetModel only).

    Returns
    -------
    ModelResult
        Fully populated result object.
    """
    n, p = X.shape

    # --- Applicability domain ---
    if leverage is None:
        leverage = stat.hat_values(X)
    h_star = 3.0 * p / n

    # --- Training metrics (universal) ---
    r2 = stat.r_squared(y, y_pred)
    rss_val = stat.rss(y, y_pred)
    tss_val = stat.tss(y)

    df_resid = n - p - 1
    s2 = rss_val / df_resid if df_resid > 0 else 0.0
    s = float(np.sqrt(s2))

    f_val: float | None = None
    if df_resid > 0 and p > 0:
        f_val = stat.f_statistic(y, y_pred, n_features=p)

    lof_val: float | None = None
    if n > 2 * p:
        lof_val = stat.lof(y, y_pred, n_features=p)

    rm2 = stat.r_squared_m(y, y_pred)
    rm2p = stat.r_squared_m_prime(y, y_pred)

    std_resid: NDArray[np.float64] = (
        (y - y_pred) / s if s > 0.0 else (y - y_pred).copy()
    )

    result = ModelResult(
        # Training metrics
        r_squared=r2,
        r_squared_adj=stat.r_squared_adj(y, y_pred, n_features=p),
        rmse=stat.rmse(y, y_pred),
        mae=stat.mae(y, y_pred),
        mse=stat.mse(y, y_pred),
        rss=rss_val,
        tss=tss_val,
        mss=stat.mss(y, y_pred),
        std_error_estimate=s,
        f_statistic=f_val,
        lof=lof_val,
        ccc=stat.ccc(y, y_pred),
        q_squared_loo=stat.q2_loo(y, y_pred_loo),
        # Origin regression diagnostics
        slope_origin=stat.k_slope(y, y_pred),
        slope_origin_reverse=stat.k_prime_slope(y, y_pred),
        r_squared_origin=stat.r_squared_0(y, y_pred),
        r_squared_origin_reverse=stat.r_squared_0_prime(y, y_pred),
        # Roy's criteria
        roy_r_squared_m_mean=stat.r_bar_m_squared(rm2, rm2p),
        roy_r_squared_m_delta=stat.delta_r_m_squared(rm2, rm2p),
        # Golbraikh-Tropsha closeness
        closeness=stat.closeness(y, y_pred),
        closeness_reverse=stat.closeness_prime(y, y_pred),
        # Coefficient statistics (None for non-linear)
        coef_std_errors=coef_std_errors,
        coef_confidence_intervals=coef_confidence_intervals,
        coef_p_values=coef_p_values,
        # Applicability domain
        leverage=leverage,
        std_residuals=std_resid,
        leverage_threshold=h_star,
        # Metadata
        model_type=model_type,
        n_features=p,
        n_train=n,
        hyperparameters=hyperparameters,
        selected_descriptors=selected_descriptors,
        cluster_assignments=cluster_assignments,
        y_train=y.copy(),
        y_pred_train=y_pred.copy(),
    )

    # --- External validation metrics ---
    if y_test is not None and y_pred_test is not None:
        result.n_test = len(y_test)
        result.press_ext = float(stat.rss(y_test, y_pred_test))
        result.q_squared_f1 = stat.q2_f1(y_test, y_pred_test, float(y.mean()))
        result.q_squared_f2 = stat.q2_f2(y_test, y_pred_test)
        result.q_squared_f3 = stat.q2_f3(y_test, y_pred_test, n_train=n)
        result.r_squared_ext = stat.r_squared_ext(y_test, y_pred_test)

    return result
