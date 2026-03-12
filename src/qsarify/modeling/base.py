"""Base model interface and SubsetModel (OLS/MLR) for QSARify.

Defines:

- :class:`BaseModel` — abstract base class all models must implement.
- :class:`SubsetModel` — concrete MLR model that operates on a fixed
  descriptor subset, computing the full suite of QSARINS metrics including
  LOO cross-validation via the efficient hat-matrix trick.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from qsarify.results.model_result import ModelResult
from qsarify.utils import statistics as stat

if TYPE_CHECKING:
    pass

__all__ = ["BaseModel", "SubsetModel"]


# ---------------------------------------------------------------------------
# BaseModel ABC
# ---------------------------------------------------------------------------


class BaseModel(abc.ABC):
    """Abstract base class for all QSARify models.

    All subclasses must implement :meth:`fit`, :meth:`predict`, and
    :meth:`get_results`.
    """

    @abc.abstractmethod
    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "BaseModel":
        """Train the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Descriptor matrix.
        y : ndarray of shape (n_samples,)
            Response vector.

        Returns
        -------
        BaseModel
            *self*, for method chaining.
        """
        ...

    @abc.abstractmethod
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return predicted response values.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Descriptor matrix.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.
        """
        ...

    @abc.abstractmethod
    def get_results(self) -> ModelResult:
        """Return a :class:`~qsarify.results.model_result.ModelResult` with all metrics.

        Returns
        -------
        ModelResult
            All computed metrics for this model.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        ...

    @staticmethod
    def _build_result(
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        y_pred: NDArray[np.float64],
        y_pred_loo: NDArray[np.float64],
        model_type: str,
        hyperparameters: dict[str, object],
        X_test: NDArray[np.float64] | None = None,
        y_test: NDArray[np.float64] | None = None,
        y_pred_test: NDArray[np.float64] | None = None,
        coef_std_errors: NDArray[np.float64] | None = None,
        coef_ci: NDArray[np.float64] | None = None,
        coef_p_values: NDArray[np.float64] | None = None,
        selected_descriptors: list[int] | None = None,
        cluster_assignments: dict[int, int] | None = None,
    ) -> ModelResult:
        """Assemble a :class:`ModelResult` from training arrays.

        Single point where all model types build their result, ensuring metric
        computation is uniform across the library.  Replaces both
        ``_build_model_result_train`` (OLS) and ``_compute_common_metrics``
        (FullModel wrappers).

        Parameters
        ----------
        X : ndarray of shape (n, p)
            Training descriptor matrix (no intercept).
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
        X_test, y_test : ndarray or None
            External test set.
        y_pred_test : ndarray or None
            Predictions on *X_test*.
        coef_std_errors, coef_ci, coef_p_values : ndarray or None
            Coefficient statistics (None for non-linear models).
        selected_descriptors : list[int] or None
            Column indices (SubsetModel only).
        cluster_assignments : dict or None
            Descriptor→cluster mapping (SubsetModel only).

        Returns
        -------
        ModelResult
            Fully populated result object.
        """
        n, p = X.shape

        h_ad: NDArray[np.float64] = stat.hat_values(X)
        h_star = 3.0 * p / n

        r2 = stat.r_squared(y, y_pred)
        r2_adj = stat.r_squared_adj(y, y_pred, n_features=p)
        rmse_val = stat.rmse(y, y_pred)
        mae_val = stat.mae(y, y_pred)
        mse_val = stat.mse(y, y_pred)
        rss_val = stat.rss(y, y_pred)
        mss_val = stat.mss(y, y_pred)
        tss_val = stat.tss(y)

        s2 = rss_val / (n - p - 1) if (n - p - 1) > 0 else 0.0
        s = float(np.sqrt(s2))

        f_val: float | None = None
        if n - p - 1 > 0 and p > 0:
            f_val = stat.f_statistic(y, y_pred, n_features=p)

        lof_val: float | None = None
        if n > 2 * p:
            lof_val = stat.lof(y, y_pred, n_features=p)

        ccc_val = stat.ccc(y, y_pred)
        q2_loo_val = stat.q2_loo(y, y_pred_loo)
        rm2 = stat.r_squared_m(y, y_pred)
        rm2p = stat.r_squared_m_prime(y, y_pred)
        std_resid: NDArray[np.float64] = (y - y_pred) / s if s > 0.0 else (y - y_pred).copy()

        result = ModelResult(
            r_squared=r2,
            r_squared_adj=r2_adj,
            rmse=rmse_val,
            mae=mae_val,
            mse=mse_val,
            rss=rss_val,
            tss=tss_val,
            mss=mss_val,
            std_error_estimate=s,
            f_statistic=f_val,
            lof=lof_val,
            ccc=ccc_val,
            q_squared_loo=q2_loo_val,
            slope_origin=stat.k_slope(y, y_pred),
            slope_origin_reverse=stat.k_prime_slope(y, y_pred),
            r_squared_origin=stat.r_squared_0(y, y_pred),
            r_squared_origin_reverse=stat.r_squared_0_prime(y, y_pred),
            roy_r_squared_m_mean=stat.r_bar_m_squared(rm2, rm2p),
            roy_r_squared_m_delta=stat.delta_r_m_squared(rm2, rm2p),
            closeness=stat.closeness(y, y_pred),
            closeness_reverse=stat.closeness_prime(y, y_pred),
            coef_std_errors=coef_std_errors,
            coef_confidence_intervals=coef_ci,
            coef_p_values=coef_p_values,
            leverage=h_ad,
            std_residuals=std_resid,
            leverage_threshold=h_star,
            model_type=model_type,
            n_features=p,
            n_train=n,
            hyperparameters=hyperparameters,
            selected_descriptors=selected_descriptors,
            cluster_assignments=cluster_assignments,
            y_train=y.copy(),
            y_pred_train=y_pred.copy(),
        )

        if y_pred_test is not None and y_test is not None:
            n_test = len(y_test)
            result.q_squared_f1 = stat.q2_f1(y_test, y_pred_test, float(y.mean()))
            result.q_squared_f2 = stat.q2_f2(y_test, y_pred_test)
            result.q_squared_f3 = stat.q2_f3(y_test, y_pred_test, n_train=n)
            result.r_squared_ext = stat.r_squared_ext(y_test, y_pred_test)
            result.press_ext = float(stat.rss(y_test, y_pred_test))
            result.n_test = n_test

        return result


# ---------------------------------------------------------------------------
# Internal OLS utilities
# ---------------------------------------------------------------------------


def _ols_fit_and_stats(
    X_sub: NDArray[np.float64],
    y: NDArray[np.float64],
) -> dict[str, object]:
    """Fit OLS on *X_sub* and return all statistics needed for ModelResult.

    Uses the hat-matrix diagonal to compute LOO PRESS in O(np²) time
    without performing N separate fits.

    Parameters
    ----------
    X_sub : ndarray of shape (n, p)
        Feature submatrix (without intercept column).
    y : ndarray of shape (n,)
        Response vector.

    Returns
    -------
    dict
        Keys: ``coef`` (p+1,), ``y_pred`` (n,), ``residuals`` (n,),
        ``h_loo`` (n,), ``h_ad`` (n,), ``h_star``, ``press_loo``,
        ``q2_loo``, ``s``, ``std_errors`` (p+1,), ``p_values`` (p+1,),
        ``ci`` (p+1, 2), ``std_residuals`` (n,).
    """
    n, p = X_sub.shape
    X_aug = np.column_stack([np.ones(n), X_sub])  # (n, p+1)

    # OLS via pseudo-inverse for numerical stability
    XtX = X_aug.T @ X_aug          # (p+1, p+1)
    XtX_inv = np.linalg.pinv(XtX)  # (p+1, p+1)
    coef: NDArray[np.float64] = XtX_inv @ (X_aug.T @ y)   # (p+1,)
    y_pred: NDArray[np.float64] = X_aug @ coef             # (n,)
    residuals: NDArray[np.float64] = y - y_pred            # (n,)

    # Hat matrix diagonal for LOO (O(np²), avoids materializing n×n H)
    # h_i = row_i(X_aug @ XtX_inv) · row_i(X_aug)
    M: NDArray[np.float64] = X_aug @ XtX_inv               # (n, p+1)
    h_loo: NDArray[np.float64] = (M * X_aug).sum(axis=1)   # (n,)

    # Guard against h_i → 1 (exact leverage point)
    denom: NDArray[np.float64] = np.clip(1.0 - h_loo, 1e-12, None)

    # LOO PRESS and Q²_LOO
    press_loo_val = float(np.sum((residuals / denom) ** 2))
    tss_val = float(np.sum((y - y.mean()) ** 2))
    q2_loo_val = 1.0 - press_loo_val / tss_val if tss_val > 0.0 else 0.0

    # Residual variance (unbiased)
    df_resid = n - p - 1
    rss_val = float(residuals @ residuals)
    s2 = rss_val / df_resid if df_resid > 0 else 0.0
    s = float(np.sqrt(s2))

    # Coefficient statistics for all p+1 coefficients (index 0 = intercept)
    coef_var: NDArray[np.float64] = np.diag(XtX_inv) * s2
    std_errors: NDArray[np.float64] = np.sqrt(np.clip(coef_var, 0.0, None))

    if df_resid > 0:
        t_stats: NDArray[np.float64] = np.where(
            std_errors > 0.0, coef / std_errors, 0.0
        )
        p_vals: NDArray[np.float64] = 2.0 * scipy_stats.t.sf(
            np.abs(t_stats), df=df_resid
        )
        t_crit = float(scipy_stats.t.ppf(0.975, df=df_resid))
        ci: NDArray[np.float64] = np.column_stack(
            [coef - t_crit * std_errors, coef + t_crit * std_errors]
        )
    else:
        p_vals = np.full_like(coef, np.nan)
        ci = np.full((len(coef), 2), np.nan)

    # Leverage for applicability domain — use X_sub WITHOUT intercept per spec
    # h_i = x_i (X^T X)^{-1} x_i^T
    if p > 0:
        XtX_sub = X_sub.T @ X_sub
        XtX_sub_inv = np.linalg.pinv(XtX_sub)
        M_ad: NDArray[np.float64] = X_sub @ XtX_sub_inv   # (n, p)
        h_ad: NDArray[np.float64] = (M_ad * X_sub).sum(axis=1)
    else:
        h_ad = np.zeros(n)

    h_star = 3.0 * p / n  # Williams-plot warning threshold

    # Standardized residuals
    std_resid: NDArray[np.float64] = residuals / s if s > 0.0 else residuals.copy()

    y_pred_loo: NDArray[np.float64] = y - residuals / denom

    return {
        "coef": coef,
        "y_pred": y_pred,
        "y_pred_loo": y_pred_loo,
        "residuals": residuals,
        "h_loo": h_loo,
        "h_ad": h_ad,
        "h_star": h_star,
        "press_loo": press_loo_val,
        "q2_loo": q2_loo_val,
        "s": s,
        "std_errors": std_errors,
        "p_values": p_vals,
        "ci": ci,
        "std_residuals": std_resid,
    }


# ---------------------------------------------------------------------------
# SubsetModel (MLR on a fixed descriptor subset)
# ---------------------------------------------------------------------------


class SubsetModel(BaseModel):
    """OLS/MLR model operating on a fixed descriptor subset.

    This is the model type produced by the GA-MLR and exhaustive enumeration
    procedures.  It stores the trained coefficients and computes the full
    suite of QSARINS regression metrics, including LOO Q² via the hat-matrix
    trick (no N separate fits required).

    Parameters
    ----------
    descriptor_indices : list[int]
        Column indices into the full descriptor matrix X to use as features.
    cluster_assignments : dict[int, int] or None, optional
        Mapping from descriptor index to cluster ID, recorded in the result.
    """

    def __init__(
        self,
        descriptor_indices: list[int],
        cluster_assignments: dict[int, int] | None = None,
    ) -> None:
        self.descriptor_indices = descriptor_indices
        self.cluster_assignments = cluster_assignments
        self._coef: NDArray[np.float64] | None = None       # shape (p+1,): [intercept, *coef]
        self._result: ModelResult | None = None

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(  # type: ignore[override]
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_test: NDArray[np.float64] | None = None,
        y_test: NDArray[np.float64] | None = None,
    ) -> "SubsetModel":
        """Fit OLS on the selected descriptor subset and compute all metrics.

        Parameters
        ----------
        X : ndarray of shape (n_train, n_all_features)
            Full descriptor matrix.  Only ``descriptor_indices`` columns are
            used for fitting.
        y : ndarray of shape (n_train,)
            Training response vector.
        X_test : ndarray of shape (n_test, n_all_features) or None, optional
            External test descriptor matrix.  When provided, external
            validation metrics (Q²_F1/F2/F3, R²_ext, PRESS_ext) are computed.
        y_test : ndarray of shape (n_test,) or None, optional
            External test response vector.

        Returns
        -------
        SubsetModel
            *self*, for method chaining.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        X_sub: NDArray[np.float64] = X_arr[:, self.descriptor_indices]

        ols_stats = _ols_fit_and_stats(X_sub, y_arr)
        self._coef = ols_stats["coef"]  # type: ignore[assignment]

        # Coefficient stats (strip intercept at index 0)
        std_errors_arr: NDArray[np.float64] = ols_stats["std_errors"]  # type: ignore[assignment]
        p_vals_arr: NDArray[np.float64] = ols_stats["p_values"]  # type: ignore[assignment]
        ci_arr: NDArray[np.float64] = ols_stats["ci"]  # type: ignore[assignment]

        # Test set predictions (if provided)
        y_pred_test: NDArray[np.float64] | None = None
        y_test_arr: NDArray[np.float64] | None = None
        X_test_arr_in: NDArray[np.float64] | None = None
        if X_test is not None and y_test is not None:
            X_test_arr_in = np.asarray(X_test, dtype=np.float64)
            y_test_arr = np.asarray(y_test, dtype=np.float64)
            X_test_sub: NDArray[np.float64] = X_test_arr_in[:, self.descriptor_indices]
            n_test = len(y_test_arr)
            coef: NDArray[np.float64] = self._coef  # type: ignore[assignment]
            X_test_aug = np.column_stack([np.ones(n_test), X_test_sub])
            y_pred_test = X_test_aug @ coef

        self._result = BaseModel._build_result(
            X_sub,
            y_arr,
            y_pred=ols_stats["y_pred"],  # type: ignore[arg-type]
            y_pred_loo=ols_stats["y_pred_loo"],  # type: ignore[arg-type]
            model_type="mlr",
            hyperparameters={},
            X_test=X_test_arr_in,
            y_test=y_test_arr,
            y_pred_test=y_pred_test,
            coef_std_errors=std_errors_arr[1:].copy(),
            coef_ci=ci_arr[1:].copy(),
            coef_p_values=p_vals_arr[1:].copy(),
            selected_descriptors=self.descriptor_indices,
            cluster_assignments=self.cluster_assignments,
        )
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict response for new samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_all_features)
            Descriptor matrix; only ``descriptor_indices`` columns are used.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted response values.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._coef is None:
            raise RuntimeError("SubsetModel must be fit before calling predict")
        X_arr = np.asarray(X, dtype=np.float64)
        X_sub: NDArray[np.float64] = X_arr[:, self.descriptor_indices]
        n = X_sub.shape[0]
        X_aug = np.column_stack([np.ones(n), X_sub])
        return X_aug @ self._coef

    def get_results(self) -> ModelResult:
        """Return the :class:`~qsarify.results.model_result.ModelResult`.

        Returns
        -------
        ModelResult
            All metrics computed during the last :meth:`fit` call.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._result is None:
            raise RuntimeError("SubsetModel must be fit before calling get_results")
        return self._result

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def coef_(self) -> NDArray[np.float64]:
        """Non-intercept regression coefficients, shape (n_features,)."""
        if self._coef is None:
            raise RuntimeError("SubsetModel must be fit first")
        return self._coef[1:]

    @property
    def intercept_(self) -> float:
        """Regression intercept."""
        if self._coef is None:
            raise RuntimeError("SubsetModel must be fit first")
        return float(self._coef[0])
