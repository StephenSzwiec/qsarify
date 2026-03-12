"""SubsetModel: OLS/MLR on a fixed descriptor subset.

This is the model type produced by the GA-MLR and exhaustive enumeration
procedures.  It stores the trained coefficients and computes the full
suite of QSARINS regression metrics, including LOO Q² via the hat-matrix
trick (no N separate fits required).

The OLS internals (:class:`OLSFitResult`, :func:`ols_fit_and_stats`) are
co-located here because they are specific to the subset/MLR workflow and
not shared with the sklearn-based :class:`~qsarify.modeling.models.FullModel`
wrappers.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from qsarify.modeling.base import BaseModel
from qsarify.modeling.metrics import compute_model_result
from qsarify.results.model_result import ModelResult

__all__ = ["OLSFitResult", "SubsetModel", "ols_fit_and_stats"]


# ---------------------------------------------------------------------------
# OLS fit result (typed replacement for the old dict)
# ---------------------------------------------------------------------------


class OLSFitResult(NamedTuple):
    """Result of :func:`ols_fit_and_stats`.

    Attributes
    ----------
    coef : ndarray of shape (p+1,)
        OLS coefficients.  Index 0 is the intercept.
    y_pred : ndarray of shape (n,)
        Full-fit training predictions.
    y_pred_loo : ndarray of shape (n,)
        Leave-one-out cross-validation predictions (hat-matrix trick).
    leverage : ndarray of shape (n,)
        Applicability-domain leverage h_i = x_i (X^T X)^{-1} x_i^T,
        computed on X_sub *without* the intercept column.
    std_residuals : ndarray of shape (n,)
        Standardized residuals (residual / s).
    std_errors : ndarray of shape (p+1,)
        Standard errors for all coefficients (index 0 = intercept).
    p_values : ndarray of shape (p+1,)
        Two-sided p-values for each coefficient.
    ci : ndarray of shape (p+1, 2)
        95% confidence intervals [lower, upper] for each coefficient.
    s : float
        Standard error of the regression estimate.
    """

    coef: NDArray[np.float64]
    y_pred: NDArray[np.float64]
    y_pred_loo: NDArray[np.float64]
    leverage: NDArray[np.float64]
    std_residuals: NDArray[np.float64]
    std_errors: NDArray[np.float64]
    p_values: NDArray[np.float64]
    ci: NDArray[np.float64]
    s: float


# ---------------------------------------------------------------------------
# OLS fitting
# ---------------------------------------------------------------------------


def ols_fit_and_stats(
    X_sub: NDArray[np.float64],
    y: NDArray[np.float64],
) -> OLSFitResult:
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
    OLSFitResult
        Typed named tuple with all OLS outputs.
    """
    n, p = X_sub.shape

    # Augment with intercept
    X_aug: NDArray[np.float64] = np.column_stack([np.ones(n), X_sub])

    # OLS via pseudo-inverse for numerical stability
    XtX: NDArray[np.float64] = X_aug.T @ X_aug
    XtX_inv: NDArray[np.float64] = np.linalg.pinv(XtX).astype(np.float64)
    coef: NDArray[np.float64] = XtX_inv @ (X_aug.T @ y)
    y_pred: NDArray[np.float64] = X_aug @ coef
    residuals: NDArray[np.float64] = y - y_pred

    # Hat-matrix diagonal for LOO (O(np²), avoids materializing n×n H)
    M: NDArray[np.float64] = X_aug @ XtX_inv
    h_loo: NDArray[np.float64] = (M * X_aug).sum(axis=1)

    # Guard against h_i → 1 (exact leverage point)
    denom: NDArray[np.float64] = np.clip(1.0 - h_loo, 1e-12, None)

    # LOO predictions via hat-matrix trick
    y_pred_loo: NDArray[np.float64] = y - residuals / denom

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

    # Applicability-domain leverage — X_sub WITHOUT intercept, per spec:
    #   h_i = x_i (X^T X)^{-1} x_i^T
    if p > 0:
        XtX_sub: NDArray[np.float64] = X_sub.T @ X_sub
        XtX_sub_inv: NDArray[np.float64] = np.linalg.pinv(XtX_sub).astype(np.float64)
        M_ad: NDArray[np.float64] = X_sub @ XtX_sub_inv
        leverage: NDArray[np.float64] = (M_ad * X_sub).sum(axis=1)
    else:
        leverage = np.zeros(n, dtype=np.float64)

    # Standardized residuals
    std_resid: NDArray[np.float64] = residuals / s if s > 0.0 else residuals.copy()

    return OLSFitResult(
        coef=coef,
        y_pred=y_pred,
        y_pred_loo=y_pred_loo,
        leverage=leverage,
        std_residuals=std_resid,
        std_errors=std_errors,
        p_values=p_vals,
        ci=ci,
        s=s,
    )


# ---------------------------------------------------------------------------
# SubsetModel
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
        self._coef: NDArray[np.float64] | None = None
        self._result: ModelResult | None = None

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(
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
            External test descriptor matrix.
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

        ols = ols_fit_and_stats(X_sub, y_arr)
        self._coef = ols.coef

        # Test-set predictions
        y_pred_test: NDArray[np.float64] | None = None
        y_test_arr: NDArray[np.float64] | None = None
        if X_test is not None and y_test is not None:
            X_test_arr = np.asarray(X_test, dtype=np.float64)
            y_test_arr = np.asarray(y_test, dtype=np.float64)
            X_test_sub: NDArray[np.float64] = X_test_arr[:, self.descriptor_indices]
            X_test_aug = np.column_stack([np.ones(len(y_test_arr)), X_test_sub])
            y_pred_test = X_test_aug @ self._coef

        # Coefficient stats: strip intercept (index 0) for the result
        self._result = compute_model_result(
            X=X_sub,
            y=y_arr,
            y_pred=ols.y_pred,
            y_pred_loo=ols.y_pred_loo,
            model_type="mlr",
            hyperparameters={},
            y_test=y_test_arr,
            y_pred_test=y_pred_test,
            leverage=ols.leverage,
            coef_std_errors=ols.std_errors[1:].copy(),
            coef_confidence_intervals=ols.ci[1:].copy(),
            coef_p_values=ols.p_values[1:].copy(),
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
