"""ModelResult dataclass for QSAR/QSPR model evaluation results.

Every :class:`qsarify.modeling.base.BaseModel` subclass returns a
:class:`ModelResult` from its ``get_results()`` method.  Fields that are
not applicable to a given model type are explicitly ``None``.

See :mod:`qsarify.utils.statistics` for the formulas used to populate each
field, and ``agent_docs/model_interface.md`` for the full metric applicability
matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["ModelResult"]


@dataclass
class ModelResult:
    """Container for all metrics produced by a fitted QSAR model.

    Parameters
    ----------
    See field descriptions below.  All ``float | None`` fields default to
    ``None``; they are populated by the model's ``fit()`` / ``get_results()``
    implementation.

    Notes
    -----
    External validation metrics (``q_squared_f1``, ``q_squared_f2``,
    ``q_squared_f3``, ``r_squared_ext``, ``press_ext``) are ``None``
    unless an external test set was provided during fitting.

    Coefficient statistics (``coef_std_errors``, ``coef_confidence_intervals``,
    ``coef_p_values``) are ``None`` for non-linear models (SVR, RF, GBR).
    """

    # ------------------------------------------------------------------
    # Universal regression metrics — training set
    # ------------------------------------------------------------------

    r_squared: float | None = None
    """Coefficient of determination R² (training set)."""

    r_squared_adj: float | None = None
    """Adjusted R² (training set)."""

    rmse: float | None = None
    """Root Mean Squared Error (training set)."""

    mae: float | None = None
    """Mean Absolute Error (training set)."""

    mse: float | None = None
    """Mean Squared Error (training set)."""

    rss: float | None = None
    """Residual Sum of Squares (training set)."""

    tss: float | None = None
    """Total Sum of Squares (training set)."""

    mss: float | None = None
    """Model Sum of Squares (training set)."""

    std_error_estimate: float | None = None
    """Standard error of the regression estimate *s* (training set)."""

    f_statistic: float | None = None
    """Overall F-statistic for regression significance (training set)."""

    lof: float | None = None
    """Friedman's Lack-of-Fit statistic (training set)."""

    ccc: float | None = None
    """Concordance Correlation Coefficient (training set)."""

    q_squared_loo: float | None = None
    """Leave-One-Out cross-validated Q² (training set)."""

    # ------------------------------------------------------------------
    # Origin-forced regression metrics — training set
    # ------------------------------------------------------------------

    slope_origin: float | None = None
    """Slope *k* through origin with predicted values on x-axis."""

    slope_origin_reverse: float | None = None
    """Reverse slope *k'* through origin with observed values on x-axis."""

    r_squared_origin: float | None = None
    """R²₀ — R² forced through origin, observed as reference."""

    r_squared_origin_reverse: float | None = None
    """R'²₀ — R² forced through origin, predicted as reference."""

    # ------------------------------------------------------------------
    # Roy's criteria and Golbraikh-Tropsha closeness — training set
    # ------------------------------------------------------------------

    roy_r_squared_m_mean: float | None = None
    """Roy's r̄²_m = (r²_m + r'²_m) / 2."""

    roy_r_squared_m_delta: float | None = None
    """Roy's Δr²_m = |r²_m − r'²_m|."""

    closeness: float | None = None
    """Golbraikh-Tropsha closeness *clos* = |R² − R²₀| / R²."""

    closeness_reverse: float | None = None
    """Golbraikh-Tropsha closeness *clos'* = |R² − R'²₀| / R²."""

    # ------------------------------------------------------------------
    # External validation metrics (None without test set)
    # ------------------------------------------------------------------

    q_squared_f1: float | None = None
    """Q²_F1 (external test set, training mean reference)."""

    q_squared_f2: float | None = None
    """Q²_F2 (external test set, external mean reference)."""

    q_squared_f3: float | None = None
    """Q²_F3 (external test set, sample-size-scaled)."""

    r_squared_ext: float | None = None
    """R² on the external test set."""

    press_ext: float | None = None
    """PRESS on the external test set."""

    # ------------------------------------------------------------------
    # Linear-model-only coefficient statistics
    # ------------------------------------------------------------------

    coef_std_errors: NDArray[np.float64] | None = None
    """Standard errors of regression coefficients (shape: (n_features,), excludes intercept)."""

    coef_confidence_intervals: NDArray[np.float64] | None = None
    """±95% confidence intervals, shape (n_features, 2).  Column 0 = lower, 1 = upper."""

    coef_p_values: NDArray[np.float64] | None = None
    """Two-sided p-values for coefficient t-tests, shape (n_features,)."""

    # ------------------------------------------------------------------
    # Applicability domain (all model types)
    # ------------------------------------------------------------------

    leverage: NDArray[np.float64] | None = None
    """Leverage h_i = x_i (X^TX)^{-1} x_i^T for training samples (shape: (n_train,))."""

    std_residuals: NDArray[np.float64] | None = None
    """Standardized residuals e_i / s (shape: (n_train,))."""

    leverage_threshold: float | None = None
    """Williams-plot warning threshold h* = 3p / n."""

    # ------------------------------------------------------------------
    # Model metadata
    # ------------------------------------------------------------------

    selected_descriptors: list[int] | None = None
    """Descriptor column indices used by this model (SubsetModel only)."""

    cluster_assignments: dict[int, int] | None = None
    """Mapping descriptor_index → cluster_id (SubsetModel only)."""

    model_type: str = ""
    """One of: 'mlr', 'ridge', 'lasso', 'svr', 'rf', 'gbr'."""

    n_features: int = 0
    """Number of descriptors used."""

    n_train: int = 0
    """Training set size."""

    n_test: int | None = None
    """External test set size (``None`` if not provided)."""

    hyperparameters: dict[str, object] = field(default_factory=dict)
    """Model hyperparameters (empty dict for MLR)."""

    # ------------------------------------------------------------------
    # Validation results (populated after validation procedures)
    # ------------------------------------------------------------------

    lmo_results: dict[str, object] | None = None
    """Leave-Many-Out cross-validation results, keyed by holdout fraction."""

    y_scrambling_results: dict[str, object] | None = None
    """Y-scrambling validation summary statistics."""

    # ------------------------------------------------------------------
    # Raw prediction arrays (needed by diagnostic plots)
    # ------------------------------------------------------------------

    y_train: NDArray[np.float64] | None = None
    """Observed training responses, shape (n_train,).  Stored for plotting."""

    y_pred_train: NDArray[np.float64] | None = None
    """Model predictions on training set, shape (n_train,).  Stored for plotting."""
