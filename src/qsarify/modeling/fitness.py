"""Shared fitness scoring for model evaluation.

Provides a single ``fitness_score`` function used by both the GA-MLR
feature-selection loop and the differential-evolution hyperparameter
tuning in :mod:`qsarify.modeling.models`.  All returned scores follow a
"higher is better" convention so that callers can uniformly maximise.

This module delegates all numerical work to
:mod:`qsarify.utils.statistics`, which provides the vectorized metric
implementations.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from qsarify.utils import statistics as stat

__all__ = ["VALID_FITNESS_FUNCTIONS", "fitness_score"]

VALID_FITNESS_FUNCTIONS: frozenset[str] = frozenset(
    {
        "q2_loo",
        "r2_adj",
        "rmse_cv",
        "lof",
    }
)


def fitness_score(
    loo: stat.LOOResult,
    y: NDArray[np.float64],
    n_features: int,
    fitness_function: str,
) -> float:
    """Return a *higher-is-better* fitness score.

    Parameters
    ----------
    loo : LOOResult
        Result of a leave-one-out cross-validation computation (either
        hat-matrix-based for OLS or sklearn-based for other estimators).
    y : ndarray of shape (n,)
        Training response vector.
    n_features : int
        Number of descriptor columns *p* used by the model.
    fitness_function : str
        One of ``'q2_loo'``, ``'r2_adj'``, ``'rmse_cv'``, ``'lof'``.

    Returns
    -------
    float
        Fitness score.  Higher values are always better; ``rmse_cv`` and
        ``lof`` are negated.

    Raises
    ------
    ValueError
        If *fitness_function* is not one of the valid options.
    """
    if fitness_function == "q2_loo":
        return float(loo.q2_loo)

    if fitness_function == "r2_adj":
        return float(stat.r_squared_adj(y, loo.y_pred, n_features))

    n = len(y)

    if fitness_function == "rmse_cv":
        return -float(np.sqrt(loo.press / n))

    if fitness_function == "lof":
        return -float(stat.lof(y, loo.y_pred, n_features))

    raise ValueError(
        f"Unknown fitness_function {fitness_function!r}. "
        f"Valid options: {sorted(VALID_FITNESS_FUNCTIONS)}"
    )
