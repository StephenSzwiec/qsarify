"""Statistical validation procedures for QSAR models.

Provides two validation methods:

- :func:`run_lmo` — Leave-Many-Out (LMO) cross-validation with configurable
  holdout fractions and iteration counts.
- :func:`run_y_scrambling` — Y-scrambling (response permutation) validation
  to test for chance correlations.

Both procedures are parallelised via :class:`concurrent.futures.ProcessPoolExecutor`
when ``n_workers > 1``.  Worker functions are module-level so they are
picklable by the multiprocessing backend.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from typing import Any

from qsarify.utils import statistics as stat

__all__ = [
    "LMOResult",
    "YScramblingResult",
    "run_lmo",
    "run_y_scrambling",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LMOResult:
    """Results from Leave-Many-Out cross-validation.

    Parameters
    ----------
    holdout_fraction : float
        Fraction of samples held out per iteration.
    n_iterations : int
        Number of LMO iterations performed.
    q2_per_iteration : ndarray of shape (n_iterations,)
        External Q² (training-mean reference) for each iteration.
    r2_per_iteration : ndarray of shape (n_iterations,)
        R² on the holdout set for each iteration.
    rmse_per_iteration : ndarray of shape (n_iterations,)
        RMSE on the holdout set for each iteration.
    mean_q2, std_q2 : float
        Mean and standard deviation of Q² across iterations.
    mean_r2, std_r2 : float
        Mean and standard deviation of R² across iterations.
    mean_rmse, std_rmse : float
        Mean and standard deviation of RMSE across iterations.
    """

    holdout_fraction: float
    n_iterations: int
    q2_per_iteration: NDArray[np.float64]
    r2_per_iteration: NDArray[np.float64]
    rmse_per_iteration: NDArray[np.float64]
    mean_q2: float
    std_q2: float
    mean_r2: float
    std_r2: float
    mean_rmse: float
    std_rmse: float


@dataclass
class YScramblingResult:
    """Results from Y-scrambling (response permutation) validation.

    Parameters
    ----------
    n_iterations : int
        Number of Y-scrambling iterations performed.
    r2_scrambled : ndarray of shape (n_iterations,)
        Training R² values from models fitted on permuted Y vectors.
    q2_scrambled : ndarray of shape (n_iterations,)
        LOO Q² values from models fitted on permuted Y vectors.
    r2_original : float
        R² of the original (unscrambled) model, as provided by the caller.
    q2_original : float
        Q²_LOO of the original model, as provided by the caller.
    mean_r2_scrambled, std_r2_scrambled : float
        Mean and standard deviation of scrambled R² values.
    mean_q2_scrambled, std_q2_scrambled : float
        Mean and standard deviation of scrambled Q²_LOO values.

    Notes
    -----
    A valid model should show: ``mean_r2_scrambled << r2_original`` and
    ``mean_q2_scrambled << q2_original``.  Per QSARINS conventions, 100
    iterations is the recommended minimum.
    """

    n_iterations: int
    r2_scrambled: NDArray[np.float64]
    q2_scrambled: NDArray[np.float64]
    r2_original: float
    q2_original: float
    mean_r2_scrambled: float
    std_r2_scrambled: float
    mean_q2_scrambled: float
    std_q2_scrambled: float


# ---------------------------------------------------------------------------
# Module-level worker functions (must be picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _lmo_worker(
    args: tuple[Any, NDArray[np.float64], NDArray[np.float64], NDArray[np.intp], float],
) -> tuple[float, float, float]:
    """Fit a clone of the estimator on the train fold, evaluate on holdout.

    Returns
    -------
    tuple[float, float, float]
        (Q²_F1, R², RMSE) on the holdout set.
    """
    estimator, X, y, holdout_idx, y_train_mean = args

    train_mask = np.ones(len(y), dtype=bool)
    train_mask[holdout_idx] = False

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_ho, y_ho = X[holdout_idx], y[holdout_idx]

    est = clone(estimator)
    est.fit(X_tr, y_tr)
    y_pred_ho = np.asarray(est.predict(X_ho), dtype=np.float64)

    q2 = float(stat.q2_f1(y_ho, y_pred_ho, float(y_tr.mean())))
    r2 = float(stat.r_squared(y_ho, y_pred_ho))
    rmse_val = float(stat.rmse(y_ho, y_pred_ho))
    return q2, r2, rmse_val


def _y_scramble_worker(
    args: tuple[Any, NDArray[np.float64], NDArray[np.float64], int],
) -> tuple[float, float]:
    """Fit a clone of the estimator on a permuted Y, compute R² and Q²_LOO.

    Returns
    -------
    tuple[float, float]
        (training R², LOO Q²) on the scrambled dataset.
    """
    estimator, X, y, seed = args

    rng = np.random.default_rng(seed)
    y_scrambled: NDArray[np.float64] = rng.permutation(y).astype(np.float64)

    # Training R²
    est_fit = clone(estimator)
    est_fit.fit(X, y_scrambled)
    y_pred_train = np.asarray(est_fit.predict(X), dtype=np.float64)
    r2 = float(stat.r_squared(y_scrambled, y_pred_train))

    # LOO Q² via cross_val_predict (n_jobs=1 to avoid nested parallelism)
    est_cv = clone(estimator)
    y_pred_loo = np.asarray(
        cross_val_predict(est_cv, X, y_scrambled, cv=LeaveOneOut(), n_jobs=1),
        dtype=np.float64,
    )
    q2 = float(stat.q2_loo(y_scrambled, y_pred_loo))

    return r2, q2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_lmo(
    estimator: Any,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    holdout_fraction: float = 0.30,
    n_iterations: int = 100,
    random_seed: int | None = None,
    n_workers: int = 1,
) -> LMOResult:
    """Run Leave-Many-Out cross-validation on a fitted-style estimator.

    The estimator is cloned (via :func:`sklearn.base.clone`) for each
    iteration so the original object is never mutated.  Parallelisation uses
    :class:`concurrent.futures.ProcessPoolExecutor` when ``n_workers > 1``.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Any object with ``.fit(X, y)`` and ``.predict(X)`` methods.
        Must be picklable (all sklearn estimators satisfy this).
    X : ndarray of shape (n_samples, n_features)
        Feature matrix; should already contain only the descriptors used
        by the model (e.g. the selected subset for MLR).
    y : ndarray of shape (n_samples,)
        Response vector.
    holdout_fraction : float, optional
        Fraction of samples held out per iteration.  Typical values:
        0.20, 0.30, 0.33.  Default 0.30.
    n_iterations : int, optional
        Number of LMO repetitions.  Default 100.
    random_seed : int or None, optional
        Seed for reproducible fold assignments.  When ``None`` the system
        uses entropy-based seeding and each call may differ.
    n_workers : int, optional
        Number of worker processes.  ``1`` (default) runs sequentially.
        ``-1`` uses all available CPU cores.

    Returns
    -------
    LMOResult
        Aggregate statistics and per-iteration arrays.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n = len(y_arr)
    n_holdout = max(1, int(np.round(n * holdout_fraction)))

    # Generate per-iteration seeds deterministically from random_seed
    seed_rng = np.random.default_rng(random_seed)
    iter_seeds: list[int] = seed_rng.integers(0, 2**31, size=n_iterations).tolist()

    # Build holdout index arrays for each iteration
    def _holdout_indices(seed: int) -> NDArray[np.intp]:
        rng = np.random.default_rng(seed)
        return rng.choice(n, size=n_holdout, replace=False).astype(np.intp)

    y_train_mean = float(y_arr.mean())
    args_list = [
        (estimator, X_arr, y_arr, _holdout_indices(s), y_train_mean)
        for s in iter_seeds
    ]

    if n_workers == 1:
        results_raw = [_lmo_worker(a) for a in args_list]
    else:
        actual_workers = None if n_workers == -1 else n_workers
        with ProcessPoolExecutor(max_workers=actual_workers) as pool:
            results_raw = list(pool.map(_lmo_worker, args_list))

    q2_arr = np.array([r[0] for r in results_raw], dtype=np.float64)
    r2_arr = np.array([r[1] for r in results_raw], dtype=np.float64)
    rmse_arr = np.array([r[2] for r in results_raw], dtype=np.float64)

    return LMOResult(
        holdout_fraction=holdout_fraction,
        n_iterations=n_iterations,
        q2_per_iteration=q2_arr,
        r2_per_iteration=r2_arr,
        rmse_per_iteration=rmse_arr,
        mean_q2=float(np.mean(q2_arr)),
        std_q2=float(np.std(q2_arr)),
        mean_r2=float(np.mean(r2_arr)),
        std_r2=float(np.std(r2_arr)),
        mean_rmse=float(np.mean(rmse_arr)),
        std_rmse=float(np.std(rmse_arr)),
    )


def run_y_scrambling(
    estimator: Any,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    r2_original: float,
    q2_original: float,
    n_iterations: int = 100,
    random_seed: int | None = None,
    n_workers: int = 1,
) -> YScramblingResult:
    """Run Y-scrambling (response permutation) validation.

    For each iteration, the response vector *y* is randomly permuted and a
    fresh clone of the estimator is fitted.  Both the training R² and the
    LOO Q² (via :func:`sklearn.model_selection.cross_val_predict` with
    :class:`~sklearn.model_selection.LeaveOneOut`) are recorded.

    A valid, non-chance model should show:
    - ``mean_r2_scrambled << r2_original``
    - ``mean_q2_scrambled << q2_original``

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Any object with ``.fit(X, y)`` and ``.predict(X)`` methods.
    X : ndarray of shape (n_samples, n_features)
        Feature matrix for the selected model.
    y : ndarray of shape (n_samples,)
        Original (unscrambled) response vector.
    r2_original : float
        R² of the original model, provided by the caller from
        :attr:`~qsarify.results.model_result.ModelResult.r_squared`.
    q2_original : float
        Q²_LOO of the original model, provided by the caller from
        :attr:`~qsarify.results.model_result.ModelResult.q_squared_loo`.
    n_iterations : int, optional
        Number of scrambling iterations.  Minimum recommended: 100.
        Default 100.
    random_seed : int or None, optional
        Seed for reproducible permutations.
    n_workers : int, optional
        Number of worker processes.  Default 1 (sequential).

    Returns
    -------
    YScramblingResult
        Scrambled statistics and comparison to the original model.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    # Derive per-iteration seeds deterministically
    seed_rng = np.random.default_rng(random_seed)
    iter_seeds: list[int] = seed_rng.integers(0, 2**31, size=n_iterations).tolist()

    args_list = [(estimator, X_arr, y_arr, s) for s in iter_seeds]

    if n_workers == 1:
        results_raw = [_y_scramble_worker(a) for a in args_list]
    else:
        actual_workers = None if n_workers == -1 else n_workers
        with ProcessPoolExecutor(max_workers=actual_workers) as pool:
            results_raw = list(pool.map(_y_scramble_worker, args_list))

    r2_scram = np.array([r[0] for r in results_raw], dtype=np.float64)
    q2_scram = np.array([r[1] for r in results_raw], dtype=np.float64)

    return YScramblingResult(
        n_iterations=n_iterations,
        r2_scrambled=r2_scram,
        q2_scrambled=q2_scram,
        r2_original=float(r2_original),
        q2_original=float(q2_original),
        mean_r2_scrambled=float(np.mean(r2_scram)),
        std_r2_scrambled=float(np.std(r2_scram)),
        mean_q2_scrambled=float(np.mean(q2_scram)),
        std_q2_scrambled=float(np.std(q2_scram)),
    )
