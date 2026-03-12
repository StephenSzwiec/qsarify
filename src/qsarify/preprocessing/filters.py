"""Pre-processing filter functions for descriptor matrices.

Provides two core filters used in QSAR/QSPR workflows:

1. :func:`remove_near_zero_variance` — drops descriptor columns whose
   coefficient of variation falls below a configurable threshold.
2. :func:`remove_high_correlation` — drops descriptor columns that are
   pairwise Pearson-correlated above a configurable threshold, using a
   greedy forward-selection approach to maximise retained information.

Both functions operate on :class:`pandas.DataFrame` objects and return a
filtered copy with the same row index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["remove_nans", "remove_near_zero_variance", "remove_high_correlation"]


def remove_nans(X: pd.DataFrame) -> pd.DataFrame:
    """Remove features with any NaN values.

    Parameters
    ----------
    X : pd.DataFrame
        Descriptor matrix, shape (n_samples, n_descriptors).

    Returns
    -------
    pd.DataFrame
        Filtered copy of *X* with columns containing NaN values removed.
    """
    return X.dropna(axis=1, how="any").copy()


def remove_near_zero_variance(
    X: pd.DataFrame,
    threshold: float = 0.01,
) -> pd.DataFrame:
    """Remove descriptor columns with near-zero or zero variance.

    A column is considered near-constant if its *coefficient of variation*
    (``std / |mean|``) is below *threshold*.  Constant columns (std == 0) are
    always removed regardless of *threshold*.

    Parameters
    ----------
    X : pd.DataFrame
        Descriptor matrix, shape (n_samples, n_descriptors).
    threshold : float, optional
        Coefficient-of-variation threshold.  Columns with CV < threshold are
        dropped.  Default is 0.01.

    Returns
    -------
    pd.DataFrame
        Filtered copy of *X* with near-constant columns removed.
    """
    stds = X.std(ddof=1)
    means = X.mean().abs()

    # Compute coefficient of variation; treat zero-mean columns as having CV = std
    cv = stds.copy()
    nonzero_mean = means > 0
    cv[nonzero_mean] = stds[nonzero_mean] / means[nonzero_mean]

    keep = cv[cv > threshold].index
    return X[keep].copy()


def remove_high_correlation(
    X: pd.DataFrame,
    threshold: float = 0.95,
) -> pd.DataFrame:
    """Remove descriptor columns that exceed a Pearson correlation threshold.

    Uses a greedy algorithm: iterate over column pairs in the upper triangle
    of the absolute Pearson correlation matrix.  When a pair exceeds
    *threshold*, the second (right-hand) column is marked for removal.  This
    preserves the first encountered column of each highly-correlated group.

    For large descriptor matrices the correlation matrix is computed via
    vectorised NumPy/BLAS operations rather than Python loops to leverage
    BLAS-level threading.

    Parameters
    ----------
    X : pd.DataFrame
        Descriptor matrix, shape (n_samples, n_descriptors).
    threshold : float, optional
        Absolute Pearson correlation threshold.  Columns whose correlation
        with a preceding column exceeds this value are removed.
        Default is 0.95.

    Returns
    -------
    pd.DataFrame
        Filtered copy of *X* with redundant columns removed.
    """
    if X.shape[1] <= 1:
        return X.copy()

    cols = list(X.columns)
    arr = X.to_numpy(dtype=np.float64)

    # Compute full absolute Pearson correlation matrix via NumPy
    corr = np.abs(np.corrcoef(arr, rowvar=False))

    n = len(cols)
    to_drop: set[int] = set()

    for i in range(n):
        if i in to_drop:
            continue
        for j in range(i + 1, n):
            if j in to_drop:
                continue
            if corr[i, j] >= threshold:
                to_drop.add(j)

    keep = [c for idx, c in enumerate(cols) if idx not in to_drop]
    return X[keep].copy()
