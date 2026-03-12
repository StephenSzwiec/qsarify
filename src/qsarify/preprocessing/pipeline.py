"""Convenience preprocessing pipeline for QSAR/QSPR descriptor matrices.

Provides :func:`preprocessing`, a single-call wrapper that applies the
standard QSARify preprocessing sequence and returns a train/test split
ready for model building.
"""

from __future__ import annotations

import pandas as pd

from qsarify.preprocessing.filters import (
    remove_high_correlation,
    remove_nans,
    remove_near_zero_variance,
)
from qsarify.preprocessing.scalers import StandardScaler
from qsarify.preprocessing.splitters import SplitResult, random_split, stratified_split

__all__ = ["preprocessing"]


def preprocessing(
    X: pd.DataFrame,
    y: pd.Series,
    split: str = "sorted",
    test_size: float = 0.2,
    constant_threshold: float = 0.0,
    correlation_threshold: float = 0.95,
    normalize: bool = True,
    random_seed: int | None = None,
) -> SplitResult:
    """One-call preprocessing pipeline for QSAR/QSPR descriptor matrices.

    Applies the standard QSARify preprocessing sequence to *X* and returns
    a train/test split ready for model building.  The input *X* is never
    mutated.

    Steps applied in order:

    1. **Deep copy** — *X* is copied; the caller's DataFrame is never modified.
    2. **NaN filter** — drops columns containing any ``NaN`` value (see
       :func:`~qsarify.preprocessing.filters.remove_nans`).
    3. **Constant filter** — drops columns whose coefficient of variation is
       ≤ *constant_threshold* (see
       :func:`~qsarify.preprocessing.filters.remove_near_zero_variance`).
       With the default of 0.0, only exactly constant columns (CV == 0) are
       removed.
    4. **Correlation filter** — drops one member of each highly correlated
       descriptor pair (see
       :func:`~qsarify.preprocessing.filters.remove_high_correlation`).
    5. **Normalisation** — optionally applies zero-mean unit-variance scaling
       (:class:`~qsarify.preprocessing.scalers.StandardScaler` fit on the
       full filtered matrix before splitting).
    6. **Train/test split** — stratified by response (``split="sorted"``) or
       random (``split="random"``).

    Parameters
    ----------
    X : pd.DataFrame
        Raw descriptor matrix, shape (n_samples, n_descriptors).
    y : pd.Series
        Response vector, length n_samples.
    split : {"sorted", "random"}, optional
        Split strategy.  ``"sorted"`` uses systematic stratified sampling
        (QSARINS convention, default); ``"random"`` uses random sampling.
    test_size : float, optional
        Fraction of samples to include in the test set.  Default 0.2.
    constant_threshold : float, optional
        Coefficient-of-variation threshold for the constant column filter.
        Default 0.0 removes only exactly constant columns (CV == 0).
        Set to e.g. 0.01 to also remove near-constant columns.
    correlation_threshold : float, optional
        Absolute Pearson correlation threshold for the collinearity filter.
        Default 0.95.
    normalize : bool, optional
        If ``True`` (default), applies
        :class:`~qsarify.preprocessing.scalers.StandardScaler` to *X*
        before splitting.
    random_seed : int or None, optional
        Random seed for :func:`~qsarify.preprocessing.splitters.random_split`.
        Ignored when *split* is ``"sorted"``.

    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test)
        Filtered (and optionally normalised) descriptor sub-matrices and
        response sub-vectors as :class:`pandas.DataFrame` /
        :class:`pandas.Series`.

    Raises
    ------
    ValueError
        If *split* is not one of ``{"sorted", "random"}``.

    Examples
    --------
    Default usage (sorted split, StandardScaler, all filters):

    >>> X_tr, X_te, y_tr, y_te = preprocessing(X, y)

    Random split, no normalisation, stricter constant filter:

    >>> X_tr, X_te, y_tr, y_te = preprocessing(
    ...     X, y, split="random", random_seed=42,
    ...     constant_threshold=0.01, normalize=False,
    ... )
    """
    if split not in ("sorted", "random"):
        raise ValueError(f"split must be 'sorted' or 'random', got {split!r}")

    # Step 1: deep copy — never mutate the caller's data
    X_work = X.copy()

    # Step 2: remove columns with any NaN values
    X_work = remove_nans(X_work)

    # Step 3: remove constant / near-constant columns
    X_work = remove_near_zero_variance(X_work, threshold=constant_threshold)

    # Step 4: remove highly correlated columns
    X_work = remove_high_correlation(X_work, threshold=correlation_threshold)

    # Step 5: normalize (fit on full filtered matrix before splitting)
    if normalize:
        scaler = StandardScaler()
        X_work = scaler.fit(X_work).transform(X_work)

    # Step 6: train/test split
    if split == "sorted":
        return stratified_split(X_work, y, test_size=test_size)
    else:
        return random_split(X_work, y, test_size=test_size, random_seed=random_seed)
