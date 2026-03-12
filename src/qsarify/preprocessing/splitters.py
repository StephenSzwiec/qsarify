"""Train/test splitting strategies for QSAR/QSPR datasets.

Provides two splitting strategies:

1. :func:`random_split` — random partition with a configurable seed.
2. :func:`stratified_split` — ordered (systematic) split that ensures the
   test set contains samples spread across the full response range, matching
   the QSARINS stratified-split convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as _sk_tts

__all__ = ["random_split", "stratified_split"]

SplitResult = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


def random_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_seed: int | None = None,
) -> SplitResult:
    """Randomly partition *X* and *y* into training and test sets.

    Wraps :func:`sklearn.model_selection.train_test_split` for consistency
    with the broader scikit-learn ecosystem.

    Parameters
    ----------
    X : pd.DataFrame
        Descriptor matrix, shape (n_samples, n_descriptors).
    y : pd.Series
        Response vector, length n_samples.
    test_size : float, optional
        Fraction of samples to include in the test set.  Default 0.2.
    random_seed : int or None, optional
        Random seed for reproducibility.  ``None`` means non-deterministic.

    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test)
        Four :class:`pd.DataFrame` / :class:`pd.Series` objects.
    """
    X_tr, X_te, y_tr, y_te = _sk_tts(
        X, y, test_size=test_size, random_state=random_seed
    )
    return (
        pd.DataFrame(X_tr, columns=X.columns),
        pd.DataFrame(X_te, columns=X.columns),
        pd.Series(y_tr, name=y.name),
        pd.Series(y_te, name=y.name),
    )


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> SplitResult:
    """Systematic (stratified-by-response) train/test split.

    Sorts compounds by the response variable and selects every *k*-th sample
    for the test set, where *k = round(1 / test_size)*.  This ensures that
    the test set spans the full range of the response distribution, consistent
    with the QSARINS stratified-split convention.

    Parameters
    ----------
    X : pd.DataFrame
        Descriptor matrix, shape (n_samples, n_descriptors).
    y : pd.Series
        Response vector, length n_samples.
    test_size : float, optional
        Approximate fraction of samples in the test set.  Default 0.2.

    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test)
        Four :class:`pd.DataFrame` / :class:`pd.Series` objects.
    """
    k = max(2, round(1.0 / test_size))
    sorted_idx = np.argsort(y.to_numpy())
    test_positions = sorted_idx[::k]
    train_positions = sorted_idx[~np.isin(sorted_idx, test_positions)]

    original_idx = y.index.to_numpy()
    te_idx = original_idx[test_positions]
    tr_idx = original_idx[train_positions]

    return (
        X.loc[tr_idx].copy(),
        X.loc[te_idx].copy(),
        y.loc[tr_idx].copy(),
        y.loc[te_idx].copy(),
    )
