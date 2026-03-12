"""Unit tests for qsarify.preprocessing.splitters."""

import numpy as np
import pandas as pd

from qsarify.preprocessing.splitters import random_split, stratified_split


N = 50
RNG_SEED = 42

X = pd.DataFrame(
    np.arange(N * 3, dtype=float).reshape(N, 3), columns=["d1", "d2", "d3"]
)
y = pd.Series(np.linspace(0.0, 10.0, N), name="y")


# ---------------------------------------------------------------------------
# random_split
# ---------------------------------------------------------------------------


def test_random_split_sizes() -> None:
    X_tr, X_te, y_tr, y_te = random_split(X, y, test_size=0.2, random_seed=RNG_SEED)
    assert len(X_tr) == 40
    assert len(X_te) == 10
    assert len(y_tr) == 40
    assert len(y_te) == 10


def test_random_split_no_overlap() -> None:
    X_tr, X_te, y_tr, y_te = random_split(X, y, test_size=0.2, random_seed=RNG_SEED)
    tr_idx = set(X_tr.index)
    te_idx = set(X_te.index)
    assert tr_idx.isdisjoint(te_idx)
    assert tr_idx | te_idx == set(X.index)


def test_random_split_returns_dataframes() -> None:
    X_tr, X_te, y_tr, y_te = random_split(X, y, test_size=0.2, random_seed=RNG_SEED)
    assert isinstance(X_tr, pd.DataFrame)
    assert isinstance(X_te, pd.DataFrame)
    assert isinstance(y_tr, pd.Series)
    assert isinstance(y_te, pd.Series)


def test_random_split_reproducible() -> None:
    a = random_split(X, y, test_size=0.2, random_seed=RNG_SEED)
    b = random_split(X, y, test_size=0.2, random_seed=RNG_SEED)
    assert list(a[0].index) == list(b[0].index)


def test_random_split_different_seeds() -> None:
    a = random_split(X, y, test_size=0.2, random_seed=1)
    b = random_split(X, y, test_size=0.2, random_seed=2)
    # With high probability, two different seeds yield different splits
    assert list(a[0].index) != list(b[0].index)


# ---------------------------------------------------------------------------
# stratified_split  (ordered by response value)
# ---------------------------------------------------------------------------


def test_stratified_split_sizes() -> None:
    X_tr, X_te, y_tr, y_te = stratified_split(X, y, test_size=0.2)
    assert len(X_tr) == 40
    assert len(X_te) == 10


def test_stratified_split_no_overlap() -> None:
    X_tr, X_te, y_tr, y_te = stratified_split(X, y, test_size=0.2)
    tr_idx = set(X_tr.index)
    te_idx = set(X_te.index)
    assert tr_idx.isdisjoint(te_idx)
    assert tr_idx | te_idx == set(X.index)


def test_stratified_split_coverage() -> None:
    """Test set should contain samples spread across the response range."""
    X_tr, X_te, y_tr, y_te = stratified_split(X, y, test_size=0.2)
    # With stratification the test set min should be near overall min
    # and max near overall max (every k-th sample selected)
    assert y_te.min() < y.mean()
    assert y_te.max() > y.mean()
