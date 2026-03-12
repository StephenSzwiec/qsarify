"""Descriptor and response scaling/normalization for QSAR/QSPR workflows.

Provides two scikit-learn-style scaler classes that operate on
:class:`pandas.DataFrame` objects and preserve column names through
transform/inverse-transform round trips:

- :class:`StandardScaler` — zero-mean, unit-variance (z-score) scaling.
- :class:`MinMaxScaler` — scales each column to a configurable [min, max] range.

Both classes follow the ``fit`` / ``transform`` / ``inverse_transform``
interface.  Calling ``transform`` before ``fit`` raises :class:`RuntimeError`.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["StandardScaler", "MinMaxScaler"]


class StandardScaler:
    """Zero-mean, unit-population-variance scaler.

    Parameters
    ----------
    None

    Examples
    --------
    >>> sc = StandardScaler()
    >>> X_scaled = sc.fit(X_train).transform(X_train)
    >>> X_orig = sc.inverse_transform(X_scaled)
    """

    def __init__(self) -> None:
        self._mean: pd.Series | None = None
        self._std: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "StandardScaler":
        """Compute mean and standard deviation from *X*.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.

        Returns
        -------
        StandardScaler
            *self* for method chaining.
        """
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0, ddof=0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply zero-mean unit-variance scaling.

        Parameters
        ----------
        X : pd.DataFrame
            Data to scale (must have same columns as *fit* data).

        Returns
        -------
        pd.DataFrame
            Scaled data with the same column names.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("StandardScaler must be fit before calling transform")
        std = self._std.replace(0.0, 1.0)  # avoid division by zero for constant cols
        return (X - self._mean) / std

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse the scaling applied by :meth:`transform`.

        Parameters
        ----------
        X : pd.DataFrame
            Scaled data.

        Returns
        -------
        pd.DataFrame
            Data in original units.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError(
                "StandardScaler must be fit before calling inverse_transform"
            )
        std = self._std.replace(0.0, 1.0)
        return X * std + self._mean


class MinMaxScaler:
    """Scale each column to a configurable [min, max] range.

    Parameters
    ----------
    feature_range : tuple of (float, float), optional
        Target range.  Default is (0.0, 1.0).

    Examples
    --------
    >>> sc = MinMaxScaler(feature_range=(0, 1))
    >>> X_scaled = sc.fit(X_train).transform(X_train)
    >>> X_orig = sc.inverse_transform(X_scaled)
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        self.feature_range = feature_range
        self._min: pd.Series | None = None
        self._max: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "MinMaxScaler":
        """Compute per-column min and max from *X*.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.

        Returns
        -------
        MinMaxScaler
            *self* for method chaining.
        """
        self._min = X.min(axis=0)
        self._max = X.max(axis=0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max scaling.

        Parameters
        ----------
        X : pd.DataFrame
            Data to scale.

        Returns
        -------
        pd.DataFrame
            Scaled data.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._min is None or self._max is None:
            raise RuntimeError("MinMaxScaler must be fit before calling transform")
        lo, hi = self.feature_range
        scale = self._max - self._min
        scale = scale.replace(0.0, 1.0)  # avoid division by zero
        X_std = (X - self._min) / scale
        return X_std * (hi - lo) + lo

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse min-max scaling.

        Parameters
        ----------
        X : pd.DataFrame
            Scaled data.

        Returns
        -------
        pd.DataFrame
            Data in original units.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if self._min is None or self._max is None:
            raise RuntimeError(
                "MinMaxScaler must be fit before calling inverse_transform"
            )
        lo, hi = self.feature_range
        scale = self._max - self._min
        scale = scale.replace(0.0, 1.0)
        return (X - lo) / (hi - lo) * scale + self._min
