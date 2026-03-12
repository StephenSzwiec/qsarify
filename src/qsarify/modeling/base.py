"""Abstract base class for all QSARify models.

This module defines *only* the interface contract.  Metric computation lives
in :mod:`qsarify.modeling.metrics`, OLS specifics in
:mod:`qsarify.modeling.subset_model`, and sklearn wrappers in
:mod:`qsarify.modeling.models`.
"""

from __future__ import annotations

import abc

import numpy as np
from numpy.typing import NDArray

from qsarify.results.model_result import ModelResult

__all__ = ["BaseModel"]


class BaseModel(abc.ABC):
    """Abstract base class for all QSARify models.

    Every concrete model must implement :meth:`fit`, :meth:`predict`, and
    :meth:`get_results`.  The ``fit`` signature includes optional external
    test arrays so that external-validation metrics can be computed uniformly
    across all model types.
    """

    @abc.abstractmethod
    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_test: NDArray[np.float64] | None = None,
        y_test: NDArray[np.float64] | None = None,
    ) -> "BaseModel":
        """Train the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training descriptor matrix.
        y : ndarray of shape (n_samples,)
            Training response vector.
        X_test : ndarray of shape (n_test, n_features) or None, optional
            External test descriptor matrix.  When provided, external
            validation metrics (Q²_F1/F2/F3, R²_ext, PRESS_ext) are computed.
        y_test : ndarray of shape (n_test,) or None, optional
            External test response vector.

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
        """Return a :class:`~qsarify.results.model_result.ModelResult`.

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
