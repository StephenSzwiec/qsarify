"""ResultSet: ordered collection of ModelResult objects with sort/filter/export.

A :class:`ResultSet` is the top-level container produced after a modeling run.
It provides convenience methods for exploring and ranking models before
generating detailed plots or validation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import pandas as pd

from qsarify.results.model_result import ModelResult

__all__ = ["ResultSet"]


# Scalar fields exported by to_dataframe().  Array-valued and nested fields
# (leverage, std_residuals, lmo_results, …) are intentionally excluded.
_SCALAR_FIELDS: list[str] = [
    "model_type",
    "n_features",
    "n_train",
    "n_test",
    "r_squared",
    "r_squared_adj",
    "rmse",
    "mae",
    "mse",
    "rss",
    "tss",
    "mss",
    "std_error_estimate",
    "f_statistic",
    "lof",
    "ccc",
    "q_squared_loo",
    "slope_origin",
    "slope_origin_reverse",
    "r_squared_origin",
    "r_squared_origin_reverse",
    "roy_r_squared_m_mean",
    "roy_r_squared_m_delta",
    "closeness",
    "closeness_reverse",
    "q_squared_f1",
    "q_squared_f2",
    "q_squared_f3",
    "r_squared_ext",
    "press_ext",
    "leverage_threshold",
]


@dataclass
class ResultSet:
    """Ordered collection of :class:`~qsarify.results.model_result.ModelResult` objects.

    Parameters
    ----------
    results : list[ModelResult], optional
        Initial list of results.  Defaults to an empty list.

    Examples
    --------
    >>> rs = ResultSet()
    >>> rs.add(some_model.get_results())
    >>> best_five = rs.sort("r_squared").results[:5]
    >>> df = rs.to_dataframe()
    """

    results: list[ModelResult] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    def add(self, result: ModelResult) -> None:
        """Append a :class:`ModelResult` to the collection.

        Parameters
        ----------
        result : ModelResult
            Result to add.
        """
        self.results.append(result)

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[ModelResult]:
        return iter(self.results)

    def __getitem__(self, idx: int) -> ModelResult:
        return self.results[idx]

    # ------------------------------------------------------------------
    # Sort
    # ------------------------------------------------------------------

    def sort(self, by: str, ascending: bool = False) -> "ResultSet":
        """Return a new :class:`ResultSet` sorted by a scalar field.

        Parameters
        ----------
        by : str
            Name of a scalar field in :class:`ModelResult` to sort by.
        ascending : bool, optional
            If ``True``, sort ascending (smallest first).  Default ``False``
            (largest first — appropriate for metrics like R² where higher
            is better).

        Returns
        -------
        ResultSet
            New :class:`ResultSet` with sorted results; the original is
            unchanged.

        Raises
        ------
        KeyError
            If *by* is not a valid :class:`ModelResult` field.
        """
        if not self.results:
            return ResultSet()

        if not hasattr(self.results[0], by):
            raise KeyError(f"ModelResult has no field {by!r}")

        # Sort key: treat None as ±inf so it is always placed last
        def _key(r: ModelResult) -> float:
            v = getattr(r, by)
            if v is None:
                return -np.inf if not ascending else np.inf
            return float(v)

        sorted_results = sorted(self.results, key=_key, reverse=not ascending)
        return ResultSet(results=sorted_results)

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def filter(
        self,
        model_type: str | None = None,
        n_features: int | None = None,
        min_r_squared: float | None = None,
        min_q_squared_loo: float | None = None,
        max_rmse: float | None = None,
        **kwargs: Any,
    ) -> "ResultSet":
        """Return a filtered subset of results.

        Parameters
        ----------
        model_type : str, optional
            Keep only results whose ``model_type`` matches exactly
            (e.g. ``"mlr"``, ``"ridge"``).
        n_features : int, optional
            Keep only results with exactly this many features.
        min_r_squared : float, optional
            Minimum R² threshold (inclusive).
        min_q_squared_loo : float, optional
            Minimum Q²_LOO threshold (inclusive).
        max_rmse : float, optional
            Maximum RMSE threshold (inclusive).
        **kwargs
            Additional field-value equality filters applied after the
            positional filters above.

        Returns
        -------
        ResultSet
            New :class:`ResultSet` containing only matching results.
        """
        filtered: list[ModelResult] = []
        for r in self.results:
            if model_type is not None and r.model_type != model_type:
                continue
            if n_features is not None and r.n_features != n_features:
                continue
            if min_r_squared is not None and (
                r.r_squared is None or r.r_squared < min_r_squared
            ):
                continue
            if min_q_squared_loo is not None and (
                r.q_squared_loo is None or r.q_squared_loo < min_q_squared_loo
            ):
                continue
            if max_rmse is not None and (r.rmse is None or r.rmse > max_rmse):
                continue
            # Generic equality filters
            if any(getattr(r, k, None) != v for k, v in kwargs.items()):
                continue
            filtered.append(r)
        return ResultSet(results=filtered)

    # ------------------------------------------------------------------
    # Best
    # ------------------------------------------------------------------

    def best(self, by: str = "r_squared", n: int = 1) -> list[ModelResult]:
        """Return the top *n* results ranked by *by* (descending).

        Parameters
        ----------
        by : str, optional
            Metric field to rank by.  Default ``"r_squared"``.
        n : int, optional
            Number of top results to return.

        Returns
        -------
        list[ModelResult]
            Up to *n* results with the highest *by* value.
        """
        return self.sort(by=by, ascending=False).results[:n]

    # ------------------------------------------------------------------
    # DataFrame export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Export scalar metrics for all results as a :class:`pandas.DataFrame`.

        Array-valued fields (``leverage``, ``std_residuals``, coefficient
        arrays) and nested structures (``lmo_results``,
        ``y_scrambling_results``) are excluded.

        Returns
        -------
        pandas.DataFrame
            One row per :class:`ModelResult`, one column per scalar field
            in :data:`_SCALAR_FIELDS`.
        """
        rows: list[dict[str, object]] = []
        for r in self.results:
            row: dict[str, object] = {f: getattr(r, f, None) for f in _SCALAR_FIELDS}
            rows.append(row)
        return pd.DataFrame(rows, columns=_SCALAR_FIELDS)
