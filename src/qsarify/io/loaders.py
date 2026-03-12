"""CSV data loading for QSARify.

Provides :func:`load_csv_dataset` which reads a CSV file in the standard
QSARINS layout and returns a validated :class:`DataSet`.

Expected CSV layout
-------------------
- Row 0: header (column names).
- Column 0 (optional): string-based compound ID/label.
- Middle columns: numeric descriptor values (X).
- Last column: numeric response variable (Y).

The ID column is detected automatically: if the first column cannot be
coerced to float, it is treated as an ID column; otherwise all columns are
treated as numeric.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qsarify.exceptions import DataImportError

__all__ = ["DataSet", "load_csv_dataset"]


@dataclass
class DataSet:
    """Validated dataset container returned by :func:`load_csv_dataset`.

    Parameters
    ----------
    X_df : pd.DataFrame
        Descriptor matrix (numeric, float64), shape (n_samples, n_descriptors).
    y_series : pd.Series
        Response variable (numeric, float64), length n_samples.
    id_series : pd.Series or None
        Compound ID/label column (string), length n_samples, or ``None`` if
        the CSV did not contain an ID column.
    """

    X_df: pd.DataFrame
    y_series: pd.Series
    id_series: pd.Series | None


def load_csv_dataset(path: Path | str) -> DataSet:
    """Load and validate a CSV dataset for QSAR/QSPR modelling.

    Parameters
    ----------
    path : Path or str
        Path to the CSV file.

    Returns
    -------
    DataSet
        Validated :class:`DataSet` containing the descriptor matrix,
        response series, and optional ID series.

    Raises
    ------
    DataImportError
        If the file is not found, is empty, has fewer than 2 numeric columns,
        contains non-numeric descriptor/response values, or contains missing
        values.
    """
    path = Path(path)

    if not path.exists():
        raise DataImportError(f"File not found: {path}")

    try:
        raw: pd.DataFrame = pd.read_csv(path, header=0)
    except Exception as exc:  # noqa: BLE001
        raise DataImportError(f"Failed to parse CSV: {exc}") from exc

    if raw.empty or len(raw) == 0:
        raise DataImportError("CSV file is empty or has no data rows")

    if raw.shape[1] <= 1:
        raise DataImportError("CSV must have at least 2 columns")

    # ------------------------------------------------------------------ #
    # Detect optional string ID column in position 0                       #
    # ------------------------------------------------------------------ #
    id_series: pd.Series | None = None
    first_col = raw.iloc[:, 0]
    has_id_col = _column_is_string(first_col)

    if has_id_col:
        id_series = first_col.astype(str)
        numeric_df = raw.iloc[:, 1:].copy()
    else:
        numeric_df = raw.copy()

    if numeric_df.shape[1] < 2:
        raise DataImportError(
            "CSV must have at least 2 numeric columns (one or more descriptors and a response)"
        )

    # ------------------------------------------------------------------ #
    # Split into X (all but last) and y (last)                             #
    # ------------------------------------------------------------------ #
    X_raw = numeric_df.iloc[:, :-1]
    y_raw = numeric_df.iloc[:, -1]

    # Validate missing values across X and y
    if X_raw.isnull().any().any() or y_raw.isnull().any():
        raise DataImportError(
            "CSV contains missing values; please handle missing data before loading"
        )

    # Coerce to float64 — non-numeric values will become NaN after coercion
    try:
        X_df = X_raw.apply(pd.to_numeric, errors="coerce")
    except Exception as exc:  # noqa: BLE001
        raise DataImportError(f"Failed to convert descriptor columns to numeric: {exc}") from exc

    if X_df.isnull().any().any():
        raise DataImportError(
            "Descriptor columns contain non-numeric values; all descriptor values must be numeric"
        )

    try:
        y_series = pd.to_numeric(y_raw, errors="coerce")
    except Exception as exc:  # noqa: BLE001
        raise DataImportError(f"Failed to convert response column to numeric: {exc}") from exc

    if y_series.isnull().any():
        raise DataImportError(
            "Response column contains non-numeric values; the response must be numeric"
        )

    # Cast dtypes
    X_df = X_df.astype("float64")
    y_series = y_series.astype("float64")

    return DataSet(X_df=X_df, y_series=y_series, id_series=id_series)


def _column_is_string(col: pd.Series) -> bool:
    """Return True if *col* cannot be entirely coerced to float (i.e. is an ID column)."""
    try:
        pd.to_numeric(col, errors="raise")
        return False
    except (ValueError, TypeError):
        return True
