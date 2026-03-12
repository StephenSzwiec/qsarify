"""Unit tests for qsarify.io.loaders."""

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from qsarify.exceptions import DataImportError
from qsarify.io.loaders import DataSet, load_csv_dataset


# ---------------------------------------------------------------------------
# Helpers — write temporary CSV files
# ---------------------------------------------------------------------------


def _write_csv(tmp_path: Path, content: str, name: str = "data.csv") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content).strip())
    return p


# ---------------------------------------------------------------------------
# Valid CSV — with ID column
# ---------------------------------------------------------------------------


VALID_WITH_ID = """
    id,d1,d2,d3,y
    mol1,1.0,2.0,3.0,10.0
    mol2,4.0,5.0,6.0,20.0
    mol3,7.0,8.0,9.0,30.0
"""


def test_load_valid_with_id(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, VALID_WITH_ID)
    ds = load_csv_dataset(csv_path)
    assert isinstance(ds, DataSet)
    assert ds.X_df.shape == (3, 3)
    assert list(ds.X_df.columns) == ["d1", "d2", "d3"]
    assert ds.y_series.name == "y"
    assert len(ds.y_series) == 3
    assert ds.id_series is not None
    assert list(ds.id_series) == ["mol1", "mol2", "mol3"]


# ---------------------------------------------------------------------------
# Valid CSV — without ID column (all numeric except last)
# ---------------------------------------------------------------------------


VALID_NO_ID = """
    d1,d2,d3,y
    1.0,2.0,3.0,10.0
    4.0,5.0,6.0,20.0
    7.0,8.0,9.0,30.0
"""


def test_load_valid_no_id(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, VALID_NO_ID)
    ds = load_csv_dataset(csv_path)
    assert ds.X_df.shape == (3, 3)
    assert ds.y_series.name == "y"
    assert ds.id_series is None


# ---------------------------------------------------------------------------
# Values round-trip correctly
# ---------------------------------------------------------------------------


def test_values_correct(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, VALID_WITH_ID)
    ds = load_csv_dataset(csv_path)
    assert ds.X_df.iloc[0]["d1"] == pytest.approx(1.0)
    assert ds.y_series.iloc[2] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_file_not_found() -> None:
    with pytest.raises(DataImportError, match="not found"):
        load_csv_dataset(Path("/nonexistent/path/data.csv"))


def test_empty_file(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, "")
    with pytest.raises(DataImportError):
        load_csv_dataset(csv_path)


def test_header_only(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, "id,d1,d2,y\n")
    with pytest.raises(DataImportError, match="no data"):
        load_csv_dataset(csv_path)


def test_non_numeric_descriptor(tmp_path: Path) -> None:
    content = """
        id,d1,d2,y
        mol1,abc,2.0,10.0
        mol2,4.0,5.0,20.0
    """
    csv_path = _write_csv(tmp_path, content)
    with pytest.raises(DataImportError, match="non-numeric"):
        load_csv_dataset(csv_path)


def test_non_numeric_response(tmp_path: Path) -> None:
    content = """
        id,d1,d2,y
        mol1,1.0,2.0,bad
        mol2,4.0,5.0,20.0
    """
    csv_path = _write_csv(tmp_path, content)
    with pytest.raises(DataImportError, match="non-numeric"):
        load_csv_dataset(csv_path)


def test_too_few_columns(tmp_path: Path) -> None:
    content = "y\n1.0\n2.0\n"
    csv_path = _write_csv(tmp_path, content)
    with pytest.raises(DataImportError, match="at least 2"):
        load_csv_dataset(csv_path)


def test_missing_values_raises(tmp_path: Path) -> None:
    content = """
        id,d1,d2,y
        mol1,1.0,,10.0
        mol2,4.0,5.0,20.0
    """
    csv_path = _write_csv(tmp_path, content)
    with pytest.raises(DataImportError, match="missing"):
        load_csv_dataset(csv_path)


# ---------------------------------------------------------------------------
# DataSet attributes are correct types
# ---------------------------------------------------------------------------


def test_dataset_types(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path, VALID_WITH_ID)
    ds = load_csv_dataset(csv_path)
    assert isinstance(ds.X_df, pd.DataFrame)
    assert isinstance(ds.y_series, pd.Series)
    assert isinstance(ds.id_series, pd.Series)
    assert ds.X_df.dtypes.apply(lambda t: t == "float64").all()
    assert ds.y_series.dtype == "float64"
