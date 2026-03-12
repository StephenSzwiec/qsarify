"""SQLite3-backed project persistence for QSARify.

Implements the no-pickle persistence contract:

- Project metadata (state, config, descriptor names) stored as key-value
  pairs in ``project_meta`` (values are JSON-encoded strings).
- Numpy arrays (X_train, X_test, y_train, y_test, working X) stored as
  BLOBs in ``project_arrays`` with ``dtype`` and ``shape`` metadata.
- :class:`~qsarify.results.model_result.ModelResult` scalar fields stored
  as JSON in ``model_results``.
- :class:`~qsarify.results.model_result.ModelResult` array fields stored
  as BLOBs in ``model_result_arrays``.

No pickle is used anywhere.  Arrays round-trip via
``ndarray.tobytes()`` / ``numpy.frombuffer()``.

References
----------
workflow_fsm.md — Persistence contract section.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qsarify.exceptions import PersistenceError
from qsarify.io.loaders import DataSet
from qsarify.results.model_result import ModelResult

if TYPE_CHECKING:
    from qsarify.project import QSARProject

__all__ = ["save_project", "load_project"]

# Bump this when the schema changes in a breaking way.
_SCHEMA_VERSION = 1

# ModelResult fields that are stored as numpy BLOB rather than in JSON.
_ARRAY_FIELDS: frozenset[str] = frozenset(
    {
        "coef_std_errors",
        "coef_confidence_intervals",
        "coef_p_values",
        "leverage",
        "std_residuals",
        "y_train",
        "y_pred_train",
    }
)

_CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS project_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS project_arrays (
    name  TEXT PRIMARY KEY,
    dtype TEXT NOT NULL,
    shape TEXT NOT NULL,
    data  BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS model_results (
    idx     INTEGER PRIMARY KEY,
    scalars TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS model_result_arrays (
    idx        INTEGER NOT NULL,
    field_name TEXT    NOT NULL,
    dtype      TEXT    NOT NULL,
    shape      TEXT    NOT NULL,
    data       BLOB    NOT NULL,
    PRIMARY KEY (idx, field_name)
);
"""


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy / non-JSON-serialisable types to Python primitives.

    Parameters
    ----------
    obj : Any
        Value to convert.

    Returns
    -------
    Any
        JSON-serialisable equivalent.
    """
    if obj is None:
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


def _store_array(
    cursor: sqlite3.Cursor,
    table: str,
    *extra_keys: Any,
    arr: NDArray[Any],
) -> None:
    """Insert a numpy array as a BLOB row.

    Parameters
    ----------
    cursor : sqlite3.Cursor
    table : str
        Target table (``'project_arrays'`` or ``'model_result_arrays'``).
    extra_keys : Any
        Additional primary-key values prepended to the row.
    arr : ndarray
        Array to store.
    """
    if table == "project_arrays":
        cursor.execute(
            "INSERT OR REPLACE INTO project_arrays VALUES (?, ?, ?, ?)",
            (*extra_keys, str(arr.dtype), json.dumps(list(arr.shape)), arr.tobytes()),
        )
    else:
        cursor.execute(
            "INSERT OR REPLACE INTO model_result_arrays VALUES (?, ?, ?, ?, ?)",
            (*extra_keys, str(arr.dtype), json.dumps(list(arr.shape)), arr.tobytes()),
        )


def _load_array(dtype_str: str, shape_json: str, data: bytes) -> NDArray[Any]:
    """Restore a numpy array from BLOB storage.

    Parameters
    ----------
    dtype_str : str
        Numpy dtype string (e.g. ``'float64'``).
    shape_json : str
        JSON-encoded shape list (e.g. ``'[48, 8]'``).
    data : bytes
        Raw bytes from ``ndarray.tobytes()``.

    Returns
    -------
    ndarray
        Restored array (a writable copy).
    """
    shape = tuple(json.loads(shape_json))
    return np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape(shape).copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_project(project: "QSARProject", path: Path) -> None:
    """Serialise a :class:`~qsarify.project.QSARProject` to a SQLite3 file.

    Parameters
    ----------
    project : QSARProject
        Project to save.
    path : Path
        Destination file.  Created or fully overwritten.

    Raises
    ------
    PersistenceError
        If writing fails.
    """
    try:
        conn = sqlite3.connect(str(path))
    except sqlite3.Error as exc:
        raise PersistenceError(f"Cannot open '{path}' for writing: {exc}") from exc

    try:
        cur = conn.cursor()
        cur.executescript(_CREATE_SCHEMA)

        # Clear existing rows (full overwrite semantics)
        for tbl in (
            "project_meta",
            "project_arrays",
            "model_results",
            "model_result_arrays",
        ):
            cur.execute(f"DELETE FROM {tbl}")  # noqa: S608 — controlled table names

        # -- project_meta -------------------------------------------------
        meta: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "state": project.state.value,
            "random_seed": project.random_seed,
            "descriptor_names": project.descriptor_names,
            "config": project._config,
        }
        # Store y_series name and X_work column names for DATA_IMPORTED restore
        if project.dataset is not None:
            meta["y_series_name"] = project.dataset.y_series.name
        if project._X_work is not None:
            meta["X_work_columns"] = list(project._X_work.columns)

        for key, value in meta.items():
            cur.execute(
                "INSERT INTO project_meta VALUES (?, ?)",
                (key, json.dumps(_to_json_safe(value))),
            )

        # -- project_arrays -----------------------------------------------
        # Configured train/test arrays
        for name, arr in [
            ("X_train", project.X_train),
            ("X_test", project.X_test),
            ("y_train", project.y_train),
            ("y_test", project.y_test),
        ]:
            if arr is not None:
                _store_array(cur, "project_arrays", name, arr=arr)

        # Working X and y_series for DATA_IMPORTED restore
        if project._X_work is not None:
            _store_array(
                cur,
                "project_arrays",
                "X_work",
                arr=np.asarray(project._X_work, dtype=np.float64),
            )
        if project.dataset is not None:
            _store_array(
                cur,
                "project_arrays",
                "y_series",
                arr=np.asarray(project.dataset.y_series, dtype=np.float64),
            )

        # -- model_results / model_result_arrays --------------------------
        for idx, result in enumerate(project.result_set):
            # Build scalar dict (exclude array fields)
            scalars: dict[str, Any] = {}
            for fname, fval in vars(result).items():
                if fname in _ARRAY_FIELDS:
                    continue
                scalars[fname] = _to_json_safe(fval)

            cur.execute(
                "INSERT INTO model_results VALUES (?, ?)",
                (idx, json.dumps(scalars)),
            )

            # Store array fields as BLOBs
            for fname in _ARRAY_FIELDS:
                arr = getattr(result, fname, None)
                if arr is not None:
                    _store_array(cur, "model_result_arrays", idx, fname, arr=arr)

        conn.commit()
    except (sqlite3.Error, OSError) as exc:
        raise PersistenceError(f"Failed to write project to '{path}': {exc}") from exc
    finally:
        conn.close()


def load_project(path: Path) -> "QSARProject":
    """Deserialise a :class:`~qsarify.project.QSARProject` from a SQLite3 file.

    Parameters
    ----------
    path : Path
        Path to the ``.sqlite3`` project file.

    Returns
    -------
    QSARProject
        Fully restored project.

    Raises
    ------
    PersistenceError
        If the file does not exist, is unreadable, or has an incompatible schema.
    """
    from qsarify.project import ProjectState, QSARProject  # avoid circular import

    if not path.exists():
        raise PersistenceError(f"Project file not found: '{path}'")

    try:
        conn = sqlite3.connect(str(path))
    except sqlite3.Error as exc:
        raise PersistenceError(f"Cannot open '{path}': {exc}") from exc

    try:
        cur = conn.cursor()

        # -- project_meta -------------------------------------------------
        cur.execute("SELECT key, value FROM project_meta")
        meta: dict[str, Any] = {k: json.loads(v) for k, v in cur.fetchall()}

        schema_ver = meta.get("schema_version", 1)
        if schema_ver != _SCHEMA_VERSION:
            raise PersistenceError(
                f"Incompatible save file schema version {schema_ver} "
                f"(expected {_SCHEMA_VERSION})."
            )

        random_seed: int | None = meta.get("random_seed")
        project = QSARProject(random_seed=random_seed)
        project.state = ProjectState(int(meta["state"]))
        project.descriptor_names = meta.get("descriptor_names") or []
        project._config = meta.get("config") or {}

        # -- project_arrays -----------------------------------------------
        cur.execute("SELECT name, dtype, shape, data FROM project_arrays")
        for name, dtype_str, shape_json, data in cur.fetchall():
            arr = _load_array(dtype_str, shape_json, data)
            if name in ("X_train", "X_test", "y_train", "y_test"):
                setattr(project, name, arr)
            elif name == "X_work":
                cols = meta.get("X_work_columns") or []
                project._X_work = pd.DataFrame(arr, columns=cols)
            elif name == "y_series":
                y_name = meta.get("y_series_name")
                y_series = pd.Series(arr, name=y_name)
                # Reconstruct a minimal DataSet so configure_variables works
                X_work = project._X_work
                if X_work is not None:
                    project.dataset = DataSet(
                        X_df=X_work,
                        y_series=y_series,
                        id_series=None,
                    )

        # -- model_results ------------------------------------------------
        cur.execute("SELECT idx, scalars FROM model_results ORDER BY idx")
        result_rows = cur.fetchall()

        # Load per-result array BLOBs grouped by index
        cur.execute(
            "SELECT idx, field_name, dtype, shape, data FROM model_result_arrays"
        )
        arrays_by_idx: dict[int, dict[str, NDArray[Any]]] = defaultdict(dict)
        for idx, field_name, dtype_str, shape_json, data in cur.fetchall():
            arrays_by_idx[idx][field_name] = _load_array(dtype_str, shape_json, data)

        for idx, scalars_json in result_rows:
            scalars: dict[str, Any] = json.loads(scalars_json)

            # JSON converts all dict keys to strings; restore int keys for
            # cluster_assignments (descriptor_index → cluster_id).
            if scalars.get("cluster_assignments") is not None:
                scalars["cluster_assignments"] = {
                    int(k): int(v) for k, v in scalars["cluster_assignments"].items()
                }

            arr_fields = arrays_by_idx.get(idx, {})
            result = ModelResult(**{**scalars, **arr_fields})
            project.result_set.add(result)

        return project

    except (sqlite3.Error, KeyError, ValueError) as exc:
        raise PersistenceError(f"Failed to load project from '{path}': {exc}") from exc
    finally:
        conn.close()
