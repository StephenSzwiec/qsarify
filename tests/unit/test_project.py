"""Unit tests for QSARProject FSM workflow controller.

Tests cover:
- FSM state transitions and enforcement
- WorkflowError on invalid operations
- WorkflowRegressionWarning on backward transitions
- All build_* methods
- run_lmo_validation and run_y_scrambling
- save/load round-trip (delegated to persistence)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsarify.exceptions import WorkflowError, WorkflowRegressionWarning
from qsarify.project import ProjectState, QSARProject


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Write a small regression CSV with 60 samples and 8 descriptors."""
    rng = np.random.default_rng(42)
    n, p = 60, 8
    X = rng.standard_normal((n, p))
    y = X[:, 0] * 2.5 + X[:, 1] * (-1.2) + rng.standard_normal(n) * 0.3
    cols = [f"D{i}" for i in range(p)] + ["activity"]
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def configured_project(sample_csv: Path) -> QSARProject:
    """QSARProject in DATA_CONFIGURED state (no models yet)."""
    p = QSARProject(random_seed=42)
    p.load_data(sample_csv)
    p.configure_variables(y_col="activity", test_size=0.2)
    return p


@pytest.fixture
def built_project(configured_project: QSARProject) -> QSARProject:
    """QSARProject in MODELS_BUILT state with one Ridge model."""
    configured_project.build_ridge(alpha=1.0)
    return configured_project


# ---------------------------------------------------------------------------
# 1. Initial state
# ---------------------------------------------------------------------------


def test_initial_state_is_empty() -> None:
    p = QSARProject()
    assert p.state == ProjectState.EMPTY


def test_initial_result_set_is_empty() -> None:
    p = QSARProject()
    assert len(p.result_set) == 0


def test_initial_models_list_is_empty() -> None:
    p = QSARProject()
    assert len(p._models) == 0


def test_initial_dataset_is_none() -> None:
    p = QSARProject()
    assert p.dataset is None


# ---------------------------------------------------------------------------
# 2. load_data → DATA_IMPORTED
# ---------------------------------------------------------------------------


def test_load_data_transitions_to_data_imported(sample_csv: Path) -> None:
    p = QSARProject()
    p.load_data(sample_csv)
    assert p.state == ProjectState.DATA_IMPORTED


def test_load_data_populates_dataset(sample_csv: Path) -> None:
    p = QSARProject()
    p.load_data(sample_csv)
    assert p.dataset is not None
    assert p.dataset.X_df.shape == (60, 8)
    assert len(p.dataset.y_series) == 60


def test_load_data_invalid_path_raises() -> None:
    p = QSARProject()
    with pytest.raises(Exception):  # DataImportError or OSError
        p.load_data("/nonexistent/path/file.csv")


# ---------------------------------------------------------------------------
# 3. filter_descriptors (DATA_IMPORTED only)
# ---------------------------------------------------------------------------


def test_filter_descriptors_requires_data_imported(sample_csv: Path) -> None:
    p = QSARProject()
    with pytest.raises(WorkflowError):
        p.filter_descriptors()


def test_filter_descriptors_stays_in_data_imported(sample_csv: Path) -> None:
    p = QSARProject()
    p.load_data(sample_csv)
    p.filter_descriptors(constant_threshold=0.01, correlation_threshold=0.95)
    assert p.state == ProjectState.DATA_IMPORTED


def test_filter_descriptors_reduces_columns(sample_csv: Path) -> None:
    """Filtering at low correlation threshold should reduce columns."""
    p = QSARProject()
    p.load_data(sample_csv)
    # Very tight threshold should keep only a subset
    original_cols = p.dataset.X_df.shape[1]  # type: ignore[union-attr]
    p.filter_descriptors(constant_threshold=0.0, correlation_threshold=0.5)
    assert p._X_work is not None
    assert p._X_work.shape[1] <= original_cols


# ---------------------------------------------------------------------------
# 4. configure_variables → DATA_CONFIGURED
# ---------------------------------------------------------------------------


def test_configure_from_empty_raises_workflow_error(sample_csv: Path) -> None:
    p = QSARProject()
    with pytest.raises(WorkflowError):
        p.configure_variables(y_col="activity")


def test_configure_transitions_to_data_configured(sample_csv: Path) -> None:
    p = QSARProject(random_seed=42)
    p.load_data(sample_csv)
    p.configure_variables(y_col="activity", test_size=0.2)
    assert p.state == ProjectState.DATA_CONFIGURED


def test_configure_populates_train_test_arrays(sample_csv: Path) -> None:
    p = QSARProject(random_seed=42)
    p.load_data(sample_csv)
    p.configure_variables(y_col="activity", test_size=0.2)
    assert p.X_train is not None
    assert p.X_test is not None
    assert p.y_train is not None
    assert p.y_test is not None
    # 80/20 split of 60 samples
    assert p.X_train.shape[0] + p.X_test.shape[0] == 60
    assert len(p.y_train) + len(p.y_test) == 60


def test_configure_stores_descriptor_names(sample_csv: Path) -> None:
    p = QSARProject(random_seed=42)
    p.load_data(sample_csv)
    p.configure_variables(y_col="activity")
    assert len(p.descriptor_names) == 8
    assert all(n.startswith("D") for n in p.descriptor_names)


def test_configure_unknown_y_col_raises(sample_csv: Path) -> None:
    p = QSARProject()
    p.load_data(sample_csv)
    with pytest.raises(WorkflowError):
        p.configure_variables(y_col="nonexistent_column")


def test_configure_unknown_x_col_raises(sample_csv: Path) -> None:
    p = QSARProject()
    p.load_data(sample_csv)
    with pytest.raises(WorkflowError):
        p.configure_variables(y_col="activity", x_cols=["D0", "BOGUS"])


def test_configure_stratified_split(sample_csv: Path) -> None:
    p = QSARProject(random_seed=42)
    p.load_data(sample_csv)
    p.configure_variables(y_col="activity", test_size=0.2, split_method="stratified")
    assert p.state == ProjectState.DATA_CONFIGURED
    assert p.X_train is not None


def test_configure_invalid_split_method_raises(sample_csv: Path) -> None:
    p = QSARProject()
    p.load_data(sample_csv)
    with pytest.raises(WorkflowError):
        p.configure_variables(y_col="activity", split_method="unknown")


# ---------------------------------------------------------------------------
# 5. build_ridge from wrong states
# ---------------------------------------------------------------------------


def test_build_ridge_from_empty_raises(sample_csv: Path) -> None:
    p = QSARProject()
    with pytest.raises(WorkflowError):
        p.build_ridge()


def test_build_ridge_from_data_imported_raises(sample_csv: Path) -> None:
    p = QSARProject()
    p.load_data(sample_csv)
    with pytest.raises(WorkflowError):
        p.build_ridge()


# ---------------------------------------------------------------------------
# 6. build_* methods → MODELS_BUILT
# ---------------------------------------------------------------------------


def test_build_ridge_transitions_to_models_built(configured_project: QSARProject) -> None:
    configured_project.build_ridge(alpha=1.0)
    assert configured_project.state == ProjectState.MODELS_BUILT


def test_build_ridge_adds_to_result_set(configured_project: QSARProject) -> None:
    configured_project.build_ridge(alpha=1.0)
    assert len(configured_project.result_set) == 1
    assert configured_project.result_set[0].model_type == "ridge"


def test_build_lasso_transitions_to_models_built(configured_project: QSARProject) -> None:
    configured_project.build_lasso(alpha=0.1)
    assert configured_project.state == ProjectState.MODELS_BUILT


def test_build_multiple_models_accumulates(configured_project: QSARProject) -> None:
    configured_project.build_ridge(alpha=1.0)
    configured_project.build_lasso(alpha=0.1)
    assert configured_project.state == ProjectState.MODELS_BUILT
    assert len(configured_project.result_set) == 2


def test_build_svr_transitions_to_models_built(configured_project: QSARProject) -> None:
    configured_project.build_svr()
    assert configured_project.state == ProjectState.MODELS_BUILT


def test_build_random_forest_transitions_to_models_built(configured_project: QSARProject) -> None:
    configured_project.build_random_forest(n_estimators=10)
    assert configured_project.state == ProjectState.MODELS_BUILT


def test_build_gradient_boosting_transitions_to_models_built(
    configured_project: QSARProject,
) -> None:
    configured_project.build_gradient_boosting(n_estimators=10)
    assert configured_project.state == ProjectState.MODELS_BUILT


def test_build_ga_mlr_transitions_to_models_built(configured_project: QSARProject) -> None:
    configured_project.build_ga_mlr(
        max_variables=2,
        population_size=10,
        max_generations=5,
        keep_best=2,
        exhaustive_max_vars=0,
    )
    assert configured_project.state == ProjectState.MODELS_BUILT
    assert len(configured_project.result_set) > 0


# ---------------------------------------------------------------------------
# 7. run_lmo_validation from wrong state
# ---------------------------------------------------------------------------


def test_run_lmo_from_data_configured_raises(configured_project: QSARProject) -> None:
    with pytest.raises(WorkflowError):
        configured_project.run_lmo_validation([])


def test_run_y_scrambling_from_data_configured_raises(configured_project: QSARProject) -> None:
    with pytest.raises(WorkflowError):
        configured_project.run_y_scrambling([])


# ---------------------------------------------------------------------------
# 8. run_lmo_validation → MODELS_EVALUATED
# ---------------------------------------------------------------------------


def test_run_lmo_transitions_to_evaluated(built_project: QSARProject) -> None:
    built_project.run_lmo_validation([0], n_iterations=10)
    assert built_project.state == ProjectState.MODELS_EVALUATED


def test_run_lmo_populates_lmo_results(built_project: QSARProject) -> None:
    built_project.run_lmo_validation([0], n_iterations=10)
    result = built_project.result_set[0]
    assert result.lmo_results is not None
    assert "mean_q2" in result.lmo_results
    assert "std_q2" in result.lmo_results


def test_run_y_scrambling_transitions_to_evaluated(built_project: QSARProject) -> None:
    built_project.run_y_scrambling([0], n_iterations=20)
    assert built_project.state == ProjectState.MODELS_EVALUATED


def test_run_y_scrambling_populates_results(built_project: QSARProject) -> None:
    built_project.run_y_scrambling([0], n_iterations=20)
    result = built_project.result_set[0]
    assert result.y_scrambling_results is not None
    assert "mean_r2_scrambled" in result.y_scrambling_results


# ---------------------------------------------------------------------------
# 9. Backward regression warnings and state clearing
# ---------------------------------------------------------------------------


def test_reconfigure_from_models_built_warns(built_project: QSARProject) -> None:
    with pytest.warns(WorkflowRegressionWarning):
        built_project.configure_variables(y_col="activity", test_size=0.2)


def test_reconfigure_from_models_built_clears_models(
    built_project: QSARProject,
    sample_csv: Path,
) -> None:
    with pytest.warns(WorkflowRegressionWarning):
        built_project.configure_variables(y_col="activity", test_size=0.2)
    assert len(built_project.result_set) == 0
    assert len(built_project._models) == 0
    assert built_project.state == ProjectState.DATA_CONFIGURED


def test_reload_data_from_models_built_warns(
    built_project: QSARProject, sample_csv: Path
) -> None:
    with pytest.warns(WorkflowRegressionWarning):
        built_project.load_data(sample_csv)


def test_reload_data_clears_all_downstream(
    built_project: QSARProject, sample_csv: Path
) -> None:
    with pytest.warns(WorkflowRegressionWarning):
        built_project.load_data(sample_csv)
    assert built_project.X_train is None
    assert len(built_project.result_set) == 0
    assert built_project.state == ProjectState.DATA_IMPORTED


def test_rebuild_from_models_evaluated_warns(built_project: QSARProject) -> None:
    built_project.run_lmo_validation([0], n_iterations=5)
    assert built_project.state == ProjectState.MODELS_EVALUATED
    with pytest.warns(WorkflowRegressionWarning):
        built_project.build_ridge(alpha=2.0)
    # Validation results on old model should be cleared
    assert built_project.result_set[0].lmo_results is None
    assert built_project.state == ProjectState.MODELS_BUILT


# ---------------------------------------------------------------------------
# 10. save / load round-trip
# ---------------------------------------------------------------------------


def test_save_creates_file(built_project: QSARProject, tmp_path: Path) -> None:
    save_path = tmp_path / "project.sqlite3"
    built_project.save(save_path)
    assert save_path.exists()


def test_save_load_state_round_trip(built_project: QSARProject, tmp_path: Path) -> None:
    save_path = tmp_path / "project.sqlite3"
    built_project.save(save_path)
    loaded = QSARProject.load(save_path)
    assert loaded.state == built_project.state


def test_save_load_result_count_round_trip(
    built_project: QSARProject, tmp_path: Path
) -> None:
    save_path = tmp_path / "project.sqlite3"
    built_project.save(save_path)
    loaded = QSARProject.load(save_path)
    assert len(loaded.result_set) == len(built_project.result_set)


def test_save_load_arrays_round_trip(built_project: QSARProject, tmp_path: Path) -> None:
    save_path = tmp_path / "project.sqlite3"
    built_project.save(save_path)
    loaded = QSARProject.load(save_path)
    assert loaded.X_train is not None
    assert loaded.y_train is not None
    np.testing.assert_array_almost_equal(loaded.X_train, built_project.X_train)  # type: ignore[arg-type]
    np.testing.assert_array_almost_equal(loaded.y_train, built_project.y_train)  # type: ignore[arg-type]


def test_save_load_preserves_random_seed(built_project: QSARProject, tmp_path: Path) -> None:
    save_path = tmp_path / "project.sqlite3"
    built_project.save(save_path)
    loaded = QSARProject.load(save_path)
    assert loaded.random_seed == built_project.random_seed


def test_checkpoint_path_auto_saves(sample_csv: Path, tmp_path: Path) -> None:
    """Auto-checkpointing saves after each forward transition."""
    ckpt = tmp_path / "ckpt.sqlite3"
    p = QSARProject(random_seed=42, checkpoint_path=ckpt)
    assert not ckpt.exists()
    p.load_data(sample_csv)
    assert ckpt.exists()  # saved after DATA_IMPORTED
    p.configure_variables(y_col="activity", test_size=0.2)
    p.build_ridge(alpha=1.0)
    loaded = QSARProject.load(ckpt)
    assert loaded.state == ProjectState.MODELS_BUILT
    assert len(loaded.result_set) == 1


def test_save_load_result_metrics_round_trip(
    built_project: QSARProject, tmp_path: Path
) -> None:
    save_path = tmp_path / "project.sqlite3"
    built_project.save(save_path)
    loaded = QSARProject.load(save_path)
    orig = built_project.result_set[0]
    saved = loaded.result_set[0]
    assert saved.model_type == orig.model_type
    assert saved.n_features == orig.n_features
    assert pytest.approx(saved.r_squared, abs=1e-9) == orig.r_squared
    assert pytest.approx(saved.rmse, abs=1e-9) == orig.rmse
