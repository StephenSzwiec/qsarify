"""QSARProject: FSM-backed workflow controller for QSAR/QSPR modeling.

The project FSM enforces valid operation ordering::

    EMPTY → DATA_IMPORTED → DATA_CONFIGURED → MODELS_BUILT → MODELS_EVALUATED

Backward transitions issue a :class:`~qsarify.exceptions.WorkflowRegressionWarning`
and clear all downstream artifacts before re-executing the step.
Save/load delegates to :mod:`qsarify.io.persistence` (SQLite3, no pickle).

References
----------
workflow_fsm.md — FSM states, transitions, backward regression rules,
persistence contract, and QSARINS reference workflow narrative.
"""

from __future__ import annotations

import enum
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.svm import SVR

from qsarify.exceptions import WorkflowError, WorkflowRegressionWarning
from qsarify.io.loaders import DataSet, load_csv_dataset
from qsarify.modeling.base import BaseModel
from qsarify.modeling.clustering import cophenetic_cluster
from qsarify.modeling.genetic_algorithm import enumerate_subsets, run_ga_mlr
from qsarify.modeling.models import (
    GradientBoostingModel,
    LassoModel,
    RandomForestModel,
    RidgeModel,
    SVRModel,
)
from qsarify.preprocessing.filters import remove_high_correlation, remove_near_zero_variance
from qsarify.preprocessing.scalers import StandardScaler
from qsarify.preprocessing.splitters import random_split, stratified_split
from qsarify.results.model_result import ModelResult
from qsarify.results.result_set import ResultSet
from qsarify.validation.procedures import run_lmo, run_y_scrambling

__all__ = ["ProjectState", "QSARProject"]


# ---------------------------------------------------------------------------
# FSM state enum
# ---------------------------------------------------------------------------


class ProjectState(enum.IntEnum):
    """Ordered FSM states for :class:`QSARProject`.

    Each integer value represents a stage in the QSAR modeling workflow.
    Operations that target a lower or equal state from the current one
    constitute backward transitions and trigger :class:`WorkflowRegressionWarning`.
    """

    EMPTY = 0
    DATA_IMPORTED = 1
    DATA_CONFIGURED = 2
    MODELS_BUILT = 3
    MODELS_EVALUATED = 4


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _make_estimator(model_type: str, hyperparameters: dict[str, Any]) -> Any:
    """Create a fresh, unfitted sklearn estimator from a *model_type* string.

    Used by :meth:`QSARProject.run_lmo_validation` and
    :meth:`QSARProject.run_y_scrambling` to reconstruct an estimator from
    the :class:`~qsarify.results.model_result.ModelResult` hyperparameter dict
    without requiring the original fitted model object.

    Parameters
    ----------
    model_type : str
        One of ``'mlr'``, ``'ridge'``, ``'lasso'``, ``'svr'``, ``'rf'``, ``'gbr'``.
    hyperparameters : dict
        Model hyperparameters, as stored in :attr:`ModelResult.hyperparameters`.

    Returns
    -------
    sklearn estimator
        Unfitted estimator ready for cloning in validation workers.

    Raises
    ------
    WorkflowError
        If *model_type* is not recognised.
    """
    if model_type == "mlr":
        return LinearRegression()
    if model_type == "ridge":
        return Ridge(alpha=float(hyperparameters.get("alpha", 1.0)), fit_intercept=True)
    if model_type == "lasso":
        return Lasso(
            alpha=float(hyperparameters.get("alpha", 0.1)),
            max_iter=int(hyperparameters.get("max_iter", 10000)),
            fit_intercept=True,
        )
    if model_type == "svr":
        return SVR(
            C=float(hyperparameters.get("C", 1.0)),
            gamma=hyperparameters.get("gamma", "scale"),
            kernel=str(hyperparameters.get("kernel", "rbf")),
        )
    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=int(hyperparameters.get("n_estimators", 100)),
            max_features=hyperparameters.get("max_features", "sqrt"),
        )
    if model_type == "gbr":
        return GradientBoostingRegressor(
            n_estimators=int(hyperparameters.get("n_estimators", 100)),
            learning_rate=float(hyperparameters.get("learning_rate", 0.1)),
            max_depth=int(hyperparameters.get("max_depth", 3)),
        )
    raise WorkflowError(f"Unknown model_type '{model_type}' for estimator reconstruction.")


def _X_for_validation(
    X_train: NDArray[np.float64],
    result: ModelResult,
) -> NDArray[np.float64]:
    """Return the feature submatrix used by a model.

    For :class:`~qsarify.modeling.base.SubsetModel` results the selected
    descriptor columns are extracted from *X_train*.  For full-descriptor
    models the full matrix is returned unchanged.

    Parameters
    ----------
    X_train : ndarray of shape (n_train, p)
        Full training descriptor matrix.
    result : ModelResult
        Model result whose :attr:`~ModelResult.selected_descriptors` is used.

    Returns
    -------
    ndarray of shape (n_train, p_model)
        Descriptor submatrix for this model.
    """
    if result.selected_descriptors is not None:
        cols = np.array(result.selected_descriptors, dtype=np.intp)
        return X_train[:, cols]
    return X_train


# ---------------------------------------------------------------------------
# QSARProject
# ---------------------------------------------------------------------------


class QSARProject:
    """FSM-backed workflow controller for QSAR/QSPR modeling.

    Provides a high-level API that mirrors the QSARINS workflow:

    1. :meth:`load_data` — import a CSV dataset.
    2. :meth:`filter_descriptors` — remove near-constant / highly correlated
       columns (optional).
    3. :meth:`configure_variables` — select response/descriptor columns,
       normalise, and split into train/test.
    4. :meth:`build_ga_mlr` / :meth:`build_ridge` / … — fit one or more models.
    5. :meth:`run_lmo_validation` / :meth:`run_y_scrambling` — validate selected
       models.
    6. :meth:`save` / :meth:`load` — persist or restore the project.

    Backward transitions (re-running an earlier step when a later step has
    already been executed) issue :class:`~qsarify.exceptions.WorkflowRegressionWarning`
    and clear all downstream artifacts automatically.

    Parameters
    ----------
    random_seed : int or None
        Global random seed for all stochastic operations.  ``None`` means
        non-deterministic runs (effective seeds are not currently logged).
    checkpoint_path : Path or str or None
        If set, the project is automatically saved to this file after every
        forward FSM state transition.  ``None`` disables auto-checkpointing.
    """

    def __init__(
        self,
        random_seed: int | None = None,
        checkpoint_path: Path | str | None = None,
    ) -> None:
        self.random_seed: int | None = random_seed
        self.checkpoint_path: Path | None = Path(checkpoint_path) if checkpoint_path else None
        self.state: ProjectState = ProjectState.EMPTY

        # --- Data storage ---
        self.dataset: DataSet | None = None
        self._X_work: pd.DataFrame | None = None  # filtered X (after filter_descriptors)
        self.descriptor_names: list[str] = []

        self.X_train: NDArray[np.float64] | None = None
        self.X_test: NDArray[np.float64] | None = None
        self.y_train: NDArray[np.float64] | None = None
        self.y_test: NDArray[np.float64] | None = None

        # --- Model storage ---
        self.result_set: ResultSet = ResultSet()
        self._models: list[BaseModel] = []  # fitted model objects (not persisted)

        # --- Configuration (persisted as JSON) ---
        self._config: dict[str, Any] = {"random_seed": random_seed}

    # -----------------------------------------------------------------------
    # FSM helpers
    # -----------------------------------------------------------------------

    def _require_state(self, min_state: ProjectState, op_name: str) -> None:
        """Raise :class:`~qsarify.exceptions.WorkflowError` if state < *min_state*."""
        if self.state < min_state:
            raise WorkflowError(
                f"'{op_name}' requires state >= {min_state.name} "
                f"but current state is {self.state.name}."
            )

    def _clear_from(self, state: ProjectState) -> None:
        """Clear all project state at and above *state*.

        Parameters
        ----------
        state : ProjectState
            The lowest state to clear (inclusive).
        """
        if state <= ProjectState.MODELS_EVALUATED:
            # Strip validation results from every ModelResult in result_set
            for result in self.result_set:
                result.lmo_results = None
                result.y_scrambling_results = None
        if state <= ProjectState.MODELS_BUILT:
            self.result_set = ResultSet()
            self._models = []
        if state <= ProjectState.DATA_CONFIGURED:
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None
            self.descriptor_names = []
        if state <= ProjectState.DATA_IMPORTED:
            self.dataset = None
            self._X_work = None

    def _maybe_backward(self, target_state: ProjectState) -> None:
        """Warn and clear downstream state if already at or past *target_state*.

        A backward (or same-level repeated) transition is detected when
        ``self.state >= target_state``.  In that case:

        1. :class:`~qsarify.exceptions.WorkflowRegressionWarning` is issued.
        2. :meth:`_clear_from` clears ``target_state`` and everything above.
        3. ``self.state`` is rewound to ``target_state - 1``.

        The caller is responsible for advancing ``self.state`` after completing
        the operation.

        Parameters
        ----------
        target_state : ProjectState
            The state the next operation will transition *into*.
        """
        if self.state.value >= target_state.value:
            warnings.warn(
                f"Re-executing step targeting {target_state.name} while in "
                f"{self.state.name}. All downstream state will be cleared.",
                WorkflowRegressionWarning,
                stacklevel=3,
            )
            self._clear_from(target_state)
            self.state = ProjectState(target_state.value - 1)

    def _checkpoint(self) -> None:
        """Save to :attr:`checkpoint_path` if auto-checkpointing is enabled."""
        if self.checkpoint_path is not None:
            self.save(self.checkpoint_path)

    # -----------------------------------------------------------------------
    # Phase I: Data import
    # -----------------------------------------------------------------------

    def load_data(self, path: Path | str) -> None:
        """Load and validate a CSV dataset.

        Transitions to :attr:`~ProjectState.DATA_IMPORTED`.  Backward
        transition warning issued if already in ``DATA_IMPORTED`` or later.

        Parameters
        ----------
        path : Path or str
            Path to the CSV file.  See :func:`~qsarify.io.loaders.load_csv_dataset`
            for the expected layout.

        Raises
        ------
        DataImportError
            If the CSV cannot be parsed or validated.
        WorkflowRegressionWarning
            (Warning) If already at ``DATA_IMPORTED`` or later state.
        """
        self._maybe_backward(ProjectState.DATA_IMPORTED)
        self.dataset = load_csv_dataset(path)
        self._X_work = self.dataset.X_df.copy()
        self.state = ProjectState.DATA_IMPORTED
        self._checkpoint()

    def filter_descriptors(
        self,
        constant_threshold: float = 0.01,
        correlation_threshold: float = 0.90,
    ) -> None:
        """Remove near-constant and highly correlated descriptor columns.

        Must be called after :meth:`load_data` and before
        :meth:`configure_variables`.  Operates on the working descriptor
        matrix in-place.  State remains ``DATA_IMPORTED``.

        Parameters
        ----------
        constant_threshold : float
            CV threshold below which a descriptor is considered near-constant.
            Default 0.01.
        correlation_threshold : float
            Pearson correlation threshold above which a redundant descriptor
            is removed.  Default 0.90.

        Raises
        ------
        WorkflowError
            If state < ``DATA_IMPORTED``.
        """
        self._require_state(ProjectState.DATA_IMPORTED, "filter_descriptors")
        assert self._X_work is not None
        self._X_work = remove_near_zero_variance(self._X_work, threshold=constant_threshold)
        self._X_work = remove_high_correlation(self._X_work, threshold=correlation_threshold)
        self._config["constant_threshold"] = constant_threshold
        self._config["correlation_threshold"] = correlation_threshold

    # -----------------------------------------------------------------------
    # Phase II: Variable selection and data configuration
    # -----------------------------------------------------------------------

    def configure_variables(
        self,
        y_col: str,
        x_cols: list[str] | None = None,
        normalize_x: bool = False,
        normalize_y: bool = False,
        test_size: float = 0.2,
        split_method: str = "random",
    ) -> None:
        """Select response / descriptor columns, normalise, and split data.

        Transitions to :attr:`~ProjectState.DATA_CONFIGURED`.

        Parameters
        ----------
        y_col : str
            Name of the response column.  Must match
            ``dataset.y_series.name``.
        x_cols : list of str or None
            Descriptor columns to include.  ``None`` = use all remaining
            columns from the (filtered) descriptor matrix.
        normalize_x : bool
            Apply z-score normalisation to descriptor columns.  Default False.
        normalize_y : bool
            Apply z-score normalisation to the response.  Default False.
        test_size : float
            Fraction of samples for the test set.  Default 0.2.
        split_method : str
            ``'random'`` or ``'stratified'``.  Default ``'random'``.

        Raises
        ------
        WorkflowError
            If state < ``DATA_IMPORTED``, or if *y_col* / any *x_cols* are
            not found in the dataset, or if *split_method* is unrecognised.
        WorkflowRegressionWarning
            (Warning) If already at ``DATA_CONFIGURED`` or later.
        """
        self._require_state(ProjectState.DATA_IMPORTED, "configure_variables")
        self._maybe_backward(ProjectState.DATA_CONFIGURED)

        assert self.dataset is not None
        assert self._X_work is not None

        # Validate y_col
        if y_col != self.dataset.y_series.name:
            raise WorkflowError(
                f"y_col '{y_col}' does not match the dataset response column "
                f"'{self.dataset.y_series.name}'."
            )

        y = self.dataset.y_series

        # Select X columns
        if x_cols is not None:
            missing = [c for c in x_cols if c not in self._X_work.columns]
            if missing:
                raise WorkflowError(
                    f"Descriptor columns not found in (filtered) dataset: {missing}"
                )
            X = self._X_work[x_cols].copy()
        else:
            X = self._X_work.copy()

        # Optional normalisation
        if normalize_x:
            sc_x = StandardScaler()
            X = sc_x.fit(X).transform(X)
        if normalize_y:
            sc_y = StandardScaler()
            y_df = pd.DataFrame({"y": y})
            y = sc_y.fit(y_df).transform(y_df)["y"]

        # Train/test split
        if split_method == "random":
            X_tr, X_te, y_tr, y_te = random_split(
                X, y, test_size=test_size, random_seed=self.random_seed
            )
        elif split_method == "stratified":
            X_tr, X_te, y_tr, y_te = stratified_split(X, y, test_size=test_size)
        else:
            raise WorkflowError(
                f"Unknown split_method '{split_method}'. "
                "Use 'random' or 'stratified'."
            )

        self.descriptor_names = list(X.columns)
        self.X_train = np.asarray(X_tr, dtype=np.float64)
        self.X_test = np.asarray(X_te, dtype=np.float64)
        self.y_train = np.asarray(y_tr, dtype=np.float64)
        self.y_test = np.asarray(y_te, dtype=np.float64)

        self._config.update(
            {
                "y_col": y_col,
                "x_cols": x_cols,
                "normalize_x": normalize_x,
                "normalize_y": normalize_y,
                "test_size": test_size,
                "split_method": split_method,
            }
        )

        self.state = ProjectState.DATA_CONFIGURED
        self._checkpoint()

    # -----------------------------------------------------------------------
    # Phase III: Model building
    # -----------------------------------------------------------------------

    def _pre_build(self) -> None:
        """Pre-build state check and regression-warning for MODELS_EVALUATED."""
        self._require_state(ProjectState.DATA_CONFIGURED, "build model")
        # Only warn if we're at MODELS_EVALUATED (losing validation results)
        self._maybe_backward(ProjectState.MODELS_EVALUATED)

    def _post_build(self, model: BaseModel) -> None:
        """Store fitted model and advance state to MODELS_BUILT."""
        self.result_set.add(model.get_results())
        self._models.append(model)
        self.state = ProjectState.MODELS_BUILT
        self._checkpoint()

    def build_ridge(self, alpha: float = 1.0, n_jobs: int = 1) -> None:
        """Fit a Ridge regression model on the configured training data.

        Parameters
        ----------
        alpha : float
            L2 regularisation strength.  Default 1.0.
        n_jobs : int
            Parallel workers for LOO CV.  Default 1.
        """
        self._pre_build()
        assert self.X_train is not None and self.y_train is not None
        model = RidgeModel(alpha=alpha, random_seed=self.random_seed, n_jobs=n_jobs)
        model.fit(self.X_train, self.y_train, X_test=self.X_test, y_test=self.y_test)
        self._post_build(model)

    def build_lasso(
        self,
        alpha: float = 0.1,
        max_iter: int = 10000,
        n_jobs: int = 1,
    ) -> None:
        """Fit a Lasso regression model.

        Parameters
        ----------
        alpha : float
            L1 regularisation strength.  Default 0.1.
        max_iter : int
            Maximum coordinate descent iterations.  Default 10000.
        n_jobs : int
            Parallel workers for LOO CV.  Default 1.
        """
        self._pre_build()
        assert self.X_train is not None and self.y_train is not None
        model = LassoModel(
            alpha=alpha, max_iter=max_iter, random_seed=self.random_seed, n_jobs=n_jobs
        )
        model.fit(self.X_train, self.y_train, X_test=self.X_test, y_test=self.y_test)
        self._post_build(model)

    def build_svr(
        self,
        C: float = 1.0,
        gamma: str | float = "scale",
        kernel: str = "rbf",
        n_jobs: int = 1,
    ) -> None:
        """Fit a Support Vector Regression model.

        Parameters
        ----------
        C : float
            Regularisation parameter.  Default 1.0.
        gamma : str or float
            Kernel coefficient.  Default ``'scale'``.
        kernel : str
            Kernel type.  Default ``'rbf'``.
        n_jobs : int
            Parallel workers for LOO CV.  Default 1.
        """
        self._pre_build()
        assert self.X_train is not None and self.y_train is not None
        model = SVRModel(
            C=C, gamma=gamma, kernel=kernel, random_seed=self.random_seed, n_jobs=n_jobs
        )
        model.fit(self.X_train, self.y_train, X_test=self.X_test, y_test=self.y_test)
        self._post_build(model)

    def build_random_forest(
        self,
        n_estimators: int = 100,
        max_features: str | float = "sqrt",
        n_jobs: int = 1,
    ) -> None:
        """Fit a Random Forest regression model.

        Parameters
        ----------
        n_estimators : int
            Number of trees.  Default 100.
        max_features : str or float
            Feature fraction / strategy per split.  Default ``'sqrt'``.
        n_jobs : int
            Parallel workers for both RF fitting and LOO CV.  Default 1.
        """
        self._pre_build()
        assert self.X_train is not None and self.y_train is not None
        model = RandomForestModel(
            n_estimators=n_estimators,
            max_features=max_features,
            random_seed=self.random_seed,
            n_jobs=n_jobs,
        )
        model.fit(self.X_train, self.y_train, X_test=self.X_test, y_test=self.y_test)
        self._post_build(model)

    def build_gradient_boosting(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        n_jobs: int = 1,
    ) -> None:
        """Fit a Gradient Boosting regression model.

        Parameters
        ----------
        n_estimators : int
            Number of boosting stages.  Default 100.
        learning_rate : float
            Shrinkage applied to each tree.  Default 0.1.
        max_depth : int
            Maximum depth of individual regression trees.  Default 3.
        n_jobs : int
            Parallel workers for LOO CV.  Default 1.
        """
        self._pre_build()
        assert self.X_train is not None and self.y_train is not None
        model = GradientBoostingModel(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_seed=self.random_seed,
            n_jobs=n_jobs,
        )
        model.fit(self.X_train, self.y_train, X_test=self.X_test, y_test=self.y_test)
        self._post_build(model)

    def build_ga_mlr(
        self,
        cut_d: float | None = None,
        min_variables: int = 1,
        max_variables: int = 5,
        population_size: int = 100,
        max_generations: int = 100,
        mutation_rate: float = 0.01,
        keep_best: int = 10,
        fitness_function: str = "q2_loo",
        quik_delta: float | None = 0.05,
        inter_cluster_mutation_ratio: float = 0.7,
        tournament_size: int = 5,
        n_workers: int = 0,
        exhaustive_max_vars: int = 3,
    ) -> None:
        """Run GA-MLR feature selection and fit the best models.

        First runs cophenetic clustering on ``X_train`` to produce a
        *cluster_map*, optionally enumerates all subsets of size 1–3, then
        runs the genetic algorithm for subset sizes ``min_variables`` through
        ``max_variables``.  All retained models are added to
        :attr:`result_set`.

        Parameters
        ----------
        cut_d : float or None
            Dendrogram cut distance.  ``None`` = automatic selection by
            maximising normalised Shannon entropy.
        min_variables : int
            Minimum subset size for GA.  Default 1.
        max_variables : int
            Maximum subset size (≤ n_train/5 and ≤ n_clusters).  Default 5.
        population_size : int
            Chromosomes per generation.  Default 100.
        max_generations : int
            Number of evolutionary generations.  Default 100.
        mutation_rate : float
            Per-gene mutation probability.  Default 0.01.
        keep_best : int
            Elite models retained per subset size.  Default 10.
        fitness_function : str
            One of ``'q2_loo'``, ``'r2_adj'``, ``'lof'``, ``'rmse_cv'``.
        quik_delta : float or None
            QUIK rule threshold.  ``None`` disables the rule.  Default 0.05.
        inter_cluster_mutation_ratio : float
            Fraction of mutations that swap clusters.  Default 0.7.
        tournament_size : int
            Tournament selection size.  Default 5.
        n_workers : int
            Worker processes for parallel fitness evaluation.  0 = sequential.
        exhaustive_max_vars : int
            Maximum subset size for exhaustive enumeration prior to GA.
            0 = skip exhaustive step.  Default 3.
        """
        self._pre_build()
        assert self.X_train is not None and self.y_train is not None

        # Cophenetic clustering
        cluster_result = cophenetic_cluster(self.X_train, cut_d=cut_d)
        cluster_map = cluster_result.cluster_map

        # Optional exhaustive enumeration for small subsets (1–3)
        if exhaustive_max_vars > 0:
            small_models = enumerate_subsets(
                self.X_train,
                self.y_train,
                cluster_map=cluster_map,
                max_size=min(exhaustive_max_vars, 3),
                X_test=self.X_test,
                y_test=self.y_test,
                quik_delta=quik_delta,
            )
            for m in small_models:
                self.result_set.add(m.get_results())
                self._models.append(m)

        # GA-MLR
        ga_models = run_ga_mlr(
            self.X_train,
            self.y_train,
            cluster_map=cluster_map,
            X_test=self.X_test,
            y_test=self.y_test,
            min_variables=min_variables,
            max_variables=max_variables,
            population_size=population_size,
            max_generations=max_generations,
            mutation_rate=mutation_rate,
            keep_best=keep_best,
            fitness_function=fitness_function,
            quik_delta=quik_delta,
            inter_cluster_mutation_ratio=inter_cluster_mutation_ratio,
            tournament_size=tournament_size,
            random_seed=self.random_seed,
            n_workers=n_workers,
        )
        for m in ga_models:
            self.result_set.add(m.get_results())
            self._models.append(m)

        self.state = ProjectState.MODELS_BUILT
        self._checkpoint()

    # -----------------------------------------------------------------------
    # Phase IV: Validation
    # -----------------------------------------------------------------------

    def run_lmo_validation(
        self,
        model_indices: list[int],
        holdout_fraction: float = 0.30,
        n_iterations: int = 100,
        n_workers: int = 0,
    ) -> None:
        """Run Leave-Many-Out cross-validation on selected models.

        Populates :attr:`~qsarify.results.model_result.ModelResult.lmo_results`
        on each requested model and transitions to
        :attr:`~ProjectState.MODELS_EVALUATED`.

        Parameters
        ----------
        model_indices : list of int
            Indices into :attr:`result_set` identifying models to validate.
        holdout_fraction : float
            Fraction of training samples held out per iteration.  Default 0.30.
        n_iterations : int
            Number of LMO iterations.  Default 100.
        n_workers : int
            Worker processes for parallel LMO.  0 = sequential.

        Raises
        ------
        WorkflowError
            If state < ``MODELS_BUILT``.
        """
        self._require_state(ProjectState.MODELS_BUILT, "run_lmo_validation")
        assert self.X_train is not None and self.y_train is not None

        # procedures.run_lmo uses 1 for sequential; map 0 → 1
        _lmo_workers = n_workers if n_workers >= 1 else 1

        for idx in model_indices:
            result = self.result_set[idx]
            estimator = _make_estimator(result.model_type, dict(result.hyperparameters))
            X_val = _X_for_validation(self.X_train, result)

            lmo_result = run_lmo(
                estimator,
                X_val,
                self.y_train,
                holdout_fraction=holdout_fraction,
                n_iterations=n_iterations,
                random_seed=self.random_seed,
                n_workers=_lmo_workers,
            )

            result.lmo_results = {
                "holdout_fraction": lmo_result.holdout_fraction,
                "n_iterations": lmo_result.n_iterations,
                "mean_q2": lmo_result.mean_q2,
                "std_q2": lmo_result.std_q2,
                "mean_r2": lmo_result.mean_r2,
                "std_r2": lmo_result.std_r2,
                "mean_rmse": lmo_result.mean_rmse,
                "std_rmse": lmo_result.std_rmse,
            }

        self.state = ProjectState.MODELS_EVALUATED
        self._checkpoint()

    def run_y_scrambling(
        self,
        model_indices: list[int],
        n_iterations: int = 100,
        n_workers: int = 0,
    ) -> None:
        """Run Y-scrambling (response permutation) validation.

        Populates
        :attr:`~qsarify.results.model_result.ModelResult.y_scrambling_results`
        on each requested model and transitions to
        :attr:`~ProjectState.MODELS_EVALUATED`.

        Parameters
        ----------
        model_indices : list of int
            Indices into :attr:`result_set` identifying models to validate.
        n_iterations : int
            Number of scrambling iterations.  Default 100.
        n_workers : int
            Worker processes.  0 = sequential.

        Raises
        ------
        WorkflowError
            If state < ``MODELS_BUILT``.
        """
        self._require_state(ProjectState.MODELS_BUILT, "run_y_scrambling")
        assert self.X_train is not None and self.y_train is not None

        # procedures.run_y_scrambling uses 1 for sequential; map 0 → 1
        _scram_workers = n_workers if n_workers >= 1 else 1

        for idx in model_indices:
            result = self.result_set[idx]
            estimator = _make_estimator(result.model_type, dict(result.hyperparameters))
            X_val = _X_for_validation(self.X_train, result)

            r2_original = result.r_squared if result.r_squared is not None else 0.0
            q2_original = result.q_squared_loo if result.q_squared_loo is not None else 0.0

            y_scram_result = run_y_scrambling(
                estimator,
                X_val,
                self.y_train,
                r2_original=r2_original,
                q2_original=q2_original,
                n_iterations=n_iterations,
                random_seed=self.random_seed,
                n_workers=_scram_workers,
            )

            result.y_scrambling_results = {
                "n_iterations": y_scram_result.n_iterations,
                "mean_r2_scrambled": y_scram_result.mean_r2_scrambled,
                "std_r2_scrambled": y_scram_result.std_r2_scrambled,
                "mean_q2_scrambled": y_scram_result.mean_q2_scrambled,
                "std_q2_scrambled": y_scram_result.std_q2_scrambled,
                "r2_original": y_scram_result.r2_original,
                "q2_original": y_scram_result.q2_original,
            }

        self.state = ProjectState.MODELS_EVALUATED
        self._checkpoint()

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Persist the project to a single SQLite3 file.

        Available in any state.  Arrays are stored as BLOBs via
        ``ndarray.tobytes()`` with dtype and shape metadata; no pickle is used.

        Parameters
        ----------
        path : Path or str
            Destination file path.  Created or overwritten.

        Raises
        ------
        PersistenceError
            If writing fails.
        """
        from qsarify.io.persistence import save_project

        save_project(self, Path(path))

    @classmethod
    def load(cls, path: Path | str) -> "QSARProject":
        """Load a project from a SQLite3 save file.

        Parameters
        ----------
        path : Path or str
            Path to the ``.sqlite3`` project file.

        Returns
        -------
        QSARProject
            Fully restored project at the saved state.

        Raises
        ------
        PersistenceError
            If the file does not exist or the schema is incompatible.
        """
        from qsarify.io.persistence import load_project

        return load_project(Path(path))
