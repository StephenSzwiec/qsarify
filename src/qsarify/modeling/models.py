"""Scikit-learn FullModel wrappers for QSARify.

Each wrapper trains on the *full* provided descriptor matrix, computes the
complete suite of QSARINS regression metrics via
:func:`~qsarify.modeling.metrics.compute_model_result`, and stores them in a
:class:`~qsarify.results.model_result.ModelResult`.

Hyperparameter tuning uses :func:`scipy.optimize.differential_evolution`
against the chosen fitness function (from
:mod:`qsarify.modeling.fitness`), ensuring all model types can be compared
on equal footing via :func:`head_to_head`.

Class hierarchy
---------------
::

    BaseModel (ABC)          — qsarify.modeling.base
    └── FullModel            — this module
        ├── RidgeModel
        ├── LassoModel
        ├── SVRModel
        ├── RandomForestModel
        └── GradientBoostingModel
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR

from qsarify.modeling.base import BaseModel
from qsarify.modeling.fitness import fitness_score
from qsarify.modeling.metrics import compute_model_result
from qsarify.results.model_result import ModelResult
from qsarify.utils import statistics as stat

__all__ = [
    "FullModel",
    "RidgeModel",
    "LassoModel",
    "SVRModel",
    "RandomForestModel",
    "GradientBoostingModel",
    "head_to_head",
]


# ---------------------------------------------------------------------------
# FullModel base class
# ---------------------------------------------------------------------------


class FullModel(BaseModel):
    """Base class for full-descriptor-set sklearn-backed models.

    Hyperparameter-tuning configuration is set at construction time (or via
    :meth:`configure_tuning`) so that the :meth:`fit` signature matches the
    :class:`BaseModel` ABC contract.

    Parameters
    ----------
    random_seed : int or None
        Random seed for reproducible results.
    n_jobs : int
        Number of parallel jobs for LOO CV.  -1 = all available cores.
    fitness_function : str
        Fitness function used for hyperparameter tuning.  One of
        ``'q2_loo'``, ``'r2_adj'``, ``'rmse_cv'``, ``'lof'``.
    population_size : int or None
        DE population budget.  ``None`` disables tuning (use constructor
        defaults).
    max_generations : int or None
        DE generation limit.
    """

    def __init__(
        self,
        random_seed: int | None = None,
        n_jobs: int = 1,
        fitness_function: str = "q2_loo",
        population_size: int | None = None,
        max_generations: int | None = None,
    ) -> None:
        self._random_seed = random_seed
        self._n_jobs = n_jobs
        self._fitness_function = fitness_function
        self._population_size = population_size
        self._max_generations = max_generations
        self._estimator: Any = None
        self._result: ModelResult | None = None

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict using the fitted sklearn estimator."""
        if self._estimator is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before predict")
        return np.asarray(
            self._estimator.predict(np.asarray(X, dtype=np.float64)),
            dtype=np.float64,
        )

    def get_results(self) -> ModelResult:
        """Return the :class:`~qsarify.results.model_result.ModelResult`."""
        if self._result is None:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fit before get_results"
            )
        return self._result

    def _fit_with_estimator(
        self,
        estimator: Any,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_test: NDArray[np.float64] | None,
        y_test: NDArray[np.float64] | None,
        model_type: str,
        hyperparameters: dict[str, object],
        coef_std_errors: NDArray[np.float64] | None = None,
        coef_ci: NDArray[np.float64] | None = None,
        coef_p_values: NDArray[np.float64] | None = None,
    ) -> ModelResult:
        """Fit *estimator*, compute LOO, and assemble a ModelResult.

        The estimator is fitted **once** here.  Callers that need
        pre-fit access to coefficients (Ridge, Lasso) should extract them
        from the fitted estimator *after* this call, not before.

        Parameters
        ----------
        estimator : sklearn-compatible estimator
            Not yet fitted.
        X, y : ndarray
            Training data.
        X_test, y_test : ndarray or None
            External test data.
        model_type : str
            Model type tag for the result.
        hyperparameters : dict
            Hyperparameters to record.
        coef_std_errors, coef_ci, coef_p_values : ndarray or None
            Pre-computed coefficient statistics (for linear models only).

        Returns
        -------
        ModelResult
        """
        estimator.fit(X, y)
        self._estimator = estimator

        loo = stat.sklearn_loo(estimator, X, y, n_jobs=self._n_jobs)

        y_pred_test: NDArray[np.float64] | None = None
        y_test_arr: NDArray[np.float64] | None = None
        if X_test is not None and y_test is not None:
            y_pred_test = np.asarray(estimator.predict(X_test), dtype=np.float64)
            y_test_arr = np.asarray(y_test, dtype=np.float64)

        return compute_model_result(
            X=X,
            y=y,
            y_pred=loo.y_pred,
            y_pred_loo=loo.y_pred_loo,
            model_type=model_type,
            hyperparameters=hyperparameters,
            y_test=y_test_arr,
            y_pred_test=y_pred_test,
            coef_std_errors=coef_std_errors,
            coef_confidence_intervals=coef_ci,
            coef_p_values=coef_p_values,
        )

    def _de_optimize(
        self,
        bounds: list[tuple[float, float]],
        objective: Any,
    ) -> NDArray[np.float64]:
        """Run differential evolution and return the best parameter vector.

        Parameters
        ----------
        bounds : list of (low, high) tuples
            Search bounds for each parameter.
        objective : callable
            Minimisation target (negated fitness).

        Returns
        -------
        ndarray
            Best parameter vector found.
        """
        pop_size = self._population_size or 100
        n_pop = max(4, pop_size // len(bounds))
        n_gen = self._max_generations or 50

        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=n_gen,
            popsize=n_pop,
            seed=self._random_seed,
            tol=1e-6,
            polish=True,
        )
        return np.asarray(de_result.x, dtype=np.float64)


# ---------------------------------------------------------------------------
# Ridge
# ---------------------------------------------------------------------------


class RidgeModel(FullModel):
    """Ridge regression wrapper.

    Parameters
    ----------
    alpha : float
        L2 regularisation strength.  Default 1.0.
        Ignored when *population_size* is set (DE tuning).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_test: NDArray[np.float64] | None = None,
        y_test: NDArray[np.float64] | None = None,
    ) -> "RidgeModel":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        X_test_arr = (
            np.asarray(X_test, dtype=np.float64) if X_test is not None else None
        )
        y_test_arr = (
            np.asarray(y_test, dtype=np.float64) if y_test is not None else None
        )
        p = X_arr.shape[1]

        best_alpha = self.alpha
        if self._population_size is not None:

            def _obj(params: NDArray[np.float64]) -> float:
                a = float(10.0 ** params[0])
                loo = stat.sklearn_loo(
                    Ridge(alpha=a, fit_intercept=True),
                    X_arr,
                    y_arr,
                    n_jobs=self._n_jobs,
                )
                return -fitness_score(loo, y_arr, p, self._fitness_function)

            best_params = self._de_optimize([(-6.0, 4.0)], _obj)
            best_alpha = float(10.0 ** best_params[0])

        estimator = Ridge(alpha=best_alpha, fit_intercept=True)

        # Fit once via _fit_with_estimator, then extract coef stats
        # from the fitted estimator — no double-fit
        self._result = self._fit_with_estimator(
            estimator,
            X_arr,
            y_arr,
            X_test_arr,
            y_test_arr,
            model_type="ridge",
            hyperparameters={"alpha": best_alpha},
        )

        # Compute coefficient statistics post-fit
        coef_arr = np.asarray(self._estimator.coef_, dtype=np.float64)
        X_centered = X_arr - X_arr.mean(axis=0)
        y_centered = y_arr - float(self._estimator.intercept_)
        std_err, ci, p_vals = stat.regularized_coef_stats(
            X_centered, y_centered, coef_arr, best_alpha, "ridge"
        )
        self._result.coef_std_errors = std_err
        self._result.coef_confidence_intervals = ci
        self._result.coef_p_values = p_vals

        return self


# ---------------------------------------------------------------------------
# Lasso
# ---------------------------------------------------------------------------


class LassoModel(FullModel):
    """Lasso regression wrapper.

    Parameters
    ----------
    alpha : float
        L1 regularisation strength.  Default 0.1.
    max_iter : int
        Maximum iterations for coordinate descent.  Default 10000.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        max_iter: int = 10000,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_test: NDArray[np.float64] | None = None,
        y_test: NDArray[np.float64] | None = None,
    ) -> "LassoModel":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        X_test_arr = (
            np.asarray(X_test, dtype=np.float64) if X_test is not None else None
        )
        y_test_arr = (
            np.asarray(y_test, dtype=np.float64) if y_test is not None else None
        )
        p = X_arr.shape[1]
        max_iter = self.max_iter

        best_alpha = self.alpha
        if self._population_size is not None:

            def _obj(params: NDArray[np.float64]) -> float:
                a = float(10.0 ** params[0])
                loo = stat.sklearn_loo(
                    Lasso(alpha=a, max_iter=max_iter, fit_intercept=True),
                    X_arr,
                    y_arr,
                    n_jobs=self._n_jobs,
                )
                return -fitness_score(loo, y_arr, p, self._fitness_function)

            best_params = self._de_optimize([(-6.0, 4.0)], _obj)
            best_alpha = float(10.0 ** best_params[0])

        estimator = Lasso(alpha=best_alpha, max_iter=self.max_iter, fit_intercept=True)

        self._result = self._fit_with_estimator(
            estimator,
            X_arr,
            y_arr,
            X_test_arr,
            y_test_arr,
            model_type="lasso",
            hyperparameters={"alpha": best_alpha},
        )

        # Coefficient statistics post-fit
        coef_arr = np.asarray(self._estimator.coef_, dtype=np.float64)
        X_centered = X_arr - X_arr.mean(axis=0)
        y_centered = y_arr - float(self._estimator.intercept_)
        std_err, ci, p_vals = stat.regularized_coef_stats(
            X_centered, y_centered, coef_arr, best_alpha, "lasso"
        )
        self._result.coef_std_errors = std_err
        self._result.coef_confidence_intervals = ci
        self._result.coef_p_values = p_vals

        return self


# ---------------------------------------------------------------------------
# SVR
# ---------------------------------------------------------------------------


class SVRModel(FullModel):
    """Support Vector Regression wrapper.

    Parameters
    ----------
    C : float
        Regularisation parameter.  Default 1.0.
    gamma : str or float
        Kernel coefficient.  Default ``'scale'``.
    kernel : str
        Kernel type.  Default ``'rbf'``.
    """

    def __init__(
        self,
        C: float = 1.0,
        gamma: str | float = "scale",
        kernel: str = "rbf",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.C = C
        self.gamma = gamma
        self.kernel = kernel

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_test: NDArray[np.float64] | None = None,
        y_test: NDArray[np.float64] | None = None,
    ) -> "SVRModel":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        X_test_arr = (
            np.asarray(X_test, dtype=np.float64) if X_test is not None else None
        )
        y_test_arr = (
            np.asarray(y_test, dtype=np.float64) if y_test is not None else None
        )
        p = X_arr.shape[1]
        kernel = self.kernel

        best_C = self.C
        best_gamma: str | float = self.gamma
        if self._population_size is not None:

            def _obj(params: NDArray[np.float64]) -> float:
                c = float(10.0 ** params[0])
                g = float(10.0 ** params[1])
                loo = stat.sklearn_loo(
                    SVR(C=c, gamma=g, kernel=kernel),
                    X_arr,
                    y_arr,
                    n_jobs=self._n_jobs,
                )
                return -fitness_score(loo, y_arr, p, self._fitness_function)

            best_params = self._de_optimize([(-3.0, 3.0), (-4.0, 2.0)], _obj)
            best_C = float(10.0 ** best_params[0])
            best_gamma = float(10.0 ** best_params[1])

        estimator = SVR(C=best_C, gamma=best_gamma, kernel=self.kernel)
        self._result = self._fit_with_estimator(
            estimator,
            X_arr,
            y_arr,
            X_test_arr,
            y_test_arr,
            model_type="svr",
            hyperparameters={
                "C": best_C,
                "gamma": best_gamma,
                "kernel": self.kernel,
            },
        )
        return self


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------


class RandomForestModel(FullModel):
    """Random Forest Regression wrapper.

    Parameters
    ----------
    n_estimators : int
        Number of trees.  Default 100.
    max_features : str or float
        Features to consider for best split.  Default ``'sqrt'``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: str | float = "sqrt",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_test: NDArray[np.float64] | None = None,
        y_test: NDArray[np.float64] | None = None,
    ) -> "RandomForestModel":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        X_test_arr = (
            np.asarray(X_test, dtype=np.float64) if X_test is not None else None
        )
        y_test_arr = (
            np.asarray(y_test, dtype=np.float64) if y_test is not None else None
        )
        p = X_arr.shape[1]
        n_jobs = self._n_jobs
        rng = self._random_seed

        best_n_est = self.n_estimators
        best_max_feat: float = (
            self.max_features if isinstance(self.max_features, float) else 0.5
        )
        if self._population_size is not None:

            def _obj(params: NDArray[np.float64]) -> float:
                ne = max(10, int(round(params[0])))
                mf = float(np.clip(params[1], 0.01, 1.0))
                loo = stat.sklearn_loo(
                    RandomForestRegressor(
                        n_estimators=ne,
                        max_features=mf,
                        random_state=rng,
                        n_jobs=n_jobs,
                    ),
                    X_arr,
                    y_arr,
                    n_jobs=n_jobs,
                )
                return -fitness_score(loo, y_arr, p, self._fitness_function)

            best_params = self._de_optimize(
                [(50.0, 500.0), (0.1, 1.0)],
                _obj,
            )
            best_n_est = max(10, int(round(best_params[0])))
            best_max_feat = float(np.clip(best_params[1], 0.01, 1.0))

        estimator = RandomForestRegressor(
            n_estimators=best_n_est,
            max_features=best_max_feat,
            random_state=self._random_seed,
            n_jobs=self._n_jobs,
        )
        self._result = self._fit_with_estimator(
            estimator,
            X_arr,
            y_arr,
            X_test_arr,
            y_test_arr,
            model_type="rf",
            hyperparameters={
                "n_estimators": best_n_est,
                "max_features": best_max_feat,
            },
        )
        return self


# ---------------------------------------------------------------------------
# Gradient Boosting
# ---------------------------------------------------------------------------


class GradientBoostingModel(FullModel):
    """Gradient Boosting Regression wrapper.

    Parameters
    ----------
    n_estimators : int
        Number of boosting stages.  Default 100.
    learning_rate : float
        Shrinkage.  Default 0.1.
    max_depth : int
        Maximum depth of individual trees.  Default 3.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_test: NDArray[np.float64] | None = None,
        y_test: NDArray[np.float64] | None = None,
    ) -> "GradientBoostingModel":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        X_test_arr = (
            np.asarray(X_test, dtype=np.float64) if X_test is not None else None
        )
        y_test_arr = (
            np.asarray(y_test, dtype=np.float64) if y_test is not None else None
        )
        p = X_arr.shape[1]
        rng = self._random_seed

        best_lr = self.learning_rate
        best_n_est = self.n_estimators
        best_depth = self.max_depth
        if self._population_size is not None:

            def _obj(params: NDArray[np.float64]) -> float:
                lr = float(np.clip(params[0], 0.001, 1.0))
                ne = max(10, int(round(10.0 ** params[1])))
                depth = max(1, int(round(params[2])))
                loo = stat.sklearn_loo(
                    GradientBoostingRegressor(
                        n_estimators=ne,
                        learning_rate=lr,
                        max_depth=depth,
                        random_state=rng,
                    ),
                    X_arr,
                    y_arr,
                    n_jobs=self._n_jobs,
                )
                return -fitness_score(loo, y_arr, p, self._fitness_function)

            best_params = self._de_optimize(
                [(0.01, 0.5), (1.7, 2.7), (2.0, 8.0)],
                _obj,
            )
            best_lr = float(np.clip(best_params[0], 0.001, 1.0))
            best_n_est = max(10, int(round(10.0 ** best_params[1])))
            best_depth = max(1, int(round(best_params[2])))

        estimator = GradientBoostingRegressor(
            n_estimators=best_n_est,
            learning_rate=best_lr,
            max_depth=best_depth,
            random_state=self._random_seed,
        )
        self._result = self._fit_with_estimator(
            estimator,
            X_arr,
            y_arr,
            X_test_arr,
            y_test_arr,
            model_type="gbr",
            hyperparameters={
                "n_estimators": best_n_est,
                "learning_rate": best_lr,
                "max_depth": best_depth,
            },
        )
        return self


# ---------------------------------------------------------------------------
# head_to_head — unified model comparison
# ---------------------------------------------------------------------------


def head_to_head(
    X_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    y_train: NDArray[np.float64],
    y_test: NDArray[np.float64],
    fitness_function: str = "q2_loo",
    population_size: int = 100,
    max_generations: int = 50,
    n_jobs: int = 1,
    random_seed: int | None = None,
) -> Any:
    """Run all FullModel types and return a ResultSet.

    Each model is hyperparameter-optimised against *fitness_function* using
    differential evolution, so results are directly comparable.

    Parameters
    ----------
    X_train : ndarray of shape (n_train, p)
        Training descriptor matrix.
    X_test : ndarray of shape (n_test, p)
        External test descriptor matrix.
    y_train : ndarray of shape (n_train,)
        Training response vector.
    y_test : ndarray of shape (n_test,)
        Test response vector.
    fitness_function : str, optional
        Default ``'q2_loo'``.
    population_size : int, optional
        DE population budget.  Default 100.
    max_generations : int, optional
        DE generation limit.  Default 50.
    n_jobs : int, optional
        Parallel workers for LOO CV.  Default 1.
    random_seed : int or None, optional
        Seed for reproducible DE runs.

    Returns
    -------
    ResultSet
    """
    from qsarify.results.result_set import ResultSet

    common: dict[str, Any] = {
        "random_seed": random_seed,
        "n_jobs": n_jobs,
        "fitness_function": fitness_function,
        "population_size": population_size,
        "max_generations": max_generations,
    }

    models: list[FullModel] = [
        RidgeModel(**common),
        LassoModel(**common),
        SVRModel(**common),
        RandomForestModel(**common),
        GradientBoostingModel(**common),
    ]

    rs = ResultSet()
    for m in models:
        m.fit(X_train, y_train, X_test=X_test, y_test=y_test)
        rs.add(m.get_results())
    return rs
