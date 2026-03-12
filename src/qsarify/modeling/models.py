"""Scikit-learn FullModel wrappers for QSARify.

Each wrapper trains on the *full* provided descriptor matrix, computes the
complete suite of QSARINS regression metrics, and stores them in a
:class:`~qsarify.results.model_result.ModelResult`.

When *population_size* is passed to :meth:`fit`, hyperparameters are tuned
via :func:`scipy.optimize.differential_evolution` against the chosen fitness
function — ensuring all model types can be compared on equal footing via
:func:`head_to_head`.

Class hierarchy
---------------
::

    BaseModel (ABC)
    └── FullModel
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
# Fitness computation helper
# ---------------------------------------------------------------------------

_VALID_FITNESS = frozenset({"q2_loo", "r2_adj", "rmse_cv", "lof"})


def _fitness_score(
    loo: stat.LOOResult,
    y: NDArray[np.float64],
    n_features: int,
    fitness_function: str,
) -> float:
    """Return a *higher-is-better* fitness score.

    Parameters
    ----------
    loo : LOOResult
        Result of LOO cross-validation.
    y : ndarray of shape (n,)
        Training response vector.
    n_features : int
        Number of descriptor columns *p*.
    fitness_function : str
        One of ``'q2_loo'``, ``'r2_adj'``, ``'rmse_cv'``, ``'lof'``.

    Returns
    -------
    float
        Higher values are always better; ``rmse_cv`` and ``lof`` are negated.
    """
    if fitness_function == "q2_loo":
        return loo.q2_loo
    if fitness_function == "r2_adj":
        return stat.r_squared_adj(y, loo.y_pred, n_features)
    n = len(y)
    if fitness_function == "rmse_cv":
        return -float(np.sqrt(loo.press / n))
    if fitness_function == "lof":
        return -stat.lof(y, loo.y_pred, n_features)
    raise ValueError(
        f"Unknown fitness_function {fitness_function!r}. "
        f"Valid options: {sorted(_VALID_FITNESS)}"
    )


# ---------------------------------------------------------------------------
# FullModel base class
# ---------------------------------------------------------------------------


class FullModel(BaseModel):
    """Base class for full-descriptor-set sklearn-backed models.

    Parameters
    ----------
    random_seed : int or None
        Random seed for reproducible results.
    n_jobs : int
        Number of parallel jobs for LOO CV.  -1 = all available cores.
    """

    def __init__(
        self,
        random_seed: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        self._random_seed = random_seed
        self._n_jobs = n_jobs
        self._estimator: Any = None
        self._result: ModelResult | None = None

    def fit(  # type: ignore[override]
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_test: NDArray[np.float64] | None = None,
        y_test: NDArray[np.float64] | None = None,
        fitness_function: str = "q2_loo",
        population_size: int | None = None,
        max_generations: int | None = None,
    ) -> "FullModel":
        raise NotImplementedError

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict using the fitted sklearn estimator."""
        if self._estimator is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before predict")
        return np.asarray(
            self._estimator.predict(np.asarray(X, dtype=np.float64)), dtype=np.float64
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
        """Fit *estimator*, compute LOO and assemble ModelResult."""
        estimator.fit(X, y)
        self._estimator = estimator

        loo = stat.sklearn_loo(estimator, X, y, n_jobs=self._n_jobs)
        y_pred_test: NDArray[np.float64] | None = None
        if X_test is not None:
            y_pred_test = np.asarray(estimator.predict(X_test), dtype=np.float64)

        return BaseModel._build_result(
            X, y,
            y_pred=loo.y_pred,
            y_pred_loo=loo.y_pred_loo,
            model_type=model_type,
            hyperparameters=hyperparameters,
            X_test=X_test,
            y_test=y_test,
            y_pred_test=y_pred_test,
            coef_std_errors=coef_std_errors,
            coef_ci=coef_ci,
            coef_p_values=coef_p_values,
        )


# ---------------------------------------------------------------------------
# Ridge
# ---------------------------------------------------------------------------


class RidgeModel(FullModel):
    """Ridge regression wrapper.

    Parameters
    ----------
    alpha : float
        L2 regularisation strength.  Default 1.0.
        Ignored when *population_size* is passed to :meth:`fit`.
    random_seed : int or None
        Not used by Ridge, kept for API consistency.
    n_jobs : int
        Parallel jobs for LOO CV.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        random_seed: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(random_seed=random_seed, n_jobs=n_jobs)
        self.alpha = alpha

    def fit(  # type: ignore[override]
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_test: NDArray[np.float64] | None = None,
        y_test: NDArray[np.float64] | None = None,
        fitness_function: str = "q2_loo",
        population_size: int | None = None,
        max_generations: int | None = None,
    ) -> "RidgeModel":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        X_test_arr = np.asarray(X_test, dtype=np.float64) if X_test is not None else None
        y_test_arr = np.asarray(y_test, dtype=np.float64) if y_test is not None else None
        p = X_arr.shape[1]

        best_alpha = self.alpha
        if population_size is not None:
            # DE search over log10(alpha)
            bounds = [(-6.0, 4.0)]
            n_pop = max(4, population_size)
            n_gen = max_generations if max_generations is not None else 50

            def _obj(params: NDArray[np.float64]) -> float:
                a = float(10.0 ** params[0])
                loo = stat.sklearn_loo(Ridge(alpha=a, fit_intercept=True), X_arr, y_arr,
                                       n_jobs=self._n_jobs)
                return -_fitness_score(loo, y_arr, p, fitness_function)

            de_result = differential_evolution(
                _obj, bounds, maxiter=n_gen, popsize=n_pop,  # type: ignore[arg-type]
                seed=self._random_seed, tol=1e-6, polish=True,
            )
            best_alpha = float(10.0 ** de_result.x[0])

        estimator = Ridge(alpha=best_alpha, fit_intercept=True)
        X_centered = X_arr - X_arr.mean(axis=0)
        coef_arr: NDArray[np.float64] | None = None
        std_err: NDArray[np.float64] | None = None
        ci: NDArray[np.float64] | None = None
        p_vals: NDArray[np.float64] | None = None

        # Fit once to get coef for stats
        estimator.fit(X_arr, y_arr)
        coef_arr = np.asarray(estimator.coef_, dtype=np.float64)
        std_err, ci, p_vals = stat.regularized_coef_stats(
            X_centered, y_arr - float(estimator.intercept_), coef_arr, best_alpha, "ridge"
        )

        self._result = self._fit_with_estimator(
            estimator, X_arr, y_arr, X_test_arr, y_test_arr,
            model_type="ridge",
            hyperparameters={"alpha": best_alpha},
            coef_std_errors=std_err,
            coef_ci=ci,
            coef_p_values=p_vals,
        )
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
        Ignored when *population_size* is passed to :meth:`fit`.
    max_iter : int
        Maximum iterations for coordinate descent.  Default 10000.
    random_seed : int or None
        Random seed.
    n_jobs : int
        Parallel jobs for LOO CV.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        max_iter: int = 10000,
        random_seed: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(random_seed=random_seed, n_jobs=n_jobs)
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(  # type: ignore[override]
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_test: NDArray[np.float64] | None = None,
        y_test: NDArray[np.float64] | None = None,
        fitness_function: str = "q2_loo",
        population_size: int | None = None,
        max_generations: int | None = None,
    ) -> "LassoModel":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        X_test_arr = np.asarray(X_test, dtype=np.float64) if X_test is not None else None
        y_test_arr = np.asarray(y_test, dtype=np.float64) if y_test is not None else None
        p = X_arr.shape[1]
        max_iter = self.max_iter

        best_alpha = self.alpha
        if population_size is not None:
            bounds = [(-6.0, 4.0)]
            n_pop = max(4, population_size)
            n_gen = max_generations if max_generations is not None else 50

            def _obj(params: NDArray[np.float64]) -> float:
                a = float(10.0 ** params[0])
                loo = stat.sklearn_loo(
                    Lasso(alpha=a, max_iter=max_iter, fit_intercept=True),
                    X_arr, y_arr, n_jobs=self._n_jobs,
                )
                return -_fitness_score(loo, y_arr, p, fitness_function)

            de_result = differential_evolution(
                _obj, bounds, maxiter=n_gen, popsize=n_pop,  # type: ignore[arg-type]
                seed=self._random_seed, tol=1e-6, polish=True,
            )
            best_alpha = float(10.0 ** de_result.x[0])

        estimator = Lasso(alpha=best_alpha, max_iter=self.max_iter, fit_intercept=True)
        estimator.fit(X_arr, y_arr)
        coef_arr: NDArray[np.float64] = np.asarray(estimator.coef_, dtype=np.float64)
        X_centered = X_arr - X_arr.mean(axis=0)
        std_err, ci, p_vals = stat.regularized_coef_stats(
            X_centered, y_arr - float(estimator.intercept_), coef_arr, best_alpha, "lasso"
        )

        self._result = self._fit_with_estimator(
            estimator, X_arr, y_arr, X_test_arr, y_test_arr,
            model_type="lasso",
            hyperparameters={"alpha": best_alpha},
            coef_std_errors=std_err,
            coef_ci=ci,
            coef_p_values=p_vals,
        )
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
        Ignored when *population_size* is passed to :meth:`fit`.
    gamma : str or float
        Kernel coefficient.  Default ``'scale'``.
    kernel : str
        Kernel type.  Default ``'rbf'``.
    random_seed : int or None
        Not used by SVR.
    n_jobs : int
        Parallel jobs for LOO CV.
    """

    def __init__(
        self,
        C: float = 1.0,
        gamma: str | float = "scale",
        kernel: str = "rbf",
        random_seed: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(random_seed=random_seed, n_jobs=n_jobs)
        self.C = C
        self.gamma = gamma
        self.kernel = kernel

    def fit(  # type: ignore[override]
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_test: NDArray[np.float64] | None = None,
        y_test: NDArray[np.float64] | None = None,
        fitness_function: str = "q2_loo",
        population_size: int | None = None,
        max_generations: int | None = None,
    ) -> "SVRModel":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        X_test_arr = np.asarray(X_test, dtype=np.float64) if X_test is not None else None
        y_test_arr = np.asarray(y_test, dtype=np.float64) if y_test is not None else None
        p = X_arr.shape[1]
        kernel = self.kernel

        best_C = self.C
        best_gamma: str | float = self.gamma
        if population_size is not None:
            # DE search: log10(C) in [-3, 3], log10(gamma) in [-4, 2]
            bounds = [(-3.0, 3.0), (-4.0, 2.0)]
            n_pop = max(4, population_size // len(bounds))
            n_gen = max_generations if max_generations is not None else 50

            def _obj(params: NDArray[np.float64]) -> float:
                c = float(10.0 ** params[0])
                g = float(10.0 ** params[1])
                loo = stat.sklearn_loo(
                    SVR(C=c, gamma=g, kernel=kernel),
                    X_arr, y_arr, n_jobs=self._n_jobs,
                )
                return -_fitness_score(loo, y_arr, p, fitness_function)

            de_result = differential_evolution(
                _obj, bounds, maxiter=n_gen, popsize=n_pop,  # type: ignore[arg-type]
                seed=self._random_seed, tol=1e-6, polish=True,
            )
            best_C = float(10.0 ** de_result.x[0])
            best_gamma = float(10.0 ** de_result.x[1])

        estimator = SVR(C=best_C, gamma=best_gamma, kernel=self.kernel)
        self._result = self._fit_with_estimator(
            estimator, X_arr, y_arr, X_test_arr, y_test_arr,
            model_type="svr",
            hyperparameters={"C": best_C, "gamma": best_gamma, "kernel": self.kernel},
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
        Ignored when *population_size* is passed to :meth:`fit`.
    max_features : str or float
        Features to consider for best split.  Default ``'sqrt'``.
    random_seed : int or None
        Random state for reproducibility.
    n_jobs : int
        Parallel workers for tree fitting **and** LOO CV.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: str | float = "sqrt",
        random_seed: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(random_seed=random_seed, n_jobs=n_jobs)
        self.n_estimators = n_estimators
        self.max_features = max_features

    def fit(  # type: ignore[override]
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_test: NDArray[np.float64] | None = None,
        y_test: NDArray[np.float64] | None = None,
        fitness_function: str = "q2_loo",
        population_size: int | None = None,
        max_generations: int | None = None,
    ) -> "RandomForestModel":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        X_test_arr = np.asarray(X_test, dtype=np.float64) if X_test is not None else None
        y_test_arr = np.asarray(y_test, dtype=np.float64) if y_test is not None else None
        p = X_arr.shape[1]
        n_jobs = self._n_jobs
        rng = self._random_seed

        best_n_est = self.n_estimators
        best_max_feat: float = (
            self.max_features if isinstance(self.max_features, float) else 0.5
        )
        if population_size is not None:
            # DE search: n_estimators in [50, 500], max_features in [0.1, 1.0]
            bounds = [(50.0, 500.0), (0.1, 1.0)]
            n_pop = max(4, population_size // len(bounds))
            n_gen = max_generations if max_generations is not None else 50

            def _obj(params: NDArray[np.float64]) -> float:
                ne = max(10, int(round(params[0])))
                mf = float(np.clip(params[1], 0.01, 1.0))
                loo = stat.sklearn_loo(
                    RandomForestRegressor(
                        n_estimators=ne, max_features=mf,
                        random_state=rng, n_jobs=n_jobs,
                    ),
                    X_arr, y_arr, n_jobs=n_jobs,
                )
                return -_fitness_score(loo, y_arr, p, fitness_function)

            de_result = differential_evolution(
                _obj, bounds, maxiter=n_gen, popsize=n_pop,  # type: ignore[arg-type]
                seed=self._random_seed, tol=1e-6, polish=True,
            )
            best_n_est = max(10, int(round(de_result.x[0])))
            best_max_feat = float(np.clip(de_result.x[1], 0.01, 1.0))

        estimator = RandomForestRegressor(
            n_estimators=best_n_est,
            max_features=best_max_feat,
            random_state=self._random_seed,
            n_jobs=self._n_jobs,
        )
        self._result = self._fit_with_estimator(
            estimator, X_arr, y_arr, X_test_arr, y_test_arr,
            model_type="rf",
            hyperparameters={"n_estimators": best_n_est, "max_features": best_max_feat},
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
        Ignored when *population_size* is passed to :meth:`fit`.
    learning_rate : float
        Shrinkage.  Default 0.1.
    max_depth : int
        Maximum depth of individual trees.  Default 3.
    random_seed : int or None
        Random state for reproducibility.
    n_jobs : int
        Parallel workers for LOO CV (GBR itself is not parallelisable).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_seed: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        super().__init__(random_seed=random_seed, n_jobs=n_jobs)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(  # type: ignore[override]
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        X_test: NDArray[np.float64] | None = None,
        y_test: NDArray[np.float64] | None = None,
        fitness_function: str = "q2_loo",
        population_size: int | None = None,
        max_generations: int | None = None,
    ) -> "GradientBoostingModel":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        X_test_arr = np.asarray(X_test, dtype=np.float64) if X_test is not None else None
        y_test_arr = np.asarray(y_test, dtype=np.float64) if y_test is not None else None
        p = X_arr.shape[1]
        rng = self._random_seed

        best_lr = self.learning_rate
        best_n_est = self.n_estimators
        best_depth = self.max_depth
        if population_size is not None:
            # DE search: lr in [0.01, 0.5], log10(n_estimators) in [1.7, 2.7], depth in [2, 8]
            bounds = [(0.01, 0.5), (1.7, 2.7), (2.0, 8.0)]
            n_pop = max(4, population_size // len(bounds))
            n_gen = max_generations if max_generations is not None else 50

            def _obj(params: NDArray[np.float64]) -> float:
                lr = float(np.clip(params[0], 0.001, 1.0))
                ne = max(10, int(round(10.0 ** params[1])))
                depth = max(1, int(round(params[2])))
                loo = stat.sklearn_loo(
                    GradientBoostingRegressor(
                        n_estimators=ne, learning_rate=lr,
                        max_depth=depth, random_state=rng,
                    ),
                    X_arr, y_arr, n_jobs=self._n_jobs,
                )
                return -_fitness_score(loo, y_arr, p, fitness_function)

            de_result = differential_evolution(
                _obj, bounds, maxiter=n_gen, popsize=n_pop,  # type: ignore[arg-type]
                seed=self._random_seed, tol=1e-6, polish=True,
            )
            best_lr = float(np.clip(de_result.x[0], 0.001, 1.0))
            best_n_est = max(10, int(round(10.0 ** de_result.x[1])))
            best_depth = max(1, int(round(de_result.x[2])))

        estimator = GradientBoostingRegressor(
            n_estimators=best_n_est,
            learning_rate=best_lr,
            max_depth=best_depth,
            random_state=self._random_seed,
        )
        self._result = self._fit_with_estimator(
            estimator, X_arr, y_arr, X_test_arr, y_test_arr,
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
) -> "Any":
    """Run all FullModel types and return a :class:`~qsarify.results.result_set.ResultSet`.

    Each model is hyperparameter-optimised against *fitness_function* using
    :func:`scipy.optimize.differential_evolution`, so results are directly
    comparable.

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
        One of ``'q2_loo'``, ``'r2_adj'``, ``'rmse_cv'``, ``'lof'``.
        Default ``'q2_loo'``.
    population_size : int, optional
        DE population budget passed to each model.  Default 100.
    max_generations : int, optional
        DE generation budget.  Default 50.
    n_jobs : int, optional
        Parallel workers for LOO CV.  Default 1.
    random_seed : int or None, optional
        Seed for reproducible DE runs.

    Returns
    -------
    ResultSet
        Contains one :class:`~qsarify.results.model_result.ModelResult` per
        model type.
    """
    from qsarify.results.result_set import ResultSet

    fit_kwargs: dict[str, object] = {
        "X_test": X_test,
        "y_test": y_test,
        "fitness_function": fitness_function,
        "population_size": population_size,
        "max_generations": max_generations,
    }

    models: list[FullModel] = [
        RidgeModel(random_seed=random_seed, n_jobs=n_jobs),
        LassoModel(random_seed=random_seed, n_jobs=n_jobs),
        SVRModel(random_seed=random_seed, n_jobs=n_jobs),
        RandomForestModel(random_seed=random_seed, n_jobs=n_jobs),
        GradientBoostingModel(random_seed=random_seed, n_jobs=n_jobs),
    ]

    rs = ResultSet()
    for m in models:
        m.fit(X_train, y_train, **fit_kwargs)  # type: ignore[arg-type]
        rs.add(m.get_results())
    return rs
