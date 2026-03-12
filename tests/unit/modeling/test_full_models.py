"""Unit tests for qsarify.modeling.models (FullModel wrappers)."""

from __future__ import annotations

import numpy as np
import pytest

from qsarify.modeling.models import (
    GradientBoostingModel,
    LassoModel,
    RandomForestModel,
    RidgeModel,
    SVRModel,
    head_to_head,
)
from qsarify.results.model_result import ModelResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N_TRAIN, P = 60, 5

X_TRAIN = RNG.standard_normal((N_TRAIN, P))
Y_TRAIN = (
    X_TRAIN[:, 0] * 2.0 - X_TRAIN[:, 2] * 1.5 + 0.5 + RNG.standard_normal(N_TRAIN) * 0.3
)

X_TEST = RNG.standard_normal((15, P))
Y_TEST = X_TEST[:, 0] * 2.0 - X_TEST[:, 2] * 1.5 + 0.5 + RNG.standard_normal(15) * 0.3

ALL_MODEL_CLASSES = [
    RidgeModel,
    LassoModel,
    SVRModel,
    RandomForestModel,
    GradientBoostingModel,
]
LINEAR_CLASSES = [RidgeModel, LassoModel]
NONLINEAR_CLASSES = [SVRModel, RandomForestModel, GradientBoostingModel]


# ---------------------------------------------------------------------------
# Fit / predict interface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_MODEL_CLASSES)
def test_fit_returns_self(cls: type) -> None:
    model = cls()
    result = model.fit(X_TRAIN, Y_TRAIN)
    assert result is model


@pytest.mark.parametrize("cls", ALL_MODEL_CLASSES)
def test_predict_shape(cls: type) -> None:
    model = cls()
    model.fit(X_TRAIN, Y_TRAIN)
    preds = model.predict(X_TRAIN)
    assert preds.shape == (N_TRAIN,)


@pytest.mark.parametrize("cls", ALL_MODEL_CLASSES)
def test_predict_test_shape(cls: type) -> None:
    model = cls()
    model.fit(X_TRAIN, Y_TRAIN)
    preds = model.predict(X_TEST)
    assert preds.shape == (15,)


@pytest.mark.parametrize("cls", ALL_MODEL_CLASSES)
def test_get_results_before_fit_raises(cls: type) -> None:
    model = cls()
    with pytest.raises(RuntimeError):
        model.get_results()


# ---------------------------------------------------------------------------
# ModelResult completeness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_MODEL_CLASSES)
def test_get_results_returns_model_result(cls: type) -> None:
    model = cls()
    model.fit(X_TRAIN, Y_TRAIN)
    assert isinstance(model.get_results(), ModelResult)


@pytest.mark.parametrize("cls", ALL_MODEL_CLASSES)
def test_training_metrics_populated(cls: type) -> None:
    model = cls()
    model.fit(X_TRAIN, Y_TRAIN)
    r = model.get_results()
    for attr in (
        "r_squared",
        "r_squared_adj",
        "rmse",
        "mae",
        "q_squared_loo",
        "leverage",
        "std_residuals",
        "leverage_threshold",
    ):
        assert getattr(r, attr) is not None, f"{cls.__name__}: {attr} is None"


@pytest.mark.parametrize("cls", ALL_MODEL_CLASSES)
def test_external_metrics_populated_with_test(cls: type) -> None:
    model = cls()
    model.fit(X_TRAIN, Y_TRAIN, X_test=X_TEST, y_test=Y_TEST)
    r = model.get_results()
    for attr in (
        "q_squared_f1",
        "q_squared_f2",
        "q_squared_f3",
        "r_squared_ext",
        "press_ext",
    ):
        assert getattr(r, attr) is not None, (
            f"{cls.__name__}: {attr} is None with test set"
        )


@pytest.mark.parametrize("cls", ALL_MODEL_CLASSES)
def test_external_metrics_none_without_test(cls: type) -> None:
    model = cls()
    model.fit(X_TRAIN, Y_TRAIN)
    r = model.get_results()
    for attr in (
        "q_squared_f1",
        "q_squared_f2",
        "q_squared_f3",
        "r_squared_ext",
        "press_ext",
    ):
        assert getattr(r, attr) is None, (
            f"{cls.__name__}: {attr} should be None without test"
        )


# ---------------------------------------------------------------------------
# Coefficient statistics — linear vs. non-linear
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", LINEAR_CLASSES)
def test_linear_coef_stats_not_none(cls: type) -> None:
    model = cls()
    model.fit(X_TRAIN, Y_TRAIN)
    r = model.get_results()
    assert r.coef_std_errors is not None, f"{cls.__name__}: coef_std_errors is None"
    assert r.coef_confidence_intervals is not None
    assert r.coef_p_values is not None


@pytest.mark.parametrize("cls", LINEAR_CLASSES)
def test_linear_coef_stats_shapes(cls: type) -> None:
    model = cls()
    model.fit(X_TRAIN, Y_TRAIN)
    r = model.get_results()
    assert r.coef_std_errors is not None
    assert r.coef_std_errors.shape == (P,)
    assert r.coef_confidence_intervals is not None
    assert r.coef_confidence_intervals.shape == (P, 2)
    assert r.coef_p_values is not None
    assert r.coef_p_values.shape == (P,)


@pytest.mark.parametrize("cls", NONLINEAR_CLASSES)
def test_nonlinear_coef_stats_are_none(cls: type) -> None:
    model = cls()
    model.fit(X_TRAIN, Y_TRAIN)
    r = model.get_results()
    assert r.coef_std_errors is None, f"{cls.__name__}: coef_std_errors should be None"
    assert r.coef_confidence_intervals is None
    assert r.coef_p_values is None


# ---------------------------------------------------------------------------
# Applicability domain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_MODEL_CLASSES)
def test_leverage_shape(cls: type) -> None:
    model = cls()
    model.fit(X_TRAIN, Y_TRAIN)
    r = model.get_results()
    assert r.leverage is not None
    assert r.leverage.shape == (N_TRAIN,)


@pytest.mark.parametrize("cls", ALL_MODEL_CLASSES)
def test_leverage_non_negative(cls: type) -> None:
    model = cls()
    model.fit(X_TRAIN, Y_TRAIN)
    r = model.get_results()
    assert r.leverage is not None
    assert (r.leverage >= -1e-10).all()


@pytest.mark.parametrize("cls", ALL_MODEL_CLASSES)
def test_leverage_threshold_formula(cls: type) -> None:
    """h* = 3p / n for all model types."""
    model = cls()
    model.fit(X_TRAIN, Y_TRAIN)
    r = model.get_results()
    assert r.leverage_threshold is not None
    expected = 3.0 * P / N_TRAIN
    assert abs(r.leverage_threshold - expected) < 1e-10


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_ridge_model_type() -> None:
    model = RidgeModel()
    model.fit(X_TRAIN, Y_TRAIN)
    assert model.get_results().model_type == "ridge"


def test_lasso_model_type() -> None:
    model = LassoModel()
    model.fit(X_TRAIN, Y_TRAIN)
    assert model.get_results().model_type == "lasso"


def test_svr_model_type() -> None:
    model = SVRModel()
    model.fit(X_TRAIN, Y_TRAIN)
    assert model.get_results().model_type == "svr"


def test_rf_model_type() -> None:
    model = RandomForestModel()
    model.fit(X_TRAIN, Y_TRAIN)
    assert model.get_results().model_type == "rf"


def test_gbr_model_type() -> None:
    model = GradientBoostingModel()
    model.fit(X_TRAIN, Y_TRAIN)
    assert model.get_results().model_type == "gbr"


@pytest.mark.parametrize("cls", ALL_MODEL_CLASSES)
def test_n_train_set_correctly(cls: type) -> None:
    model = cls()
    model.fit(X_TRAIN, Y_TRAIN)
    assert model.get_results().n_train == N_TRAIN


@pytest.mark.parametrize("cls", ALL_MODEL_CLASSES)
def test_n_features_set_correctly(cls: type) -> None:
    model = cls()
    model.fit(X_TRAIN, Y_TRAIN)
    assert model.get_results().n_features == P


# ---------------------------------------------------------------------------
# Hyperparameter storage
# ---------------------------------------------------------------------------


def test_ridge_stores_alpha_in_hyperparameters() -> None:
    """Without DE, the user-supplied alpha must be stored as-is."""
    model = RidgeModel(alpha=0.5)
    model.fit(X_TRAIN, Y_TRAIN)
    assert model.get_results().hyperparameters.get("alpha") == 0.5


def test_lasso_stores_alpha_in_hyperparameters() -> None:
    """Without DE, the user-supplied alpha must be stored as-is."""
    model = LassoModel(alpha=0.1)
    model.fit(X_TRAIN, Y_TRAIN)
    assert model.get_results().hyperparameters.get("alpha") == 0.1


def test_ridge_de_optimization_stores_float_alpha() -> None:
    """With DE, the optimized alpha (float) must be stored in hyperparameters."""
    model = RidgeModel(random_seed=42)
    model.fit(
        X_TRAIN,
        Y_TRAIN,
        fitness_function="q2_loo",
        population_size=8,
        max_generations=5,
    )
    hp = model.get_results().hyperparameters
    assert "alpha" in hp
    assert isinstance(hp["alpha"], float)


def test_svr_de_optimization_stores_c_gamma() -> None:
    """With DE, SVR must store optimized C and gamma in hyperparameters."""
    model = SVRModel(random_seed=42)
    model.fit(
        X_TRAIN,
        Y_TRAIN,
        fitness_function="q2_loo",
        population_size=4,
        max_generations=3,
    )
    hp = model.get_results().hyperparameters
    assert "C" in hp and "gamma" in hp
    assert isinstance(hp["C"], float) and isinstance(hp["gamma"], float)


# ---------------------------------------------------------------------------
# head_to_head
# ---------------------------------------------------------------------------


def test_head_to_head_returns_result_set() -> None:
    from qsarify.results.result_set import ResultSet

    rs = head_to_head(
        X_TRAIN,
        X_TEST,
        Y_TRAIN,
        Y_TEST,
        fitness_function="q2_loo",
        population_size=4,
        max_generations=3,
    )
    assert isinstance(rs, ResultSet)
    assert len(rs) == 5  # Ridge, Lasso, SVR, RF, GBM


def test_head_to_head_all_model_types_present() -> None:
    rs = head_to_head(
        X_TRAIN,
        X_TEST,
        Y_TRAIN,
        Y_TEST,
        fitness_function="q2_loo",
        population_size=4,
        max_generations=3,
    )
    types = {r.model_type for r in rs}
    assert types == {"ridge", "lasso", "svr", "rf", "gbr"}


def test_head_to_head_external_metrics_populated() -> None:
    rs = head_to_head(
        X_TRAIN,
        X_TEST,
        Y_TRAIN,
        Y_TEST,
        fitness_function="q2_loo",
        population_size=4,
        max_generations=3,
    )
    for r in rs:
        assert r.q_squared_f1 is not None, f"{r.model_type}: q_squared_f1 is None"
        assert r.r_squared_ext is not None
