import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from skopt.space import Integer, Real

from .base import BaseModel, OptimizationConfig
from .ga import GeneticConfig


# Enhanced wrapper classes with genetic feature selection
class RidgeWrapper(BaseModel):
    """Ridge Regression wrapper with genetic feature selection."""

    def _get_param_space(self) -> list:
        return [
            Real(0.01, 100.0, name="alpha", prior="log-uniform"),
            Real(1e-6, 1e-2, name="tol", prior="log-uniform"),
        ]

    def _create_model(self, **params) -> Ridge:
        return Ridge(**params, random_state=self.optimization_config.random_state)

    def get_default_params(self) -> Dict[str, Any]:
        return {"alpha": 1.0, "tol": 1e-4}


class LassoWrapper(BaseModel):
    """Lasso Regression wrapper with genetic feature selection."""

    def _get_param_space(self) -> list:
        return [
            Real(0.01, 10.0, name="alpha", prior="log-uniform"),
            Real(1e-6, 1e-2, name="tol", prior="log-uniform"),
            Integer(100, 2000, name="max_iter"),
        ]

    def _create_model(self, **params) -> Lasso:
        return Lasso(**params, random_state=self.optimization_config.random_state)

    def get_default_params(self) -> Dict[str, Any]:
        return {"alpha": 1.0, "tol": 1e-4, "max_iter": 1000}


class SVRWrapper(BaseModel):
    """Support Vector Regression wrapper with genetic feature selection."""

    def _get_param_space(self) -> list:
        return [
            Real(0.01, 100.0, name="C", prior="log-uniform"),
            Real(0.001, 10.0, name="epsilon", prior="log-uniform"),
            Real(0.001, 10.0, name="gamma", prior="log-uniform"),
        ]

    def _create_model(self, **params) -> SVR:
        return SVR(kernel="rbf", **params)

    def get_default_params(self) -> Dict[str, Any]:
        return {"C": 1.0, "epsilon": 0.1, "gamma": "scale"}


class RandomForestWrapper(BaseModel):
    """Random Forest Regression wrapper with genetic feature selection."""

    def _get_param_space(self) -> list:
        return [
            Integer(50, 500, name="n_estimators"),
            Integer(1, 20, name="max_depth"),
            Integer(2, 20, name="min_samples_split"),
            Integer(1, 10, name="min_samples_leaf"),
            Real(0.1, 1.0, name="max_features"),
        ]

    def _create_model(self, **params) -> RandomForestRegressor:
        return RandomForestRegressor(
            **params,
            random_state=self.optimization_config.random_state,
            n_jobs=1  # Parallel processing handled at higher level
        )

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 1.0,
        }


class GradientBoostingWrapper(BaseModel):
    """Gradient Boosting Regression wrapper with genetic feature selection."""

    def _get_param_space(self) -> list:
        return [
            Integer(50, 300, name="n_estimators"),
            Real(0.01, 0.3, name="learning_rate", prior="log-uniform"),
            Integer(1, 15, name="max_depth"),
            Integer(2, 20, name="min_samples_split"),
            Integer(1, 10, name="min_samples_leaf"),
            Real(0.1, 1.0, name="subsample"),
        ]

    def _create_model(self, **params) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            **params, random_state=self.optimization_config.random_state
        )

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "subsample": 1.0,
        }


class LinearRegressionWrapper(BaseModel):
    """Multiple Linear Regression wrapper optimized for genetic feature selection."""

    def _get_param_space(self) -> list:
        # MLR has no hyperparameters to optimize
        return []

    def _create_model(self, **params) -> LinearRegression:
        return LinearRegression(**params)

    def get_default_params(self) -> Dict[str, Any]:
        return {"fit_intercept": True}

    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """MLR has no hyperparameters to optimize."""
        return self.get_default_params()


# Enhanced model registry
MODEL_REGISTRY = {
    "ridge": RidgeWrapper,
    "lasso": LassoWrapper,
    "svr": SVRWrapper,
    "random_forest": RandomForestWrapper,
    "gradient_boosting": GradientBoostingWrapper,
    "linear_regression": LinearRegressionWrapper,
    "mlr": LinearRegressionWrapper,  # Alias for multiple linear regression
}


def create_model(
    model_type: str,
    optimization_config: Optional[OptimizationConfig] = None,
    genetic_config: Optional[GeneticConfig] = None,
) -> BaseModel:
    """
    Enhanced factory function to create model instances with genetic feature selection.

    Args:
        model_type: Type of model to create
        optimization_config: Configuration for hyperparameter optimization
        genetic_config: Configuration for genetic algorithm feature selection

    Returns:
        Initialized model instance
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[model_type](optimization_config, genetic_config)


class ModelComparison:
    """
    Utility class for comparing multiple models with comprehensive statistics.
    """

    def __init__(self, models: Dict[str, BaseModel]):
        """
        Initialize model comparison.

        Args:
            models: Dictionary mapping model names to model instances
        """
        self.models = models
        self.results = {}
        self.statistics = {}

    def fit_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        optimize: bool = True,
        select_features: bool = None,
    ) -> Dict[str, Any]:
        """
        Fit all models and calculate comprehensive statistics.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            optimize: Whether to perform hyperparameter optimization
            select_features: Whether to perform feature selection

        Returns:
            Dictionary containing results for all models
        """
        print("Fitting and evaluating all models...")

        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training {name}...")
            print(f"{'='*50}")

            try:
                # Fit model
                start_time = datetime.datetime.now()
                model.fit(
                    X_train, y_train, optimize=optimize, select_features=select_features
                )
                fit_time = datetime.datetime.now() - start_time

                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Calculate basic metrics
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

                # Calculate comprehensive statistics
                comprehensive_stats = model.calculate_comprehensive_statistics(
                    X_train, X_test, y_train, y_test
                )

                self.results[name] = {
                    "model": model,
                    "fit_time": fit_time.total_seconds(),
                    "train_r2": train_r2,
                    "test_r2": test_r2,
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse,
                    "selected_features": getattr(model, "selected_features_", None),
                    "n_selected_features": len(getattr(model, "selected_features_", [])),
                    "best_params": getattr(model, "best_params_", None),
                    "feature_selection_results": getattr(
                        model, "feature_selection_results_", None
                    ),
                }

                self.statistics[name] = comprehensive_stats

                print(f"✓ {name} completed successfully")
                print(f"  Fit time: {fit_time.total_seconds():.2f}s")
                print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
                print(f"  Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
                if model.selected_features_:
                    print(f"  Selected features: {len(model.selected_features_)}")

            except Exception as e:
                print(f"✗ {name} failed: {e}")
                self.results[name] = {"error": str(e)}
                self.statistics[name] = {}

        return self.results

    def get_summary_table(self) -> pd.DataFrame:
        """
        Generate summary table of all model results.

        Returns:
            DataFrame containing summary statistics for all models
        """
        summary_data = []

        for name, result in self.results.items():
            if "error" not in result:
                row = {
                    "Model": name,
                    "Train_R2": result.get("train_r2", np.nan),
                    "Test_R2": result.get("test_r2", np.nan),
                    "Train_RMSE": result.get("train_rmse", np.nan),
                    "Test_RMSE": result.get("test_rmse", np.nan),
                    "N_Features": result.get("n_selected_features", 0),
                    "Fit_Time_s": result.get("fit_time", np.nan),
                }

                # Add comprehensive statistics if available
                stats = self.statistics.get(name, {})
                if stats:
                    row.update(
                        {
                            "Q2_LOO": stats.get("q_squared_loo", np.nan),
                            "R2_Adj": stats.get("r_squared_adj", np.nan),
                            "F_Statistic": stats.get("f_statistic", np.nan),
                            "LOF": stats.get("lof", np.nan),
                            "CCC": stats.get("ccc", np.nan),
                            "Q2_F1": stats.get("q_squared_f1", np.nan),
                            "Q2_F2": stats.get("q_squared_f2", np.nan),
                            "Q2_F3": stats.get("q_squared_f3", np.nan),
                        }
                    )

                summary_data.append(row)
            else:
                summary_data.append({"Model": name, "Error": result["error"]})

        return pd.DataFrame(summary_data)

    def get_best_.model(self, metric: str = "test_r2") -> tuple[str, BaseModel]:
        """
        Get the best performing model based on specified metric.

        Args:
            metric: Metric to use for comparison

        Returns:
            Tuple of (model_name, model_instance)
        """
        valid_results = {k: v for k, v in self.results.items() if "error" not in v}

        if not valid_results:
            raise ValueError("No valid models found")

        best_name = max(
            valid_results.keys(), key=lambda x: valid_results[x].get(metric, -np.inf)
        )
        return best_name, valid_results[best_name]["model"]


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data

    print("Generating sample data...")
    X, y = make_regression(
        n_samples=200, n_features=20, n_informative=10, noise=0.1, random_state=42
    )

    # Convert to DataFrames
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Configuration for optimization and genetic algorithm
    opt_config = OptimizationConfig(n_calls=20, n_initial_points=5, cv_folds=3)

    genetic_config = GeneticConfig(
        max_vars=8,
        population_size=30,
        max_generations=50,
        mutation_rate=0.15,
        keep_best=5,
        fitness_functions=["Q2loo", "R2Adj", "LOF"],
        clustering_distance=2.5,
        verbose=True,
    )

    # Create models for comparison
    models = {
        "MLR_with_GA": create_model("mlr", opt_config, genetic_config),
        "Ridge_with_GA": create_model("ridge", opt_config, genetic_config),
        "Lasso_with_GA": create_model("lasso", opt_config, genetic_config),
        "MLR_no_GA": create_model("mlr", opt_config, None),  # Control without GA
    }

    # Run model comparison
    comparison = ModelComparison(models)
    results = comparison.fit_all(
        X_train, y_train, X_test, y_test, optimize=True, select_features=None
    )  # Use model's genetic_config

    # Display results
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)

    summary_table = comparison.get_summary_table()
    print(summary_table.to_string(index=False))

    # Get best model
    try:
        best_name, best_model = comparison.get_best_model("test_r2")
        print(f"\nBest model: {best_name}")
        print(f"Test R²: {results[best_name]['test_r2']:.4f}")
        print(f"Selected features: {results[best_name]['selected_features']}")

        # Show comprehensive statistics for best model
        best_stats = comparison.statistics[best_name]
        if best_stats:
            print(f"\nComprehensive Statistics for {best_name}:")
            for metric, value in best_stats.items():
                if not isinstance(value, (list, tuple, dict)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

    except Exception as e:
        print(f"Error getting best model: {e}")

    print("\n" + "=" * 80)
    print("FEATURE SELECTION ANALYSIS")
    print("=" * 80)

    for name, result in results.items():
        if "error" not in result and result.get("feature_selection_results"):
            print(f"\n{name} Feature Selection Results:")
            fs_results = result["feature_selection_results"]
            for n_vars, details in fs_results.items():
                print(
                    f"  {n_vars} variables: {details['fitness']} -> {details['features']}"
                )

    print("\nTesting completed successfully!")