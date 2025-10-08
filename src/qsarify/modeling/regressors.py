from datetime import datetime
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


class UnifiedModelComparison:
    """
    Unified head-to-head model comparison interface.
    
    Generates all model types for a given X_train, X_test, Y_train, Y_test tuple
    and configuration, allowing comprehensive comparison of MLR (with GA feature selection)
    vs non-MLR models (with gradient-based hyperparameter optimization).
    """

    def __init__(self, 
                 optimization_config: Optional[OptimizationConfig] = None,
                 genetic_config: Optional[GeneticConfig] = None,
                 include_models: Optional[List[str]] = None):
        """
        Initialize unified model comparison.

        Args:
            optimization_config: Configuration for hyperparameter optimization
            genetic_config: Configuration for genetic algorithm feature selection
            include_models: List of model types to include (None = all models)
        """
        self.optimization_config = optimization_config or OptimizationConfig()
        self.genetic_config = genetic_config
        
        # Define which models to include
        available_models = list(MODEL_REGISTRY.keys())
        self.include_models = include_models or available_models
        
        self.models = {}
        self.results = {}
        self.statistics = {}
        self.preprocessing_info = None
        
    def setup_models(self, use_preprocessing: bool = True) -> None:
        """
        Setup all model instances with appropriate configurations.
        
        Args:
            use_preprocessing: Whether to apply preprocessing pipeline
        """
        for model_name in self.include_models:
            if model_name not in MODEL_REGISTRY:
                continue
                
            # Configure optimization based on model type
            if model_name in ["linear_regression", "mlr"]:
                # MLR models use GA feature selection
                model_config = OptimizationConfig(
                    use_gradient_optimization=False,
                    **self.optimization_config.__dict__
                )
                self.models[model_name] = create_model(
                    model_name, model_config, self.genetic_config
                )
            else:
                # Non-MLR models use full feature set with gradient optimization
                model_config = OptimizationConfig(
                    use_gradient_optimization=True,
                    **self.optimization_config.__dict__
                )
                self.models[model_name] = create_model(
                    model_name, model_config, None  # No genetic config for non-MLR
                )

    def run_comprehensive_comparison(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        preprocessing_config: Optional[Any] = None,
        optimize: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive head-to-head model comparison.
        
        Takes raw data, applies preprocessing, and trains all models with
        appropriate configurations (GA-MLR vs full-feature with gradient optimization).
        
        Args:
            X: Raw feature matrix
            y: Raw target variable  
            preprocessing_config: Configuration for preprocessing pipeline
            optimize: Whether to perform hyperparameter optimization
            
        Returns:
            Dictionary containing comprehensive results for all models
        """
        from ..preprocessing.preprocessing import comprehensive_preprocess, PreprocessingConfig
        
        # Apply preprocessing pipeline
        if preprocessing_config is None:
            preprocessing_config = PreprocessingConfig()
            
        X_train, X_test, y_train, y_test, scaler, preprocessing_info = comprehensive_preprocess(
            X, y, preprocessing_config
        )
        
        self.preprocessing_info = preprocessing_info
        
        # Setup models with appropriate configurations
        self.setup_models()
        
        print("="*80)
        print("UNIFIED HEAD-TO-HEAD MODEL COMPARISON")
        print("="*80)
        print(f"Dataset: {len(X)} samples, {len(X.columns)} original features")
        print(f"After preprocessing: {preprocessing_info['n_final_features']} features")
        print(f"Train/Test split: {len(X_train)}/{len(X_test)} samples")
        print("="*80)

        # Train all models
        return self._fit_all_models(X_train, X_test, y_train, y_test, optimize)
    
    def _fit_all_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame, 
        y_train: pd.Series,
        y_test: pd.Series,
        optimize: bool = True
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

    def get_best_model(self, metric: str = "test_r2") -> tuple[str, BaseModel]:
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