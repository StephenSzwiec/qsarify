import abc
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler

from ..utils.statistics import calculate_all
from .ga import GeneticConfig, GeneticFeatureSelector

from skopt import gp_minimize
from skopt.space import Integer, Real


@dataclass
class OptimizationConfig:
    """Configuration for Bayesian optimization"""
    n_calls: int = 50
    n_initial_points: int = 10
    cv_folds: int = 3
    val_split: float = 0.2
    n_jobs: int = -1
    random_state: int = 42


class BaseModel(abc.ABC):
    """Enhanced abstract base class with genetic feature selection."""

    def __init__(self, optimization_config: Optional[OptimizationConfig] = None,
                 genetic_config: Optional[GeneticConfig] = None):
        self.optimization_config = optimization_config or OptimizationConfig()
        self.genetic_config = genetic_config
        self.model = None
        self.scaler = StandardScaler()
        self.best_params_ = None
        self.best_score_ = None
        self.optimization_history_ = []
        self.selected_features_ = None
        self.feature_selection_results_ = None
        self.is_fitted = False

        # Initialize genetic feature selector if config provided
        self.genetic_selector = None
        if self.genetic_config:
            self.genetic_selector = GeneticFeatureSelector(self.genetic_config)

    @abc.abstractmethod
    def _get_param_space(self) -> list:
        pass

    @abc.abstractmethod
    def _create_model(self, **params) -> Any:
        pass

    @abc.abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        pass

    def select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Perform genetic algorithm feature selection.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Selected feature subset
        """
        if not self.genetic_selector:
            warnings.warn("No genetic configuration provided. Skipping feature selection.")
            return X

        print("Performing genetic algorithm feature selection...")
        self.feature_selection_results_ = self.genetic_selector.fit(X, y)

        # Choose best feature set (you can modify this logic)
        if self.feature_selection_results_:
            # Select the model with best primary fitness function
            best_model = max(
                self.feature_selection_results_.values(),
                key=lambda x: x['fitness'].get(self.genetic_config.fitness_functions[0], -np.inf)
            )
            self.selected_features_ = best_model['features']

            print(f"Selected {len(self.selected_features_)} features: {self.selected_features_}")
            print(f"Fitness scores: {best_model['fitness']}")

            return X[self.selected_features_]
        else:
            warnings.warn("Feature selection failed. Using all features.")
            return X

    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Enhanced hyperparameter optimization."""

        param_space = self._get_param_space()
        objective = partial(self._objective_function, X=X, y=y)

        result = gp_minimize(
            func=objective,
            dimensions=param_space,
            n_calls=self.optimization_config.n_calls,
            n_initial_points=self.optimization_config.n_initial_points,
            random_state=self.optimization_config.random_state,
            verbose=False
        )

        self.best_score_ = -result.fun
        self.optimization_history_ = result.func_vals

        best_params = dict(zip([p.name for p in param_space], result.x))
        return best_params

    def _objective_function(self, params: list, X: np.ndarray, y: np.ndarray) -> float:
        """Enhanced objective function with statistical validation."""
        param_dict = dict(zip([p.name for p in self._get_param_space()], params))

        try:
            model = self._create_model(**param_dict)

            # Use Leave-One-Out cross-validation for small datasets, k-fold for larger
            if len(X) < 50:
                loo = LeaveOneOut()
                scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error', n_jobs=1)
            else:
                scores = cross_val_score(
                    model, X, y,
                    cv=self.optimization_config.cv_folds,
                    scoring='neg_mean_squared_error',
                    n_jobs=1
                )

            return -scores.mean()
        except Exception as e:
            warnings.warn(f"Error in objective function: {e}")
            return float('inf')

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series],
            optimize: bool = True, select_features: bool = None) -> 'BaseModel':
        """
        Enhanced fit method with feature selection and comprehensive statistics.

        Args:
            X: Training features
            y: Training targets
            optimize: Whether to perform hyperparameter optimization
            select_features: Whether to perform feature selection (uses genetic_config if None)

        Returns:
            Self for method chaining
        """
        # Convert to appropriate formats
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        # Feature selection
        if select_features is None:
            select_features = self.genetic_config is not None

        if select_features:
            X_selected = self.select_features(X, y)
        else:
            X_selected = X
            self.selected_features_ = list(X.columns)

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_selected),
            columns=X_selected.columns,
            index=X_selected.index
        )

        # Hyperparameter optimization
        if optimize:
            print("Optimizing hyperparameters using Bayesian optimization...")
            self.best_params_ = self._optimize_hyperparameters(X_scaled.values, y.values)
        else:
            self.best_params_ = self.get_default_params()

        # Create and fit final model
        self.model = self._create_model(**self.best_params_)

        # Use parallel processing if supported
        if hasattr(self.model, 'n_jobs'):
            self.model.set_params(n_jobs=self.optimization_config.n_jobs)

        self.model.fit(X_scaled, y)
        self.is_fitted = True

        print(f"Model fitted with parameters: {self.best_params_}")

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Enhanced predict method with feature selection."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions.")

        # Convert to DataFrame if necessary
        if isinstance(X, np.ndarray):
            if hasattr(self, 'selected_features_') and self.selected_features_:
                X = pd.DataFrame(X, columns=self.selected_features_)
            else:
                X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

        # Select features if feature selection was performed
        if hasattr(self, 'selected_features_') and self.selected_features_:
            X_selected = X[self.selected_features_]
        else:
            X_selected = X

        # Scale features
        X_scaled = self.scaler.transform(X_selected)

        return self.model.predict(X_scaled)

    def calculate_comprehensive_statistics(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                         y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive statistics using the statistics module."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating statistics.")

        # Prepare data for statistics calculation
        X_train_selected = X_train[self.selected_features_] if self.selected_features_ else X_train
        X_test_selected = X_test[self.selected_features_] if self.selected_features_ else X_test

        X_train_scaled = self.scaler.transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)

        # Calculate all statistics
        try:
            stats = calculate_all(
                X_train_scaled, X_test_scaled,
                y_train.values, y_test.values,
                self.model
            )
            return stats
        except Exception as e:
            warnings.warn(f"Error calculating comprehensive statistics: {e}")
            return {}
