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
from skopt.utils import use_named_args


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization"""
    # Bayesian optimization settings
    n_calls: int = 50
    n_initial_points: int = 10
    cv_folds: int = 3
    val_split: float = 0.2
    n_jobs: int = -1
    random_state: int = 42
    
    # Gradient-based optimization settings (for non-MLR models)
    use_gradient_optimization: bool = False
    max_iter: int = 100
    learning_rate: float = 0.01
    convergence_tol: float = 1e-6
    n_cv_splits: int = 5


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
        """Enhanced hyperparameter optimization with gradient-based option for non-MLR models."""
        
        if self.optimization_config.use_gradient_optimization:
            return self._gradient_based_optimization(X, y)
        else:
            return self._bayesian_optimization(X, y)
    
    def _bayesian_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Traditional Bayesian optimization using scikit-optimize."""
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
    
    def _gradient_based_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Gradient-based hyperparameter optimization for non-MLR models.
        Uses gradient descent on CV performance with parallelizable training.
        """
        from sklearn.model_selection import KFold
        
        param_space = self._get_param_space()
        if not param_space:
            return self.get_default_params()
        
        # Initialize parameters at default values
        current_params = {}
        param_bounds = {}
        
        for param in param_space:
            if isinstance(param, Real):
                # Start at geometric mean for log-uniform, arithmetic mean otherwise
                if hasattr(param, 'prior') and param.prior == 'log-uniform':
                    current_params[param.name] = np.exp((np.log(param.low) + np.log(param.high)) / 2)
                else:
                    current_params[param.name] = (param.low + param.high) / 2
                param_bounds[param.name] = (param.low, param.high)
            elif isinstance(param, Integer):
                current_params[param.name] = int((param.low + param.high) / 2)
                param_bounds[param.name] = (param.low, param.high)
        
        kf = KFold(n_splits=self.optimization_config.n_cv_splits, 
                   shuffle=True, random_state=self.optimization_config.random_state)
        
        best_score = -np.inf
        best_params = current_params.copy()
        convergence_count = 0
        
        for iteration in range(self.optimization_config.max_iter):
            # Evaluate current parameters
            current_score = self._evaluate_params_cv(current_params, X, y, kf)
            
            if current_score > best_score:
                best_score = current_score
                best_params = current_params.copy()
                convergence_count = 0
            else:
                convergence_count += 1
            
            # Check convergence
            if convergence_count >= 10:  # Early stopping
                break
            
            # Gradient approximation using finite differences
            gradients = {}
            step_sizes = {}
            
            for param_name in current_params:
                if isinstance(param_space[[p.name for p in param_space].index(param_name)], Real):
                    # Adaptive step size (smaller for log-uniform parameters)
                    param_obj = param_space[[p.name for p in param_space].index(param_name)]
                    if hasattr(param_obj, 'prior') and param_obj.prior == 'log-uniform':
                        step_size = current_params[param_name] * 0.1
                    else:
                        step_size = (param_bounds[param_name][1] - param_bounds[param_name][0]) * 0.05
                    step_sizes[param_name] = step_size
                    
                    # Forward difference
                    perturbed_params = current_params.copy()
                    perturbed_params[param_name] = min(
                        param_bounds[param_name][1],
                        current_params[param_name] + step_size
                    )
                    forward_score = self._evaluate_params_cv(perturbed_params, X, y, kf)
                    
                    gradients[param_name] = (forward_score - current_score) / step_size
                elif isinstance(param_space[[p.name for p in param_space].index(param_name)], Integer):
                    # For integer parameters, try ±1
                    step_sizes[param_name] = 1
                    
                    perturbed_params = current_params.copy()
                    perturbed_params[param_name] = min(
                        param_bounds[param_name][1],
                        current_params[param_name] + 1
                    )
                    forward_score = self._evaluate_params_cv(perturbed_params, X, y, kf)
                    
                    gradients[param_name] = forward_score - current_score
            
            # Update parameters using gradient ascent (maximizing score)
            for param_name in current_params:
                if gradients[param_name] != 0:
                    if isinstance(param_space[[p.name for p in param_space].index(param_name)], Real):
                        update = self.optimization_config.learning_rate * gradients[param_name]
                        current_params[param_name] += update
                        # Clip to bounds
                        current_params[param_name] = np.clip(
                            current_params[param_name],
                            param_bounds[param_name][0],
                            param_bounds[param_name][1]
                        )
                    elif isinstance(param_space[[p.name for p in param_space].index(param_name)], Integer):
                        if gradients[param_name] > 0:
                            current_params[param_name] = min(
                                param_bounds[param_name][1],
                                current_params[param_name] + 1
                            )
                        elif gradients[param_name] < 0:
                            current_params[param_name] = max(
                                param_bounds[param_name][0],
                                current_params[param_name] - 1
                            )
        
        self.best_score_ = best_score
        self.optimization_history_ = []  # Not tracking full history for gradient method
        
        return best_params
    
    def _evaluate_params_cv(self, params: Dict[str, Any], X: np.ndarray, y: np.ndarray, kf) -> float:
        """Evaluate parameter set using cross-validation."""
        try:
            scores = []
            for train_idx, val_idx in kf.split(X):
                X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                
                model = self._create_model(**params)
                model.fit(X_train_cv, y_train_cv)
                score = model.score(X_val_cv, y_val_cv)
                scores.append(score)
            
            return np.mean(scores)
        except Exception:
            return -np.inf

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
