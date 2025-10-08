import random
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit

from ..utils import statistics
from ..exceptions import ValidationError


@dataclass
class ValidationResult:
    """Base class for validation results."""
    original_metrics: Dict[str, float]
    procedure_name: str
    n_iterations: int
    model_name: Optional[str] = None
    

@dataclass
class LeaveManyOutResult(ValidationResult):
    """Results from Leave-Many-Out cross-validation."""
    cv_scores: List[float]
    mean_score: float
    std_score: float
    confidence_interval: tuple[float, float]
    test_size: float
    
    def __post_init__(self):
        self.procedure_name = "Leave-Many-Out CV"


@dataclass  
class YScramblingResult(ValidationResult):
    """Results from Y-scrambling validation."""
    scrambled_metrics: Dict[str, List[float]]
    scrambled_mean: Dict[str, float]
    scrambled_std: Dict[str, float]
    p_values: Dict[str, float]
    is_significant: Dict[str, bool]
    significance_threshold: float
    
    def __post_init__(self):
        self.procedure_name = "Y-scrambling"


def _calculate_y_scrambling_metrics(model, X_train, y_train_perm):
    """Helper function to calculate metrics for a single y-scrambling permutation."""
    mdl = clone(model)
    mdl.fit(X_train, y_train_perm)
    y_pred_perm = mdl.predict(X_train)

    r2_train = statistics.calculate_r_squared(y_train_perm, y_pred_perm)
    q2_loo = statistics.calculate_q_squared_loo(clone(model), X_train, y_train_perm)
    kxy = statistics.calculate_kxy(X_train, y_train_perm)

    return r2_train, q2_loo, kxy


def y_scrambling(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_permutations: int = 100,
    significance_threshold: float = 0.05,
    random_state: Optional[int] = None,
    n_jobs: int = -1,
    model_name: Optional[str] = None
) -> YScramblingResult:
    """
    Performs Y-scrambling validation to test for chance correlations.
    
    This procedure randomly permutes the target variable while keeping features intact,
    then trains models on scrambled data. If the original model significantly outperforms
    scrambled models, it suggests the model captures real relationships rather than noise.

    Args:
        model: The regression model to validate (unfitted).
        X_train: The training input features.
        y_train: The training target variable.
        n_permutations: The number of random permutations to perform.
        significance_threshold: P-value threshold for statistical significance.
        random_state: Seed for the random number generator.
        n_jobs: The number of jobs to run in parallel (-1 uses all available cores).
        model_name: Optional name for the model being validated.

    Returns:
        YScramblingResult containing comprehensive scrambling validation statistics.
        
    Raises:
        ValidationError: If validation cannot be performed.
    """
    if n_permutations < 50:
        raise ValidationError(f"n_permutations should be at least 50 for reliable statistics, got {n_permutations}")
    
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    # Calculate original metrics
    try:
        original_model = clone(model)
        original_model.fit(X_train, y_train)
        y_pred_train = original_model.predict(X_train)

        original_r2_train = statistics.calculate_r_squared(y_train.values, y_pred_train)
        original_q2_loo = statistics.calculate_q_squared_loo(
            clone(model), X_train.values, y_train.values
        )
        original_kxy = statistics.calculate_kxy(X_train.values, y_train.values)
        original_rmse = statistics.calculate_rmse(y_train.values, y_pred_train)
        
        original_metrics = {
            "R2_train": original_r2_train,
            "Q2_LOO": original_q2_loo, 
            "Kxy": original_kxy,
            "RMSE": original_rmse
        }
    except Exception as e:
        raise ValidationError(f"Failed to calculate original model metrics: {e}")

    # Generate scrambled datasets
    y_train_perms = []
    for _ in range(n_permutations):
        perm = np.random.permutation(y_train.values)
        # Ensure scrambled data is different from original
        while np.array_equal(perm, y_train.values) and len(y_train) > 1:
            perm = np.random.permutation(y_train.values)
        y_train_perms.append(perm)

    # Calculate scrambled metrics in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_calculate_y_scrambling_metrics)(
            model, X_train.values, y_perm
        )
        for y_perm in y_train_perms
    )

    # Process results
    valid_results = [r for r in results if not any(np.isnan(v) for v in r)]
    
    if len(valid_results) < n_permutations * 0.8:
        raise ValidationError(f"Too many scrambling iterations failed ({len(valid_results)}/{n_permutations} successful)")

    scrambled_r2_trains, scrambled_q2_loos, scrambled_kxys = zip(*valid_results)
    
    # Calculate additional metrics for scrambled models
    scrambled_rmses = []
    for y_perm in y_train_perms[:len(valid_results)]:
        try:
            temp_model = clone(model)
            temp_model.fit(X_train.values, y_perm)
            y_pred_perm = temp_model.predict(X_train.values)
            rmse = statistics.calculate_rmse(y_perm, y_pred_perm)
            scrambled_rmses.append(rmse)
        except:
            scrambled_rmses.append(np.nan)
    
    scrambled_rmses = [r for r in scrambled_rmses if not np.isnan(r)]

    scrambled_metrics = {
        "R2_train": list(scrambled_r2_trains),
        "Q2_LOO": list(scrambled_q2_loos),
        "Kxy": list(scrambled_kxys),
        "RMSE": scrambled_rmses
    }

    # Calculate summary statistics for scrambled results
    scrambled_mean = {
        metric: np.mean(values) for metric, values in scrambled_metrics.items()
    }
    
    scrambled_std = {
        metric: np.std(values, ddof=1) for metric, values in scrambled_metrics.items()
    }

    # Calculate p-values (fraction of scrambled models that exceed original performance)
    p_values = {}
    is_significant = {}
    
    for metric in ["R2_train", "Q2_LOO"]:  # Higher is better
        n_better = sum(1 for val in scrambled_metrics[metric] 
                      if val >= original_metrics[metric])
        p_values[metric] = (n_better + 1) / (len(scrambled_metrics[metric]) + 1)
        is_significant[metric] = p_values[metric] <= significance_threshold
    
    for metric in ["RMSE"]:  # Lower is better  
        n_better = sum(1 for val in scrambled_metrics[metric]
                      if val <= original_metrics[metric])
        p_values[metric] = (n_better + 1) / (len(scrambled_metrics[metric]) + 1)
        is_significant[metric] = p_values[metric] <= significance_threshold
    
    # Don't calculate p-value for Kxy as it's a descriptor-dependent metric
    p_values["Kxy"] = np.nan
    is_significant["Kxy"] = False

    return YScramblingResult(
        original_metrics=original_metrics,
        procedure_name="Y-scrambling",
        n_iterations=len(valid_results),
        model_name=model_name,
        scrambled_metrics=scrambled_metrics,
        scrambled_mean=scrambled_mean,
        scrambled_std=scrambled_std,
        p_values=p_values,
        is_significant=is_significant,
        significance_threshold=significance_threshold
    )


def leave_many_out_cv(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 100,
    test_size: float = 0.2,
    confidence_level: float = 0.95,
    n_jobs: int = -1,
    random_state: Optional[int] = 42,
    model_name: Optional[str] = None
) -> LeaveManyOutResult:
    """
    Performs Leave-Many-Out cross-validation (Monte Carlo cross-validation).
    
    This procedure repeatedly splits the data randomly, fits the model on the training
    portion, and evaluates on the test portion. Provides robust assessment of model
    generalization performance with confidence intervals.

    Args:
        model: The regression model to validate (can be fitted or unfitted).
        X: The input features.
        y: The target variable.
        n_iter: The number of iterations.
        test_size: The proportion of the dataset to include in the test split.
        confidence_level: Confidence level for the confidence interval (0-1).
        n_jobs: The number of jobs to run in parallel (-1 uses all available cores).
        random_state: Seed for reproducibility.
        model_name: Optional name for the model being validated.

    Returns:
        LeaveManyOutResult containing comprehensive cross-validation statistics.
        
    Raises:
        ValidationError: If validation cannot be performed.
    """
    if not (0.1 <= test_size <= 0.5):
        raise ValidationError(f"test_size should be between 0.1 and 0.5, got {test_size}")
        
    if n_iter < 10:
        raise ValidationError(f"n_iter should be at least 10 for reliable statistics, got {n_iter}")
    
    # Calculate original metrics if model is already fitted
    original_metrics = {}
    try:
        if hasattr(model, 'predict') and hasattr(model, 'is_fitted') and getattr(model, 'is_fitted', False):
            # Model is fitted, calculate original metrics
            y_pred_full = model.predict(X)
            original_metrics['R2'] = statistics.calculate_r_squared(y.values, y_pred_full)
            original_metrics['RMSE'] = statistics.calculate_rmse(y.values, y_pred_full)
            original_metrics['MAE'] = statistics.calculate_mae(y.values, y_pred_full)
        else:
            # Model is unfitted, fit on full data for original metrics
            model_temp = clone(model)
            model_temp.fit(X, y)
            y_pred_full = model_temp.predict(X)
            original_metrics['R2'] = statistics.calculate_r_squared(y.values, y_pred_full)
            original_metrics['RMSE'] = statistics.calculate_rmse(y.values, y_pred_full)
            original_metrics['MAE'] = statistics.calculate_mae(y.values, y_pred_full)
    except Exception as e:
        raise ValidationError(f"Failed to calculate original metrics: {e}")

    def fit_and_score(train_index, test_index):
        """Fit model and calculate comprehensive metrics."""
        try:
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)
            
            # Calculate multiple metrics
            r2 = statistics.calculate_r_squared(y_test.values, y_pred)
            rmse = statistics.calculate_rmse(y_test.values, y_pred)
            mae = statistics.calculate_mae(y_test.values, y_pred)
            
            return {'R2': r2, 'RMSE': rmse, 'MAE': mae}
        except Exception:
            # Return NaN values if fitting/prediction fails
            return {'R2': np.nan, 'RMSE': np.nan, 'MAE': np.nan}

    # Perform cross-validation
    cv = ShuffleSplit(n_splits=n_iter, test_size=test_size, random_state=random_state)

    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_and_score)(train_index, test_index)
        for train_index, test_index in cv.split(X)
    )
    
    # Extract R2 scores (primary metric)
    cv_scores = [r['R2'] for r in results if not np.isnan(r['R2'])]
    
    if len(cv_scores) < n_iter * 0.8:  # Less than 80% successful
        raise ValidationError(f"Too many CV iterations failed ({len(cv_scores)}/{n_iter} successful)")
    
    # Calculate statistics
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores, ddof=1)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    if len(cv_scores) >= 30:
        # Use normal approximation for large samples
        z_value = 1.96 if confidence_level == 0.95 else abs(np.percentile(np.random.standard_normal(10000), alpha/2 * 100))
        margin_error = z_value * std_score / np.sqrt(len(cv_scores))
    else:
        # Use t-distribution for small samples
        from scipy import stats as scipy_stats
        t_value = scipy_stats.t.ppf(1 - alpha/2, len(cv_scores) - 1)
        margin_error = t_value * std_score / np.sqrt(len(cv_scores))
    
    confidence_interval = (mean_score - margin_error, mean_score + margin_error)

    return LeaveManyOutResult(
        original_metrics=original_metrics,
        procedure_name="Leave-Many-Out CV",
        n_iterations=len(cv_scores),
        model_name=model_name,
        cv_scores=cv_scores,
        mean_score=mean_score,
        std_score=std_score,
        confidence_interval=confidence_interval,
        test_size=test_size
    )
