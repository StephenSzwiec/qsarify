import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit

from ..utils import statistics


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
    random_state: Optional[int] = None,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Performs Y-scrambling validation to test for chance correlations.

    Args:
        model: The regression model to validate.
        X_train: The training input features.
        y_train: The training target variable.
        n_permutations: The number of permutations to perform.
        random_state: Seed for the random number generator.
        n_jobs: The number of jobs to run in parallel.

    Returns:
        A dictionary containing the results of the Y-scrambling validation.
    """
    if random_state:
        np.random.seed(random_state)

    # Original metrics
    original_model = clone(model)
    original_model.fit(X_train, y_train)
    y_pred_train = original_model.predict(X_train)

    original_r2_train = statistics.calculate_r_squared(y_train, y_pred_train)
    original_q2_loo = statistics.calculate_q_squared_loo(
        clone(model), X_train.values, y_train.values
    )
    original_kxy = statistics.calculate_kxy(X_train.values, y_train.values)

    # Scrambled metrics
    y_train_perms = [
        np.random.permutation(y_train.values) for _ in range(n_permutations)
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_calculate_y_scrambling_metrics)(
            model, X_train.values, y_perm
        )
        for y_perm in y_train_perms
    )

    scrambled_r2_trains, scrambled_q2_loos, scrambled_kxys = zip(*results)

    return {
        "original": {
            "R2_train": original_r2_train,
            "Q2_LOO": original_q2_loo,
            "Kxy": original_kxy,
        },
        "scrambled": {
            "R2_train": list(scrambled_r2_trains),
            "Q2_LOO": list(scrambled_q2_loos),
            "Kxy": list(scrambled_kxys),
        },
    }


def leave_many_out_cv(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 100,
    test_size: float = 0.2,
    n_jobs: int = -1,
) -> Dict[str, List[float]]:
    """
    Performs Leave-Many-Out cross-validation (Monte Carlo cross-validation).

    Args:
        model: The regression model to validate.
        X: The input features.
        y: The target variable.
        n_iter: The number of iterations.
        test_size: The proportion of the dataset to include in the test split.
        n_jobs: The number of jobs to run in parallel.

    Returns:
        A dictionary containing the R-squared scores for each iteration.
    """

    def fit_and_score(train_index, test_index):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)
        return statistics.calculate_r_squared(y_test, y_pred)

    cv = ShuffleSplit(n_splits=n_iter, test_size=test_size, random_state=42)

    scores = Parallel(n_jobs=n_jobs)(
        delayed(fit_and_score)(train_index, test_index)
        for train_index, test_index in cv.split(X)
    )

    return {"scores": scores}
