"""
Core statistical metrics and utility functions.

This module provides standalone, vectorized functions for calculating various
statistical metrics used throughout the QSARify library, particularly for model
evaluation and validation. These functions are designed to work with NumPy arrays
and are optimized for performance.

The metrics implemented here are based on the definitions provided in the
project's technical design document (GEMINI.md, Section 6.1).
"""
import numpy as np
from sklearn.model_selection import LeaveOneOut

def calculate_kxx(X: np.ndarray) -> float:
    """Calculates the correlation among descriptors in X."""
    p = X.shape[1]
    if p < 2:
        return 0.0
    eigenvalues = np.linalg.eigvalsh(np.corrcoef(X, rowvar=False))
    sum_eigenvalues = np.sum(eigenvalues)
    if sum_eigenvalues == 0:
        return 0.0
    return np.sum(np.abs(eigenvalues / sum_eigenvalues - 1 / p)) / (2 * (p - 1) / p)

def calculate_kxy(X: np.ndarray, y: np.ndarray) -> float:
    """Calculates the correlation between the response y and the descriptors X."""
    Xy = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    return calculate_kxx(Xy)

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the Mean Absolute Error (MAE)."""
    return np.mean(np.abs(y_true - y_pred))

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the Root Mean Squared Error (RMSE)."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the Mean Squared Error (MSE)."""
    return np.mean((y_true - y_pred) ** 2)

def calculate_rss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the Residual Sum of Squares (RSS)."""
    return np.sum((y_true - y_pred) ** 2)

def calculate_mss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the Model Sum of Squares (MSS)."""
    return np.sum((y_pred - np.mean(y_true)) ** 2)

def calculate_press_loo(model, X: np.ndarray, y: np.ndarray) -> float:
    """
    Calculates the Predicted Residual Error Sum of Squares (PRESS)
    using Leave-One-Out cross-validation.
    """
    loo = LeaveOneOut()
    y_pred_loo = np.zeros_like(y, dtype=float)
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, _ = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred_loo[test_index] = model.predict(X_test)
    return np.sum((y - y_pred_loo) ** 2)

def calculate_tss(y_true: np.ndarray) -> float:
    """Calculates the Total Sum of Squares (TSS)."""
    return np.sum((y_true - np.mean(y_true)) ** 2)

def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the coefficient of determination (R²)."""
    rss = calculate_rss(y_true, y_pred)
    tss = calculate_tss(y_true)
    if tss == 0:
        return 1.0 if rss == 0 else 0.0
    return 1 - rss / tss

def calculate_r_squared_adj(y_true: np.ndarray, y_pred: np.ndarray, p: int) -> float:
    """Calculates the adjusted R²."""
    n = len(y_true)
    if n - p - 1 == 0:
        return np.nan
    r_squared = calculate_r_squared(y_true, y_pred)
    return 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

def calculate_s(y_true: np.ndarray, y_pred: np.ndarray, p: int) -> float:
    """Calculates the standard error of the estimate."""
    n = len(y_true)
    if n - p - 1 == 0:
        return np.nan
    rss = calculate_rss(y_true, y_pred)
    return np.sqrt(rss / (n - p - 1))

def calculate_f_statistic(y_true: np.ndarray, y_pred: np.ndarray, p: int) -> float:
    """Calculates the F-statistic."""
    n = len(y_true)
    if n - p - 1 == 0 or p == 0:
        return np.nan
    mss = calculate_mss(y_true, y_pred)
    rss = calculate_rss(y_true, y_pred)
    return (mss / p) / (rss / (n - p - 1))

def calculate_lof(y_true: np.ndarray, y_pred: np.ndarray, p: int) -> float:
    """Calculates Friedman's Lack of Fit (LOF)."""
    n = len(y_true)
    if (1 - 2 * p / n) == 0:
        return np.nan
    mse = calculate_mse(y_true, y_pred)
    return mse / (1 - 2 * p / n) ** 2

def calculate_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Concordance Correlation Coefficient (CCC).

    This metric assesses the agreement between two variables (e.g., true and
    predicted values). It accounts for both correlation and bias. The calculation
    is adjusted to use the sample variance (ddof=1) to maintain consistency
    with the sample covariance calculation.
    """
    # Use ddof=1 to calculate the sample variance, matching np.cov's default
    var_true = np.var(y_true, ddof=1)
    var_pred = np.var(y_pred, ddof=1)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # np.cov returns the covariance matrix, we need the covariance value
    cov = np.cov(y_true, y_pred)[0, 1]

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    # Handle the edge case where the denominator is zero
    if denominator == 0:
        # If both variances and mean difference are zero, it's a perfect match
        return 1.0
    
    return (2 * cov) / denominator

def calculate_q_squared_loo(model, X: np.ndarray, y: np.ndarray) -> float:
    """
    Calculates the cross-validated R² (Q²_LOO) using Leave-One-Out cross-validation.
    """
    press = calculate_press_loo(model, X, y)
    tss = calculate_tss(y)
    if tss == 0:
        return 1.0 if press == 0 else 0.0
    return 1 - (press / tss)

def calculate_press_ext(y_true_ext: np.ndarray, y_pred_ext: np.ndarray) -> float:
    """Calculates the PRESS on an external test set."""
    return calculate_rss(y_true_ext, y_pred_ext)

def calculate_q_squared_f1(y_true_ext: np.ndarray, y_pred_ext: np.ndarray, y_true_train: np.ndarray) -> float:
    """Calculates the external Q² (F1)."""
    press_ext = calculate_press_ext(y_true_ext, y_pred_ext)
    ss_tr_mean = np.sum((y_true_ext - np.mean(y_true_train)) ** 2)
    if ss_tr_mean == 0:
        return 1.0 if press_ext == 0 else 0.0
    return 1 - press_ext / ss_tr_mean

def calculate_q_squared_f2(y_true_ext: np.ndarray, y_pred_ext: np.ndarray) -> float:
    """Calculates the external Q² (F2)."""
    press_ext = calculate_press_ext(y_true_ext, y_pred_ext)
    ss_ext_mean = calculate_tss(y_true_ext)
    if ss_ext_mean == 0:
        return 1.0 if press_ext == 0 else 0.0
    return 1 - press_ext / ss_ext_mean

def calculate_q_squared_f3(y_true_ext: np.ndarray, y_pred_ext: np.ndarray, y_true_train: np.ndarray) -> float:
    """Calculates the external Q² (F3)."""
    press_ext = calculate_press_ext(y_true_ext, y_pred_ext)
    tss_train = calculate_tss(y_true_train)
    if tss_train == 0 or len(y_true_ext) == 0:
        return 1.0 if press_ext == 0 else 0.0
    return 1 - (press_ext * len(y_true_train)) / (tss_train * len(y_true_ext))

def calculate_k(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the slope of the regression line through the origin (y_true = k * y_pred)."""
    if np.sum(y_pred ** 2) == 0:
        return np.inf
    return np.sum(y_true * y_pred) / np.sum(y_pred ** 2)

def calculate_k_prime(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the slope of the regression line through the origin (y_pred = k' * y_true)."""
    if np.sum(y_true ** 2) == 0:
        return np.inf
    return np.sum(y_true * y_pred) / np.sum(y_true ** 2)

def calculate_r_squared_ext(y_true_ext: np.ndarray, y_pred_ext: np.ndarray) -> float:
    """Calculates the external R²."""
    return calculate_q_squared_f2(y_true_ext, y_pred_ext)

def calculate_r_squared_0(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates R² for regression through the origin (y_true = k * y_pred).
    It measures the performance of a new model without an intercept.
    """
    k = calculate_k(y_true, y_pred)
    y_pred_origin = k * y_pred
    rss_origin = np.sum((y_true - y_pred_origin) ** 2)
    tss_origin = np.sum(y_true ** 2)
    if tss_origin == 0:
        return 1.0 if rss_origin == 0 else 0.0
    return 1 - rss_origin / tss_origin

def calculate_r_prime_squared_0(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates R'² for regression through the origin (y_pred = k' * y_true).
    It measures the performance of a new model without an intercept.
    """
    k_prime = calculate_k_prime(y_true, y_pred)
    y_pred_origin = k_prime * y_true
    rss_origin = np.sum((y_pred - y_pred_origin) ** 2)
    tss_origin = np.sum(y_pred ** 2)
    if tss_origin == 0:
        return 1.0 if rss_origin == 0 else 0.0
    return 1 - rss_origin / tss_origin

def calculate_roy_metrics(r_squared_ext: float, r_squared_0: float, r_prime_squared_0: float, r_squared: float) -> tuple[float, float]:
    """Calculates Roy's criteria (r_m_bar_sq and delta_r_m_sq)."""
    r_m_sq = r_squared * (1 - np.sqrt(np.abs(r_squared_ext - r_squared_0)))
    r_prime_m_sq = r_squared * (1 - np.sqrt(np.abs(r_squared_ext - r_prime_squared_0)))
    r_m_bar_sq = (r_m_sq + r_prime_m_sq) / 2
    delta_r_m_sq = np.abs(r_m_sq - r_prime_m_sq)
    return r_m_bar_sq, delta_r_m_sq

def calculate_closeness_metrics(r_squared: float, r_squared_0: float, r_prime_squared_0: float) -> tuple[float, float]:
    """Calculates the closeness criteria from Golbraikh and Tropsha."""
    if r_squared == 0:
        return (np.inf, np.inf)
    clos = np.abs(r_squared - r_squared_0) / r_squared
    clos_prime = np.abs(r_squared - r_prime_squared_0) / r_squared
    return clos, clos_prime