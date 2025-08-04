"""
Unit tests for the core statistical utility functions.

These tests use a shared synthetic dataset defined in the `conftest.py`
file to ensure all statistical metrics are evaluated against a consistent
and reproducible baseline.
"""
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from qsarify.utils import statistics

def test_calculate_kxx(synthetic_data):
    """Test Kxx calculation for descriptor correlation."""
    X = synthetic_data["X_train"]
    # The exact value is not critical, but it should be a valid float
    kxx = statistics.calculate_kxx(X)
    assert isinstance(kxx, float)
    assert not np.isnan(kxx)

    # Test edge case: perfectly correlated descriptors
    X_correlated = np.array([[1, 2], [2, 4], [3, 6]])
    assert np.isclose(statistics.calculate_kxx(X_correlated), 1.0)

    # Test edge case: uncorrelated descriptors
    X_uncorrelated = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    assert np.isclose(statistics.calculate_kxx(X_uncorrelated), 0.0, atol=1e-9)

def test_calculate_kxy(synthetic_data):
    """Test Kxy calculation for descriptor-response correlation."""
    X = synthetic_data["X_train"]
    y = synthetic_data["y_true_train"]
    kxy = statistics.calculate_kxy(X, y)
    assert isinstance(kxy, float)
    assert not np.isnan(kxy)

def test_calculate_mae(synthetic_data):
    """Test Mean Absolute Error (MAE) calculation."""
    y_true = synthetic_data["y_true_train"]
    y_pred = synthetic_data["y_pred_train"]
    mae = statistics.calculate_mae(y_true, y_pred)
    expected_mae = np.mean(np.abs(y_true - y_pred))
    assert np.isclose(mae, expected_mae)

def test_calculate_rmse(synthetic_data):
    """Test Root Mean Squared Error (RMSE) calculation."""
    y_true = synthetic_data["y_true_train"]
    y_pred = synthetic_data["y_pred_train"]
    rmse = statistics.calculate_rmse(y_true, y_pred)
    expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    assert np.isclose(rmse, expected_rmse)

def test_calculate_mse(synthetic_data):
    """Test Mean Squared Error (MSE) calculation."""
    y_true = synthetic_data["y_true_train"]
    y_pred = synthetic_data["y_pred_train"]
    mse = statistics.calculate_mse(y_true, y_pred)
    expected_mse = np.mean((y_true - y_pred) ** 2)
    assert np.isclose(mse, expected_mse)

def test_calculate_rss(synthetic_data):
    """Test Residual Sum of Squares (RSS) calculation."""
    y_true = synthetic_data["y_true_train"]
    y_pred = synthetic_data["y_pred_train"]
    rss = statistics.calculate_rss(y_true, y_pred)
    expected_rss = np.sum((y_true - y_pred) ** 2)
    assert np.isclose(rss, expected_rss)

def test_calculate_mss(synthetic_data):
    """Test Model Sum of Squares (MSS) calculation."""
    y_true = synthetic_data["y_true_train"]
    y_pred = synthetic_data["y_pred_train"]
    mss = statistics.calculate_mss(y_true, y_pred)
    expected_mss = np.sum((y_pred - np.mean(y_true)) ** 2)
    assert np.isclose(mss, expected_mss)

def test_calculate_press_loo(synthetic_data):
    """Test PRESS (Leave-One-Out) calculation."""
    X = synthetic_data["X_train"]
    y = synthetic_data["y_true_train"]
    model = synthetic_data["model"]
    press = statistics.calculate_press_loo(model, X, y)
    assert isinstance(press, float)
    assert not np.isnan(press)
    # PRESS should be greater than RSS for a non-perfect model
    rss = statistics.calculate_rss(y, synthetic_data["y_pred_train"])
    assert press > rss

def test_calculate_tss(synthetic_data):
    """Test Total Sum of Squares (TSS) calculation."""
    y_true = synthetic_data["y_true_train"]
    tss = statistics.calculate_tss(y_true)
    expected_tss = np.sum((y_true - np.mean(y_true)) ** 2)
    assert np.isclose(tss, expected_tss)

def test_calculate_r_squared(synthetic_data):
    """Test R² calculation."""
    y_true = synthetic_data["y_true_train"]
    y_pred = synthetic_data["y_pred_train"]
    r2 = statistics.calculate_r_squared(y_true, y_pred)
    assert 0 < r2 <= 1.0

def test_calculate_r_squared_adj(synthetic_data):
    """Test adjusted R² calculation."""
    y_true = synthetic_data["y_true_train"]
    y_pred = synthetic_data["y_pred_train"]
    p = synthetic_data["p"]
    r2_adj = statistics.calculate_r_squared_adj(y_true, y_pred, p)
    r2 = statistics.calculate_r_squared(y_true, y_pred)
    assert r2_adj < r2

def test_calculate_s(synthetic_data):
    """Test standard error of the estimate (s) calculation."""
    y_true = synthetic_data["y_true_train"]
    y_pred = synthetic_data["y_pred_train"]
    p = synthetic_data["p"]
    s = statistics.calculate_s(y_true, y_pred, p)
    rss = statistics.calculate_rss(y_true, y_pred)
    n = len(y_true)
    expected_s = np.sqrt(rss / (n - p - 1))
    assert np.isclose(s, expected_s)

def test_calculate_f_statistic(synthetic_data):
    """Test F-statistic calculation."""
    y_true = synthetic_data["y_true_train"]
    y_pred = synthetic_data["y_pred_train"]
    p = synthetic_data["p"]
    f_stat = statistics.calculate_f_statistic(y_true, y_pred, p)
    assert f_stat > 0

def test_calculate_lof(synthetic_data):
    """Test Friedman's Lack of Fit (LOF) calculation."""
    y_true = synthetic_data["y_true_train"]
    y_pred = synthetic_data["y_pred_train"]
    p = synthetic_data["p"]
    lof = statistics.calculate_lof(y_true, y_pred, p)
    assert lof > 0

def test_calculate_ccc(synthetic_data):
    """Test Concordance Correlation Coefficient (CCC) calculation."""
    y_true = synthetic_data["y_true_train"]
    y_pred = synthetic_data["y_pred_train"]
    ccc = statistics.calculate_ccc(y_true, y_pred)
    assert 0 < ccc <= 1.0

def test_calculate_q_squared_loo(synthetic_data):
    """Test Q²_LOO calculation."""
    X = synthetic_data["X_train"]
    y = synthetic_data["y_true_train"]
    model = synthetic_data["model"]
    q2_loo = statistics.calculate_q_squared_loo(model, X, y)
    r2 = statistics.calculate_r_squared(y, synthetic_data["y_pred_train"])
    assert q2_loo < r2

def test_calculate_press_ext(synthetic_data):
    """Test PRESS calculation on an external set."""
    y_true = synthetic_data["y_true_ext"]
    y_pred = synthetic_data["y_pred_ext"]
    press_ext = statistics.calculate_press_ext(y_true, y_pred)
    expected_press_ext = np.sum((y_true - y_pred) ** 2)
    assert np.isclose(press_ext, expected_press_ext)

def test_calculate_q_squared_f_metrics(synthetic_data):
    """Test the calculation of external Q² metrics (F1, F2, F3)."""
    y_train = synthetic_data["y_true_train"]
    y_ext = synthetic_data["y_true_ext"]
    y_pred_ext = synthetic_data["y_pred_ext"]

    q2_f1 = statistics.calculate_q_squared_f1(y_ext, y_pred_ext, y_train)
    q2_f2 = statistics.calculate_q_squared_f2(y_ext, y_pred_ext)
    q2_f3 = statistics.calculate_q_squared_f3(y_ext, y_pred_ext, y_train)

    assert isinstance(q2_f1, float)
    assert isinstance(q2_f2, float)
    assert isinstance(q2_f3, float)

def test_calculate_k_and_k_prime(synthetic_data):
    """Test calculation of slopes through the origin."""
    y_true = synthetic_data["y_true_train"]
    y_pred = synthetic_data["y_pred_train"]
    k = statistics.calculate_k(y_true, y_pred)
    k_prime = statistics.calculate_k_prime(y_true, y_pred)
    assert isinstance(k, float)
    assert isinstance(k_prime, float)

def test_calculate_r_squared_ext(synthetic_data):
    """Test external R² calculation."""
    y_true_ext = synthetic_data["y_true_ext"]
    y_pred_ext = synthetic_data["y_pred_ext"]
    r2_ext = statistics.calculate_r_squared_ext(y_true_ext, y_pred_ext)
    # This should be equivalent to Q²_F2
    q2_f2 = statistics.calculate_q_squared_f2(y_true_ext, y_pred_ext)
    assert np.isclose(r2_ext, q2_f2)

def test_calculate_r_squared_origin_metrics(synthetic_data):
    """Test R² through the origin metrics (R²_0, R'_²_0)."""
    y_true = synthetic_data["y_true_train"]
    y_pred = synthetic_data["y_pred_train"]
    r2_0 = statistics.calculate_r_squared_0(y_true, y_pred)
    r_prime_2_0 = statistics.calculate_r_prime_squared_0(y_true, y_pred)
    assert 0 <= r2_0 <= 1.0
    assert 0 <= r_prime_2_0 <= 1.0

def test_calculate_roy_metrics(synthetic_data):
    """Test Roy's criteria calculation."""
    y_true_train = synthetic_data["y_true_train"]
    y_pred_train = synthetic_data["y_pred_train"]
    y_true_ext = synthetic_data["y_true_ext"]
    y_pred_ext = synthetic_data["y_pred_ext"]

    r_sq = statistics.calculate_r_squared(y_true_train, y_pred_train)
    r_sq_ext = statistics.calculate_r_squared_ext(y_true_ext, y_pred_ext)
    r_sq_0 = statistics.calculate_r_squared_0(y_true_train, y_pred_train)
    r_prime_sq_0 = statistics.calculate_r_prime_squared_0(y_true_train, y_pred_train)

    r_m_bar_sq, delta_r_m_sq = statistics.calculate_roy_metrics(
        r_sq_ext, r_sq_0, r_prime_sq_0, r_sq
    )
    assert isinstance(r_m_bar_sq, float)
    assert isinstance(delta_r_m_sq, float)

def test_calculate_closeness_metrics(synthetic_data):
    """Test closeness criteria calculation."""
    y_true = synthetic_data["y_true_train"]
    y_pred = synthetic_data["y_pred_train"]

    r_sq = statistics.calculate_r_squared(y_true, y_pred)
    r_sq_0 = statistics.calculate_r_squared_0(y_true, y_pred)
    r_prime_sq_0 = statistics.calculate_r_prime_squared_0(y_true, y_pred)

    clos, clos_prime = statistics.calculate_closeness_metrics(
        r_sq, r_sq_0, r_prime_sq_0
    )
    assert clos > 0
    assert clos_prime > 0