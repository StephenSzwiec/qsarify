"""
Configuration and fixtures for statistical utility tests.

This file defines a shared fixture that provides a consistent set of synthetic
data for all tests in the `tests/utils` directory. Using a common dataset
ensures that the statistical functions are tested against a reproducible
and coherent baseline.
"""
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression

@pytest.fixture(scope="module")
def synthetic_data():
    """
    Provides a consistent set of synthetic data for testing statistical functions.

    This fixture generates a reproducible dataset with training and external sets,
    fits a simple linear regression model, and provides all necessary components
    for testing the functions in the `qsarify.utils.statistics` module.

    The data has a known linear relationship with added noise:
    y = 2*x1 + 3*x2 - 1.5*x3 + 5 + noise

    Returns:
        dict: A dictionary containing:
            - X_train (np.ndarray): Training set descriptors.
            - y_true_train (np.ndarray): Training set true response values.
            - y_pred_train (np.ndarray): Training set predicted response values.
            - X_ext (np.ndarray): External set descriptors.
            - y_true_ext (np.ndarray): External set true response values.
            - y_pred_ext (np.ndarray): External set predicted response values.
            - p (int): The number of descriptors.
            - model (LinearRegression): A model fitted on the training data.
    """
    np.random.seed(42)
    n_train = 50
    n_ext = 20
    p = 3

    # Training data
    X_train = np.random.rand(n_train, p) * 10
    # Create a linear relationship with some noise
    y_true_train = (
        2 * X_train[:, 0]
        + 3 * X_train[:, 1]
        - 1.5 * X_train[:, 2]
        + 5
        + np.random.normal(0, 2, n_train)
    )

    # Fit a model to get predicted values
    model = LinearRegression()
    model.fit(X_train, y_true_train)
    y_pred_train = model.predict(X_train)

    # External (test) data
    X_ext = np.random.rand(n_ext, p) * 10
    y_true_ext = (
        2 * X_ext[:, 0]
        + 3 * X_ext[:, 1]
        - 1.5 * X_ext[:, 2]
        + 5
        + np.random.normal(0, 2, n_ext)
    )
    y_pred_ext = model.predict(X_ext)

    return {
        "X_train": X_train,
        "y_true_train": y_true_train,
        "y_pred_train": y_pred_train,
        "X_ext": X_ext,
        "y_true_ext": y_true_ext,
        "y_pred_ext": y_pred_ext,
        "p": p,
        "model": model,
    }
