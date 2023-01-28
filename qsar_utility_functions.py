from sklearn.metrics import mean_squared_error
import numpy as np
from itertools import repeat

"""
Utility functions for QSAR modeling and analysis which are specific to the field

Includes functions for:
    - calculating Q2 scoring metric
    - calculating Q2_f3 scoring metric
    - calculating CCC (concordance correlation coefficient) index
"""
def q2_score(y_true, y_pred):
    """
    Calculates the Q2 score

    Parameters
    ----------
    y_true : pandas DataFrame, shape (n_samples,)
    y_pred : pandas DataFrame, shape (n_samples,)

    Returns
    -------
    float
    """
    press = mean_squared_error(y_true, y_pred)
    tss = mean_squared_error(y_true, repeat(y_true.mean()))
    return 1 - press/tss

def q2f3_score(y_true, y_pred):
    """
    Calculates the Q2_f3 score

    Parameters
    ----------
    y_true : pandas DataFrame, shape (n_samples,)
    y_pred : pandas DataFrame, shape (n_samples,)
    n_external : int
        number of external samples
    n_train : int
        number of training samples

    Returns
    -------
    float
    """
    n_train = len(y_true.index)
    n_external = len(y_pred.index)
    press = mean_squared_error(y_true, y_pred)
    tss = mean_squared_error(y_true, repeat(y_true.mean()))
    return 1 - (press / n_external) / (tss * n_train)

def ccc_score(y_true, y_pred):
    """
    Calculates the CCC score

    Parameters
    ----------
    y_true : pandas DataFrame, shape (n_samples,)
    y_pred : pandas DataFrame, shape (n_samples,)

    Returns
    -------
    float
    """
    mean_true = y_true.mean()
    mean_pred = y_pred.mean()
    var_true = y_true.var()
    var_pred = y_pred.var()
    covar_true_pred = np.cov(y_true, y_pred)[0,1]
    return 2 * covar_true_pred / (var_true + var_pred + (mean_true - mean_pred)**2)
