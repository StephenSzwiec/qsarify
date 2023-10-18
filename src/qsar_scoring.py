# -*- coding: utf-8 -*-
# Author: Stephen Szwiec
# Date: 2023-02-19
# Description: QSAR Scoring Module
"""
Copyright (C) 2023 Stephen Szwiec

This file is part of qsarify.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import numpy as np
"""
Commonly used scoring functions for QSAR models
"""

def rmse_score(y_true, y_pred):
    """
    Calculates the RMSE score

    Parameters
    ----------
    y_true : numpy array , shape (n_samples,)
    y_pred : numpy array, shape (n_samples,)

    Returns
    -------
    float
    """
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

def q2_score(y_true, y_pred):
    """
    Calculates the Q2 score

    Parameters
    ----------
    y_true : numpy array , shape (n_samples,)
    y_pred : numpy array, shape (n_samples,)

    Returns
    -------
    float
    """
    press = np.sum(np.square(y_true - y_pred))
    tss =  np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - press/tss

def q2f_score(y_true, y_pred, y_mean):
    """
    Calculates the Q2_f1 or Q2_f2 score
    depending on whether the mean is calculated from the training set or the external set

    Parameters
    ----------
    y_true : numpy array, shape (n_samples,)
    y_pred : numpy array, shape (n_samples,)
    y_mean : float, mean of the training (for q2f1) or test (for q2f2) set

    Returns
    -------
    float
    """
    press = np.sum(np.square(y_true - y_pred))
    tss =  np.sum(np.square(y_true - y_mean))
    return 1 - press/tss

def q2f3_score(y_true, y_pred, n_train, n_external):
    """
    Calculates the Q2_f3 score

    Parameters
    ----------
    y_true : numpy array, shape (n_samples,)
    y_pred : numpy array, shape (n_samples,)
    n_external : int
        number of external samples
    n_train : int
        number of training samples

    Returns
    -------
    float
    """
    press = np.sum(np.square(y_true - y_pred))
    tss = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - (press / n_external) / (tss * n_train)

def ccc_score(y_true, y_pred):
    """
    Calculates the CCC score

    Parameters
    ----------
    y_true : numpy array, shape (n_samples,)
    y_pred : numpy array, shape (n_samples,)

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
