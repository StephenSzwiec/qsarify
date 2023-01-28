#-*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

"""
Remove features with 'NaN' as value

Parameters
----------
X_data : pandas DataFrame , shape = (n_samples, n_features)

Returns
-------
Modified DataFrame
"""
def rm_nan(X_data):
    # get the indices of the features with 'NaN' as value
    A = X_data.isnull().any()
    # delete the features with 'NaN' as value
    return X_data.drop(X_data.columns[A], axis=1)


"""
Remove features with constant values

Parameters
----------
X_data : pandas DataFrame , shape = (n_samples, n_features)

Returns
-------
Modified DataFrame
"""
def rm_constant(X_data):
    A = X_data.std() == 0
    return X_data.drop(X_data.columns[A], axis=1)

"""
Remove features with low variance

Parameters
----------
X_data : pandas DataFrame , shape = (n_samples, n_features)
cutoff : float, default = 0.9

Returns
-------
Modified DataFrame
"""
def rm_lowVar(X_data, cutoff=0.9):
    A = X_data.var() < cutoff
    return X_data.drop(X_data.columns[A], axis=1)

"""
Remove features with 'NaN' as value when calculating correlation coefficients

Parameters
----------
X_data : pandas DataFrame , shape = (n_samples, n_features)

Returns
-------
Modified DataFrame
"""
def rm_nanCorr(X_data):
    corr_mtx = abs(X_data.corr())
    A = corr_mtx.isnull().sum()
    return X_data.drop(X_data.columns[A], axis=1)

"""
Transform features by scaling each feature to a given range

Parameters
----------
X_data : pandas DataFrame , shape = (n_samples, n_features)

Returns
-------
X_scaled : pandas DataFrame , shape = (n_samples, n_features)
"""
def train_scale(X_data):
    header = list(X_data.columns.values)
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_data), columns=header)
    return X_scaled
