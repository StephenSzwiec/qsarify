#-*- coding: utf-8 -*-
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

"""
Perform the entire data cleaning process as one function

Parameters
----------
X_data : pandas DataFrame, shape = (n_samples, n_features)
cutoff : float, auto-correlaton coefficient below which we keep

Returns
-------
Modified DataFrame
"""
def clean_data(X_data, cutoff=0.8):
    # Create a deep copy of the data
    df = X_data.copy()
    # Remove columns with NaN values
    df.dropna(axis=1, inplace=True)
    # Remove columns with constant data
    df = df.loc[:, (df != df.iloc[0]).any()]
    # Remove columns with low variance
    df.drop((df.var())[df.var() > cutoff].index, axis=1, inplace=True)
    # Scale the data and return
    return pd.DataFrame(MinMaxScaler().fit_transform(df), columns=list(df.columns.values))

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
    A = X_data.var() >= cutoff
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

"""
Generate a sequential train-test split by sorting the data by response variable

Parameters
----------
X_data : pandas DataFrame , shape = (n_samples, m_features)
y_data : pandas DataFrame , shape = (n_samples, 1)
test_size : float, default = 0.2

Returns
-------
X_train : pandas DataFrame , shape = (n_samples, m_features)
X_test : pandas DataFrame , shape = (n_samples, m_features)
y_train : pandas DataFrame , shape = (n_samples, 1)
y_test : pandas DataFrame , shape = (n_samples, 1)
"""
# Do everything above as a list comprehension
def sorted_split(X_data, y_data, test_size=0.2):
    # every n-th row is a test row, computed from test_size as a fraction
    n = int(1 / test_size)
    # sort the data by response variable
    df = pd.concat([X_data, y_data], axis=1)
    df.sort_values(by=list(y_data.columns.values), inplace=True)
    # return indices of test rows
    test_idx = df.index[::n]
    # return indices of train rows
    train_idx = df.index.difference(test_idx)
    # return train and test data
    return X_data.loc[train_idx], X_data.loc[test_idx], y_data.loc[train_idx], y_data.loc[test_idx]
