#-*- coding: utf-8 -*-
# Author: Stephen Szwiec
# Date: 2023-02-19
# Description: Data Preprocessing Module
#
#Copyright (C) 2023 Stephen Szwiec
#
#This file is part of qsarify.
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#

"""
Data Preprocessing Module

This module contains functions for data preprocessing, including:
    - removing features with 'NaN' as value
    - removing features with constant values
    - removing features with low variance
    - removing features with 'NaN' as value when calculating correlation coefficients
    - generating a sequential train-test split by sorting the data by response variable
    - generating a random train-test split
    - scaling data

The main function of this module is `clean_data`, which performs all of the above functions.

"""



import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def rm_nan(X_data):
    """
    Remove features with 'NaN' as value

    Parameters
    ----------
    X_data : pandas DataFrame , shape = (n_samples, n_features)

    Returns
    -------
    Modified DataFrame
    """
    # get the indices of the features with 'NaN' as value
    A = X_data.isnull().any()
    # delete the features with 'NaN' as value
    return X_data.drop(X_data.columns[A], axis=1)

def rm_constant(X_data):
    """
    Remove features with constant values

    Parameters
    ----------
    X_data : pandas DataFrame , shape = (n_samples, n_features)

    Returns
    -------
    Modified DataFrame
    """
    A = X_data.std() == 0
    return X_data.drop(X_data.columns[A], axis=1)

def rm_lowVar(X_data, cutoff=0.9):
    """
    Remove features with low variance

    Parameters
    ----------
    X_data : pandas DataFrame , shape = (n_samples, n_features)
    cutoff : float, default = 0.1

    Returns
    -------
    Modified DataFrame
    """
    A = X_data.var() >= cutoff
    return X_data.drop(X_data.columns[A], axis=1)

def rm_nanCorr(X_data):
    """
    Remove features with 'NaN' as value when calculating correlation coefficients

    Parameters
    ----------
    X_data : pandas DataFrame , shape = (n_samples, n_features)

    Returns
    -------
    Modified DataFrame
    """
    corr_mtx = pd.DataFrame(np.corrcoef(X_data, rowvar=False), columns=X_data.columns, index=X_data.columns)
    A = corr_mtx.isnull().any()
    return X_data.drop(X_data.columns[A], axis=1)


def sorted_split(X_data, y_data, test_size=0.2):
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
    X_test : pandas DataFrame, shape = (n_samples, m_features)
    y_train : pandas DataFrame , shape = (n_samples, 1)
    y_test : pandas DataFrame , shape = (n_samples, 1)
    """
    # every n-th row is a test row, computed from test_size as a fraction
    n = int(1 / test_size)
    # sort by response variable
    df = pd.concat([X_data, y_data], axis=1)
    df.sort_values(by=y_data.name, inplace=True)
    test_idx = df.index[::n]
    train_idx = df.index.difference(test_idx)
    # return train and test data
    return X_data.loc[train_idx], X_data.loc[test_idx], y_data.loc[train_idx], y_data.loc[test_idx]

def random_split(X_data, y_data, test_size=0.2):
    """
    Generate a random train-test split

    Parameters
    ----------
    X_data : pandas DataFrame , shape = (n_samples, m_features)
    y_data : pandas DataFrame , shape = (n_samples, 1)
    test_size : float, default = 0.2

    Returns
    -------give count of NaN in pandas dataframe
    X_train : pandas DataFrame , shape = (n_samples, m_features)
    X_test : pandas DataFrame , shape = (n_samples, m_features)
    y_train : pandas DataFrame , shape = (n_samples, 1)
    y_test : pandas DataFrame , shape = (n_samples, 1)
    """
    # every n-th row is a test row, computed from test_size as a fraction
    n = int(1 / test_size)
    # return indices of test rows
    test_idx = np.random.choice(X_data.index, size=int(len(X_data) * test_size), replace=False)
    # return indices of train rows
    train_idx = X_data.index.difference(test_idx)
    # return train and test data
    return X_data.loc[train_idx], X_data.loc[test_idx], y_data.loc[train_idx], y_data.loc[test_idx]

def scale_data(X_train, X_test):
    """
    Scale the data using the training data; apply the same transformation to the test data

    Parameters
    ----------
    X_train : pandas DataFrame , shape = (n_samples, m_features)
    X_test : pandas DataFrame , shape = (p_samples, m_features)

    Returns
    -------
    X_train_scaled : pandas DataFrame , shape = (n_samples, m_features)
    X_test_scaled : pandas DataFrame , shape = (p_samples, m_features)
    """

    # scale the data
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=list(X_train.columns.values))
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=list(X_test.columns.values))
    return X_train_scaled, X_test_scaled

def clean_data(X_data, y_data, split='sorted', test_size=0.2, cutoff=None, plot=False):
    """
    Perform the entire data cleaning process as one function
    Optionally, plot the correlation matrix

    Parameters
    ----------
    X_data : pandas DataFrame, shape = (n_samples, n_features)
    split : string, optional, 'sorted' or 'random'
    test_size : float, optional, default = 0.2
    cutoff : float, optional, auto-correlaton coefficient below which we keep
    plot : boolean, optional, default = False

    Returns
    -------
    X_train : pandas DataFrame , shape = (n_samples, m_features)
    X_test : pandas DataFrame , shape = (p_samples, m_features)
    y_train : pandas DataFrame , shape = (n_samples, 1)
    y_test : pandas DataFrame , shape = (p_samples, 1)


    """
    # Create a deep copy of the data
    df = X_data.copy()
    # Remove columns with constant data
    df = rm_constant(df)
    # Remove columns with NaN values
    df = rm_nan(df)
    # Remove columns with NaN values when calculating correlation coefficients
    df = rm_nanCorr(df)
    # Remove columns with low variance
    if cutoff:
        df = rm_lowVar(df, cutoff)
    # Create split
    if split == 'random':
        X_train, X_test, y_train, y_test = random_split(df, y_data, test_size)
    else:
        X_train, X_test, y_train, y_test = sorted_split(df, y_data, test_size)
    # Scale the data and return
    X_train, X_test = scale_data(X_train, X_test)
    if plot:
        plt.matshow(df.corr())
        plt.set_cmap('seismic')
        # show legend for the matrix
        plt.colorbar()
        plt.show()
    return X_train, X_test, y_train, y_test
