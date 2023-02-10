#-*- coding: utf-8 -*-
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
    cutoff : float, default = 0.9

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
    corr_mtx = abs(X_data.corr())
    A = corr_mtx.isnull().sum()
    return X_data.drop(X_data.columns[A], axis=1)

def train_scale(X_data):
    """
    Transform features by scaling each feature to a given range

    Parameters
    ----------
    X_data : pandas DataFrame , shape = (n_samples, n_features)

    Returns
    -------
    X_scaled : pandas DataFrame , shape = (n_samples, n_features)
    """
    header = list(X_data.columns.values)
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_data), columns=header)
    return X_scaled

def clean_data(X_data, cutoff=None, plot=False):
    """
    Perform the entire data cleaning process as one function
    Optionally, plot the correlation matrix

    Parameters
    ----------
    X_data : pandas DataFrame, shape = (n_samples, n_features)
    cutoff : float, optional, auto-correlaton coefficient below which we keep

    Returns
    -------
    Modified DataFrame
    """
    # Create a deep copy of the data
    df = X_data.copy()
    # Remove columns with constant data
    df = rm_constant(df)
    # Remove columns with NaN values
    df = rm_nan(df)
    # Remove columns with low variance
    if cutoff: 
        df = rm_lowVar(df, cutoff)
    # Scale the data and return
    if plot:
        plt.matshow(df.corr())
        plt.set_cmap('seismic')
        plt.show()
    return pd.DataFrame(MinMaxScaler().fit_transform(df), columns=list(df.columns.values))


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
    -------
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
