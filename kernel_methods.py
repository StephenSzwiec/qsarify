#-*- encoding utf-8 -*-
# Author: Stephen Szwiec
# Date: 2023-02-19
# Description: Kernel Methods Module 
"""
Copyright (C) 2023 Stephen Szwiec

This file is part of pyqsarplus.

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

def epanechnikov(t):
    """
    Returns the Epanechnikov kernel function evaluated at t

    Parameters
    ----------
    t : numpy array, shape (n_samples,)

    Returns
    -------
    numpy array, shape (n_samples,)

    """
    return np.where(np.abs(t) <= 1, 0.75*(1-t**2), 0)

def gaussian(t):
    """ 
    Returns the Gaussian kernel function evaluated at t

    Parameters
    ----------
    t : numpy array, shape (n_samples,)

    Returns
    -------
    numpy array, shape (n_samples,)

    """
    return (1/np.sqrt(2*np.pi))*np.exp(-0.5*t**2)

def laplacian(t):
    """ 
    Returns the Laplacian kernel function evaluated at t

    Parameters
    ----------
    t : numpy array, shape (n_samples,)

    Returns
    -------
    numpy array, shape (n_samples,)

    """
    return np.where(np.abs(t) <= 1, 0.5*np.exp(-np.abs(t)), 0)

def cosine(t):
    """ 
    Returns the Cosine kernel function evaluated at t

    Parameters
    ----------
    t : numpy array, shape (n_samples,)

    Returns
    -------
    numpy array, shape (n_samples,)

    """
    return np.where(np.abs(t) <= 1, np.pi/4*np.cos(np.pi*t/2), 0)

def logistic(t):
    """ 
    Returns the Logistic kernel function evaluated at t

    Parameters
    ----------
    t : numpy array, shape (n_samples,)

    Returns
    -------
    numpy array, shape (n_samples,)

    """
    return 1/(np.exp(t) + 2 + np.exp(-t))

def sigmoid(t):
    """ 
    Returns the Sigmoid kernel function evaluated at t

    Parameters
    ----------
    t : numpy array, shape (n_samples,)

    Returns
    -------
    numpy array, shape (n_samples,)

    """
    return 2/(1 + np.exp(-t))

def kernel_weighted_polynomial_regressor(x, y, degree=2, kernel=gaussian, h=1):
    """ 
    Returns the kernel weighted polynomial regressor

    Parameters
    ----------
    x : numpy array, shape (n_samples,)
    y : numpy array, shape (n_samples,)
    degree : int, default=2, degree of polynomial
    kernel : function, default=gaussian, kernel function to use
    h : float, default=1, bandwidth parameter

    Returns
    -------

    """
    n = x.shape[0]
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = kernel((x[i]-x[j])/h)
    X = np.zeros((n, degree+1))
    for i in range(n):
        for j in range(degree+1):
            X[i,j] = x[i]**j
    return np.linalg.inv(X.T.dot(K).dot(X)).dot(X.T).dot(K).dot(y)

def kernel_weighted_polynomial_classifier(x, y, degree=2, kernel=gaussian, h=1):
    """ 
    Returns the kernel weighted polynomial classifier

    Parameters
    ----------
    x : numpy array, shape (n_samples,)
    y : numpy array, shape (n_samples,)
    degree : int, default=2, degree of polynomial
    kernel : function, default=gaussian, kernel function to use
    h : float, default=1, bandwidth parameter, a radius of smoothing

    Returns
    -------
    numpy array, shape (n_samples,)

    """
    n = x.shape[0]
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = kernel((x[i]-x[j])/h)
    X = np.zeros((n, degree+1))
    for i in range(n):
        for j in range(degree+1):
            X[i,j] = x[i]**j
    return np.sign(np.linalg.inv(X.T.dot(K).dot(X)).dot(X.T).dot(K).dot(y))
