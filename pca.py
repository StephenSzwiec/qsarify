#-*- encoding: utf-8 -*-
# Author: Stephen Szwiec
# Date: 2023-02-19
# Description: Principal Component Analysis Module
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
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from data_tools import train_scale

def pca(X_data, feature_set=None):
    """
    Perform PCA on the data
    Generate a scatter plot of the data
    Optionally: plot the features_set as a projection within the scatter plot

    Parameters
    ----------
    X_data : pandas DataFrame, shape = (n_samples, n_features)
    feature_set : list, default = None

    Returns
    -------
    None
    """

    # scale the data
    X_data = train_scale(X_data)
    # perform PCA
    pca = PCA(n_components=2)
    pca.fit(X_data)
    # get the principal components
    X_pca = pca.transform(X_data)
    # create the loadings of selected features if feature_set is not None
    if feature_set is not None:
        loadings = np.zeros((2, len(feature_set)))
        for i, feature in enumerate(feature_set):
            loadings[:, i] = pca.components_[:, X_data.columns.get_loc(feature)]
        loadings_norm = loadings / np.linalg.norm(loadings, axis=0)
    # plot the data
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
    # plot the loadings of selected features if feature_set is not None
    if feature_set is not None:
        for i, feature in enumerate(feature_set):
            arrow_length = 0.5 
            arrow_style = '->'
            plt.arrow(0, 0, arrow_length * loadings_norm[0, i], arrow_length * loadings_norm[1, i], color='r')
            plt.text(loadings_norm[0, i] * 1.2, loadings_norm[1, i] * 1.2, feature, color='r')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()
    