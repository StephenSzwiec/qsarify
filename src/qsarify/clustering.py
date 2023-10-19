#-*- coding: utf-8 -*-
# Author: Stephen Szwiec
# Date: 2023-02-19
# Description: Clustering Module
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
Clustering Module

This module contains functions for clustering features based on hierarchical clustering method
and calculating the cophenetic correlation coefficient of linkages. The cophenetic correlation
coefficient is a measure of the correlation between the distance of observations in feature space
and the distance of observations in cluster space. The cophenetic correlation coefficient is
calculated for each linkage method and the method with the highest cophenetic correlation
coefficient is used to cluster the features. The cophenetic correlation coefficient is calculated
using the scipy.cluster.hierarchy.cophenet function.

"""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet

def cophenetic(X_data):
    """
    Calculate the cophenetic correlation coefficient of linkages

    Parameters
    ----------
    X_data : pandas DataFrame, shape = (n_samples, m_features)
    method : str, method for linkage generation, default = 'corr' (Pearson correlation)

    Returns
    -------
    None
    """
    distance = abs(np.corrcoef(X_data, rowvar=False))
    # drop any columns and rows that produced NaNs
    distance = distance[~np.isnan(distance).any(axis=1)]
    distance = distance[:, ~np.isnan(distance).any(axis=0)]
    # calculate the cophenetic correlation coefficient
    Z1 = linkage(distance, method='average', metric='euclidean')
    Z2 = linkage(distance, method='complete', metric='euclidean')
    Z3 = linkage(distance, method='single', metric='euclidean')
    c1, coph_dists1 = cophenet(Z1, pdist(distance))
    c2, coph_dists2 = cophenet(Z2, pdist(distance))
    c3, coph_dists3 = cophenet(Z3, pdist(distance))
    print("cophenetic correlation average linkage: ", c1)
    print("cophenetic correlation complete linkage: ", c2)
    print("cophenetic correlation single linkage: ", c3)

class featureCluster:
    """
    Make cluster of features based on hierarchical clustering method

    Parameters
    ----------
    X_data : pandas DataFrame, shape = (n_samples, n_features)
    link : str, kind of linkage method, default = 'average', 'complete', 'single'
    cut_d : int, depth in cluster(dendrogram), default = 3

    Sub functions
    -------------
    set_cluster(self)
    cluster_dist(self)
    """

    def __init__(self, X_data, method='corr', link='average', cut_d=3):
        """
        Initializes cluster object:
        Makes a cluster of features based on hierarchical clustering method
        and calculates the cophenetic correlation coefficient of linkages

        Parameters
        ----------
        X_data : pandas DataFrame, shape = (n_samples, n_features)
        link : str, kind of linkage method, default = 'average', 'complete', 'single'
        cut_d : int, depth in cluster(dendrogram), default = 3
                This is a tunable parameter for clustering
        """
        self.method = method
        self.cluster_info = []
        self.assignments = np.array([])
        self.cluster_output = DataFrame()
        self.cludict = {}
        self.X_data = X_data
        self.link = link
        self.cut_d = cut_d
        self.xcorr = pd.DataFrame(abs(np.corrcoef(self.X_data, rowvar=False)), columns=X_data.columns, index=X_data.columns)

    def set_cluster(self, verbose=False, graph=False):
        """
        Make cluster of features based on hierarchical clustering method

        Parameters
        ----------
        verbose : bool, print cluster information, default = False
        graph : bool, show dendrogram, default = False

        Returns
        -------
        cludict : dict, cluster information of features as a dictionary
        """
        Z = linkage( self.xcorr, method=self.link, metric='euclidean')
        self.assignments = fcluster(Z, self.cut_d, criterion='distance')
        self.cluster_output = DataFrame({'Feature':list(self.X_data.columns.values), 'cluster':self.assignments})
        nc = list(self.cluster_output.cluster.values)
        name = list(self.cluster_output.Feature.values)
        # zip cluster number and feature name
        self.cludict = dict(zip(name, nc))
        # make cluster information as an input for feature selection function
        # print cluster information for key in cludict.items if range of cluster number is 1~nnc
        for t in range(1, max(nc)+1):
            self.cluster_info.append( [k for k, v in self.cludict.items() if v == t] )
            if verbose:
                print('\n','\x1b[1;46m'+'Cluster'+'\x1b[0m',t,self.cluster_info[t-1],)
        if graph:
            plt.figure(figsize=(25, 40))
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('sample index')
            plt.ylabel('distance')
            dendrogram(Z, color_threshold=self.cut_d, above_threshold_color='k', no_labels=True, leaf_label_func=None, show_contracted=True, orientation='left')
            plt.show()
        return self.cludict

    def cluster_dist(self):
        """
        Show dendrogram of hierarchical clustering

        Returns
        -------
        None
        """

        # have we actually clustered? If not, please do so first:
        if self.assignments.any() == False:
            self.set_cluster()
        nc = list(self.cluster_output.cluster.values)
        cluster = [[k for k, value in self.cludict.items() if value == t] for t in range(1, max(nc)+1)]
        # list comprehension which returns a list of average autocorrelation values for each cluster, unless the cluster length is 1
        # in which case it returns nothing
        dist_box = [ (np.array([self.xcorr.loc[i,i]]).sum() - len(i)/2)/(len(i)**2 - len(i)/2) for i in cluster if len(i) > 1]
        plt.hist(dist_box)
        plt.ylabel("Frequency")
        if self.method == 'info':
            plt.xlabel("Shannon mutual information of each cluster")
        else:
            plt.xlabel("Correlation coefficient of each cluster")
        plt.show()
