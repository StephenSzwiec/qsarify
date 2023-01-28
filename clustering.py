#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet

"""
Calculate the cophenetic correlation coefficient of linkages

Parameters
----------
X_data : pandas DataFrame, shape = (n_samples, n_features)

Returns
-------
None
"""
def cophenetic(X_data):
    xcorr = abs(X_data.corr())
    Z1 = linkage(xcorr, method='average', metric='euclidean')
    Z2 = linkage(xcorr, method='complete', metric='euclidean')
    Z3 = linkage(xcorr, method='single', metric='euclidean')
    c1, coph_dists1 = cophenet(Z1, pdist(xcorr))
    c2, coph_dists2 = cophenet(Z2, pdist(xcorr))
    c3, coph_dists3 = cophenet(Z3, pdist(xcorr))
    print("cophenetic correlation average linkage: ", c1)
    print("cophenetic correlation complete linkage: ", c2)
    print("cophenetic correlation single linkage: ", c3)

"""
Make cluster of features based on hierarchical clustering method

Parameters
----------
X_data : pandas DataFrame, shape = (n_samples, n_features)
link : str, kind of linkage method, default = 'average'
cut_d : int, depth in cluster(dendrogram), default = 3

Sub functions
-------------
set_cluster(self)
cluster_dist(self)
"""
class featureCluster:
    def __init__(self, X_data, link='average', cut_d=3):
        self.cluster_info = []
        self.assignments = np.array([])
        self.cluster_output = DataFrame()
        self.cludict = {}
        self.X_data = X_data
        self.xcorr = abs(X_data.corr())
        self.link = link
        self.cut_d = cut_d

    """
    Make cluster of features based on hierarchical clustering method

    Parameters
    ----------
    None

    Returns
    -------
    cludict : dict, cluster information of features as a dictionary
    """
    def set_cluster(self):
        Z = linkage(self.xcorr, method=self.link, metric='euclidean')
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
            print('\n','\x1b[1;46m'+'Cluster'+'\x1b[0m',t,self.cluster_info[t-1],)
        return self.cludict

    """
    Show dendrogram of hierarchical clustering

    Returns
    -------
    None
    """
    def cluster_dist(self):
        # have we actually clustered? If not, please do so first:
        if self.assignments.any() == False:
            self.set_cluster()
        nc = list(self.cluster_output.cluster.values)
        cluster = [[k for k, value in self.cludict.items() if value == t] for t in range(1, max(nc)+1)]
        dist_box = list(map(lambda x: (np.array(self.xcorr.loc[x,x]).sum() - len(x)) / (len(x)**2 - len(x)) if (len(x) != 1) else np.nan, cluster))
        plt.hist(dist_box)
        plt.ylabel("Frequency")
        plt.xlabel("Correlation coefficient of each cluster")
        plt.show()
