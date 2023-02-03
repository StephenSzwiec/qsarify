#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
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
        self.xcorr = pd.DataFrame(abs(np.corrcoef(self.X_data, rowvar=False)), columns=X_data.columns, index=X_data.columns)
        self.link = link
        self.cut_d = cut_d

    """
    Make cluster of features based on hierarchical clustering method

    Parameters
    ----------
    None

    -------
    cludict : dict, cluster information of features as a dictionary
    """
    def set_cluster(self):
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
        # list comprehension which returns a list of average autocorrelation values for each cluster, unless the cluster length is 1
        # in which case it returns nothing
        dist_box = [ (np.array([self.xcorr.loc[i,i]]).sum() - len(i)/2)/(len(i)**2 - len(i)/2) for i in cluster if len(i) > 1]
        plt.hist(dist_box)
        plt.ylabel("Frequency")
        plt.xlabel("Correlation coefficient of each cluster")
        plt.show()
