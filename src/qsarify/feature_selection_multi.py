#-*- coding: utf-8 -*-
# Author: Stephen Szwiec
# Date: 2023-02-19
# Description: Multi-Processing Feature Selection Module
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


"""
Multi-Processing Feature Selection Module

This module contains the functions for performing feature selection using
the clustering module's output as a guide for feature selection, and implements
a genetic algorithm for feature selection using reflection.

"""

import datetime
import random
import numpy as np
from sklearn import linear_model as lm
from sklearn.svm import SVC
import itertools
import multiprocessing as mp

"""
Reflector class for the evolve function; allows for the use of a pool of workers.
"""
class Evolution:
    """
    Initializes the evolution class with the learning algorithm to be used
    """
    def __init__(self, evolve):
        self.e_mlr = lm.LinearRegression()
        self.evolve = evolve

    """
    Function call for the evolution function
    """
    def __call__(self, i, cluster_info, cluster, X_data, y_data):
        return self.evolve(i, cluster_info, cluster, X_data, y_data, self.e_mlr)

    """
    Evolution of descriptors for learning algorithm, implemented as a function map

    Parameters
    ----------
    i: list, descriptor set
    cluster_info: dict, descriptor cluster information
    cluster: list, descriptor cluster
    X_data: DataFrame, descriptor data
    y_data: DataFrame, target data
    """
    def evolve(i, cluster_info, cluster, X_data, y_data, e_mlr):
        # Get the descriptors in the model
        i = i[1]
        # Get the groups of descriptors in model
        group_n = [cluster_info[x]-1 for x in i]
        # randomly select one descriptor to remove
        sw_index = random.randrange(0, len(i))
        # randomly select new group from cluster to swap with
        sw_group = random.randrange(0, max(list(cluster_info.values())))
        while sw_group in group_n:
            # make sure the new group is not in the current group
            sw_group = random.randrange(0, len(cluster))
        # list comprehension which generates a new list of descriptors by
        # swapping the indexed descriptor with a new one randomly chosen from the new cluster group
        b_set = [random.choice(cluster[sw_group]) if x == sw_index else i[x] for x in range(0, len(i))]
        b_set.sort()
        x = X_data[b_set].values
        y = y_data.values.ravel()
        score = e_mlr.fit(x, y).score(x, y)
        return [score, b_set]

def selection(X_data, y_data, cluster_info, model="regression", learning=500000, bank=200, component=4, interval=1000, cores=(mp.cpu_count()*2)-1):
    """
    Forward feature selection using cophenetically correlated data on mutliple cores

    Parameters
    ----------
    X_data : pandas DataFrame , shape = (n_samples, n_features)
    y_data : pandas DataFrame , shape = (n_samples,)
    cluster_info : dictionary returned by clustering.featureCluster.set_cluster()
    model : default="regression", otherwise "classification"
    learning : default=500000, number of overall models to be trained
    bank : default=200, number of models to be trained in each iteration
    component : default=4, number of features to be selected
    interval : optional, default=1000, print current scoring and selected features
               every interval
    cores: optional, default=(mp.cpu_count()*2)-1, number of processes to be used
           for multiprocessing; default is twice the number of cores minus 1, which
           is assuming you have SMT, HT, or something similar) If you have a large
           number of cores, you may want to set this to a lower number to avoid
           memory issues.

    Returns
    -------
    list, result of selected best feature set
    """
    now = datetime.datetime.now()
    print("Start time: ", now.strftime('%H:%M:%S'))

    if model == "regression":
        print('\x1b[1;42m','Regression','\x1b[0m')
        y_mlr = lm.LinearRegression()
        e_mlr = lm.LinearRegression()
    else:
        print('\x1b[1;42m','Classification','\x1b[0m')
        y_mlr = SVC(kernel='rbf', C=1, gamma=0.1, random_state=0)
        e_mlr = SVC(kernel='rbf', C=1, gamma=0.1, random_state=0)

    # a list of numbered clusters
    nc = list(cluster_info.values())
    num_clusters = list(range(max(nc)))

    # extract information from dictionary by inversion
    inv_cluster_info = dict()
    for k, v in cluster_info.items():
        inv_cluster_info.setdefault(v, list()).append(k)

    # an ordered list of features in each cluster
    cluster = list(dict(sorted(inv_cluster_info.items())).values())

    # fill the interation bank with random models
    # models contain (1 - component) number of features
    # ensure the models are not duplicated and non redundant
    index_sort_bank = set()
    model_bank = [ ini_desc for _ in range(bank) for ini_desc in [sorted([random.choice(cluster[random.choice(num_clusters)]) for _ in range(random.randint(1,component))])] if ini_desc not in tuple(index_sort_bank) and not index_sort_bank.add(tuple(ini_desc))]

    # score each set of features, saving each score and the corresponding feature set
    scoring_bank = list(map(lambda x: [y_mlr.fit(np.array(X_data.loc[:,x]), y_data.values.ravel()).score(np.array(X_data.loc[:,x]), y_data), list(X_data.loc[:,x].columns.values)], model_bank))

    # create a reflection of the evolution function
    evolver = Evolution(Evolution.evolve)

    with mp.Pool(processes = cores) as pool:
        # perform main learning loop
        for n in range(learning):
            # initialize best score to the worst possible score
            best_score = -float("inf")
            # Evolve the bank of models and allow those surpassing the best score to replace the worst models up to the bank size
            results = pool.starmap(evolver, [(i, cluster_info, cluster, X_data, y_data) for i in scoring_bank])
            rank_filter = [x for x in results if (best_score := max(best_score, x[0])) == x[0]]
            scoring_bank = sorted(itertools.chain(scoring_bank, rank_filter), reverse = True)[:bank]
            if n % interval == 0 and n != 0:
                tt = datetime.datetime.now()
                print(n, '=>', tt.strftime('%H:%M:%S'), scoring_bank[0])


    # print output and return best model found during training
    print("Best score: ", scoring_bank[0])
    clulog = [cluster_info[y] for _, y in scoring_bank[0][1]]
    print("Model's cluster info", clulog)
    fi = datetime.datetime.now()
    fiTime = fi.strftime('%H:%M:%S')
    print("Finish Time : ", fiTime)
    return scoring_bank[0][1]
