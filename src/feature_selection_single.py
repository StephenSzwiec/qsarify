#-*- coding: utf-8 -*-
# Author: Stephen Szwiec
# Date: 2023-02-19
# Description: Single-Threaded Feature Selection Module
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
Single-Threaded Feature Selection Module

This module contains the single-threaded version of the feature selection algorithm,
which is a genetic algorithm that uses a linear regression model to score each set of features,
using the output of clustering to ensure that the features are not redundant.

"""
import datetime
import random
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import itertools

def mlr_selection(X_data, y_data, cluster_info, component, model="regression", learning=50000, bank=200, interval=1000):
    """
    Performs feature selection using a using a linear regression model and a genetic algorithm on a single thread.
    This is the vanilla version of the algorithm, which is not parallelized.

    Parameters
    ----------
    X_data: DataFrame, descriptor data
    y_data: DataFrame, target data
    cluster_info: dict, descriptor cluster information
    component: int, number of features to select
    model: str, learning algorithm to use, default = "regression"
    learning: int, number of iterations to perform, default = 50000
    bank: int, number of models to keep in the bank, default = 200
    interval: int, number of iterations to perform before printing the current time, default = 1000

    Returns
    -------
    best_model: list, best model found
    best_score: float, best score found
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
    # models contain 1-component number of features
    # ensure the models are not duplicated and non redundant
    index_sort_bank = set()
    model_bank = [ ini_desc for _ in range(bank) for ini_desc in [sorted([random.choice(cluster[random.choice(num_clusters)]) for _ in range(random.randint(1,component))])] if ini_desc not in tuple(index_sort_bank) and not index_sort_bank.add(tuple(ini_desc))]

    # score each set of features, saving each score and the corresponding feature set
    scoring_bank = list(map(lambda x: [y_mlr.fit(np.array(X_data.loc[:,x]), y_data.values.ravel()).score(np.array(X_data.loc[:,x]), y_data), list(X_data.loc[:,x].columns.values)], model_bank))

    def evolve(i):
        """
        Evolution of descriptors for learning algorithm, implemented as a function map

        Parameters
        ----------
        i: list, descriptor set
        """
        i = i[1]
        group_n = [cluster_info[x]-1 for x in i]
        sw_index = random.randrange(0, len(i))
        sw_group = random.randrange(0, max(nc))
        while sw_group in group_n:
            sw_group = random.randrange(0, max(nc))
        b_set = [random.choice(cluster[sw_group]) if x == sw_index else i[x] for x in range(0, len(i))]
        b_set.sort()
        x = X_data[b_set].values
        y = y_data.values.ravel()
        score = e_mlr.fit(x, y).score(x, y)
        return [score, b_set]

    # perform main learning loop
    for n in range(learning):
        # initialize best score to the worst possible score
        best_score = -float("inf")
        # Evolve the bank of models and allow those surpassing the best score to replace the worst models up to the bank size
        rank_filter = filter(lambda x, best_score=best_score: x[0] > best_score and (best_score := x[0]), map(evolve, scoring_bank))
        scoring_bank = sorted(itertools.chain(scoring_bank, rank_filter), reverse = True)[:bank]
        if n % interval == 0 and n != 0:
            tt = datetime.datetime.now()
            print(n, '=>', tt.strftime('%H:%M:%S'), scoring_bank[0])

    # print output and return best model found during training
    print("Best score: ", scoring_bank[0][0])
    clulog = [cluster_info[y] for y in scoring_bank[0][1]]
    print("Model's cluster info", clulog)
    fi = datetime.datetime.now()
    fiTime = fi.strftime('%H:%M:%S')
    print("Finish Time : ", fiTime)
    return scoring_bank[0][1]

def rf_selection(X_data, y_data, cluster_info, component, model="regression", learning=50000, bank=200, interval=1000):
    """
    Performs feature selection using a using a random forest model and a genetic algorithm on a single thread.
    This is the vanilla version of the algorithm, which is not parallelized.

    Parameters
    ----------
    X_data: DataFrame, descriptor data
    y_data: DataFrame, target data
    cluster_info: dict, descriptor cluster information
    component: int, number of features to select
    model: str, learning algorithm to use, default = "regression"
    learning: int, number of iterations to perform, default = 50000
    bank: int, number of models to keep in the bank, default = 200
    interval: int, number of iterations to perform before printing the current time, default = 1000

    Returns
    -------
    best_model: list, best model found
    best_score: float, best score found
    """

    now = datetime.datetime.now()
    print("Start time: ", now.strftime('%H:%M:%S'))

    if model == "regression":
        print('\x1b[1;42m','Regression','\x1b[0m')
        y_rf = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=0)
        e_rf = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=0)
    else:
        print('\x1b[1;42m','Classification','\x1b[0m')
        y_rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        e_rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

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
    # models contain 1-component number of features
    # ensure the models are not duplicated and non redundant
    index_sort_bank = set()
    model_bank = [ ini_desc for _ in range(bank) for ini_desc in [sorted([random.choice(cluster[random.choice(num_clusters)]) for _ in range(random.randint(1,component))])] if ini_desc not in tuple(index_sort_bank) and not index_sort_bank.add(tuple(ini_desc))]

    # score each set of features, saving each score and the corresponding feature set
    scoring_bank = list(map(lambda x: [y_rf.fit(np.array(X_data.loc[:,x]), y_data.values.ravel()).score(np.array(X_data.loc[:,x]), y_data), list(X_data.loc[:,x].columns.values)], model_bank))

    def evolve(i):
        """
        Evolution of descriptors for learning algorithm, implemented as a function map

        Parameters
        ----------
        i: list, descriptor set
        """
        i = i[1]
        group_n = [cluster_info[x]-1 for x in i]
        sw_index = random.randrange(0, len(i))
        sw_group = random.randrange(0, max(nc))
        while sw_group in group_n:
            sw_group = random.randrange(0, max(nc))
        b_set = [random.choice(cluster[sw_group]) if x == sw_index else i[x] for x in range(0, len(i))]
        b_set.sort()
        x = X_data[b_set].values
        y = y_data.values.ravel()
        score = e_rf.fit(x, y).score(x, y)
        return [score, b_set]

    # perform main learning loop
    for n in range(learning):
        # initialize best score to the worst possible score
        best_score = -float("inf")
        # Evolve the bank of models and allow those surpassing the best score to replace the worst models up to the bank size
        rank_filter = filter(lambda x, best_score=best_score: x[0] > best_score and (best_score := x[0]), map(evolve, scoring_bank))
        scoring_bank = sorted(itertools.chain(scoring_bank, rank_filter), reverse = True)[:bank]
        if n % interval == 0 and n != 0:
            tt = datetime.datetime.now()
            print(n, '=>', tt.strftime('%H:%M:%S'), scoring_bank[0])

    # print output and return best model found during training
    print("Best score: ", scoring_bank[0][0])
    clulog = [cluster_info[y] for y in scoring_bank[0][1]]
    print("Model's cluster info", clulog)
    fi = datetime.datetime.now()
    fiTime = fi.strftime('%H:%M:%S')
    print("Finish Time : ", fiTime)
    return scoring_bank[0][1]
