#-*- coding: utf-8 -*-
import datetime
import random
import copy
import pandas as pd
import numpy as np
from sklearn import linear_model as lm
from sklearn.svm import SVC
import itertools

"""
Forward feature selection using cophenetically correlated data on a single core

Parameters
----------
X_data : pandas DataFrame , shape = (n_samples, n_features)
y_data : pandas DataFrame , shape = (n_samples,)
cluster_info : dictionary returned by clustering.featureCluster.set_cluster()
model : default="regression", otherwise "classification"
learning : default=500000, number of overall models to be trained
bank : default=200, number of models to be trained in each iteration
component : default=4, number of features to be selected
interval : optional, default=1000, print current scoring and selected features every interval

Returns
-------
list, result of selected best feature set
"""
def selection(X_data, y_data, cluster_info, model="regression", learning=500000, bank=200, component=4, interval=1000):
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

    """
    Evolution of descriptors for learning algorithm, implemented as a function map

    Parameters
    ----------
    j: float, score of the model, unused within body of function
    i: list, descriptor set
    """
    def evolve(i):
        i = i[1]
        group_n = [cluster_info[x]-1 for x in i]
        sw_index = random.randrange(0, len(i))
        sw_group = random.randrange(0, max(nc))
        while sw_group in group_n:
            sw_group = random.randrange(0, max(nc))
        b_set = copy.deepcopy(i)
        b_set[sw_index] = random.choice(cluster[sw_group])
        b_set.sort()
        x = pd.DataFrame(X_data, columns=b_set)
        xx = np.array(x, dtype=np.float64)
        y = np.array(y_data, dtype=np.float64)
        e_mlr.fit(xx, y_data.values.ravel())
        score = e_mlr.score(xx, y)
        return [score, b_set]

    # initialize a best score variable used to compare with the current score
    # perform main learning loop
    for n in range(learning):
        best_score = -float("inf")
        # Evolve the bank of models and allow those surpassing the best score to replace the worst models up to the bank size
        rank_filter = filter(lambda x, best_score=best_score: x[0] > best_score and (best_score := x[0]), map(evolve, scoring_bank))
        scoring_bank = sorted(itertools.chain(scoring_bank, rank_filter), reverse = True)[:bank]
        if n % interval == 0 and n != 0:
            tt = datetime.datetime.now()
            print(n, '=>', tt.strftime('%H:%M:%S'), scoring_bank[0])

    # print output and return best model found during training
    print([0])
    clulog = [cluster_info[y] for _, y in scoring_bank[0][1]]
    print("Model's cluster info", clulog)
    fi = datetime.datetime.now()
    fiTime = fi.strftime('%H:%M:%S')
    print("Finish Time : ", fiTime)
    return scoring_bank[0][1]
