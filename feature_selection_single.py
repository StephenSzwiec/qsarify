#-*- coding: utf-8 -*-
import datetime
import random
import copy
import math
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import sklearn.linear_model as lm
from sklearn import preprocessing, datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVC
from joblib import Parallel, delayed
import itertools


def selection(X_data, y_data, cluster_info, model="regression", learning=50000, bank=200, component=4, interval=1000):
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


class MultiSelection :
    """
    This class is used to select features by using multi-core processing using the same method as feature_selection_single.py.

    Parameters
    ----------
    X_data : pandas DataFrame , shape = (n_samples, n_features)
    y_data : pandas DataFrame , shape = (n_samples,)
    cluster_info : Return value of clustering.FeatureCluster.set_cluster()
    model : default='regression' / when user use default value, operate this function for regression model /
    when user use other value like 'Classification', operate this function for classification model/
    This function only have two options : regression(default) or classification
    learning : Number of learning
    bank : Size of bank(pool) in selection operation
    component : Number of features in prediction model

    Sub functions
    -------
    run (self, n_core=3, run_job=3)
    """

    def __init__(self, X_data, y_data, cluster_info, model='regression', learning=10000, bank=100, component=3) :
        self.X_data = X_data
        self.y_data = y_data
        self.cluster_info = cluster_info
        self.model = model
        self.learning = learning
        self.bank = bank
        self.component = component
        if self.model == 'regression' :
            print('\x1b[1;42m'+'Regression'+'\x1b[0m')
        else :
            print('\x1b[1;42m'+'Classification'+'\x1b[0m')

    def run (self, n_core=3, run_job=3) :
        """
        Parameters
        ----------
        n_core : Number of used cores
        run_job : Number of times to perform the selection function

        Returns
        -------
        list, result of selected best Feature set
        """

        def __init__(self, X_data,y_data,cluster_info,model='regression',learning=5,bank=100,component=3) :
            self.X_data = X_data
            self.y_data = y_data
            self.cluster_info= cluster_info
            self.learning = learning
            self.bank = bank
            self.component =component
            self.model = model
            if self.model == 'regression' :
                print('\x1b[1;42m'+'Regression'+'\x1b[0m')
            else :
                print('\x1b[1;42m'+'Classification'+'\x1b[0m')

            def run(self, n_core=3, run_job=3) :
                """
                Parameters
                ----------
                n_core : Number of used cores
                run_job : Number of times to perform the selection function

                Returns
                -------
                list, result of selected best Feature set
                """
                merger = Parallel(n_jobs=n_core)(delayed(selection)(self.X_data, self.y_data, self.cluster_info, self.model, self.learning, self.bank, self.component) for i in range(run_job))
                merger.sort(reverse=True)
                for m in merger:
                    print(m)
                return merger[0][1]
