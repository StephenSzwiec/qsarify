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
import feature_selection_single as fss

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
                merger = Parallel(n_jobs=n_core)(delayed(ffs.selection)(self.X_data, self.y_data, self.cluster_info, self.model, self.learning, self.bank, self.component) for i in range(run_job))
                merger.sort(reverse=True)
                for m in merger:
                    print(m)
                return merger[0][1]
