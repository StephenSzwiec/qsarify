#-*- coding: utf-8 -*-
# Author: Stephen Szwiec
# Date: 2023-02-19
# Description: Cross Validation Module
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

"""
Cross Validation Module

This module contains the cross_validation class which is used to perform cross validation on a data set using a linear regression model

"""
from sklearn.model_selection import KFold, LeaveOneOut
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from qsarify.qsar_scoring import q2_score, q2f_score, q2f3_score, ccc_score

class cross_validation:

    """
    Class for performing cross validation on a data set using a linear regression model


    initializes a cross_validation object, performs the regression and stores the results

    Parameters
    ----------
    X_data : pandas dataframe, shape = [n_samples, n_features]
    y_data : pandas dataframe, shape = [n_samples, ]
    feature_set : list, set of features to be used for the model
    verbose : boolean, if true, performs all scoring functions
    """
    def __init__ (self, X_data, y_data, feature_set):
        """
        Does the preliminary work of regenerating the original model locally for later use and comparison
        """
        self.mlr = LinearRegression()
        self.feature_set = feature_set
        self.x = X_data.loc[:, feature_set].values
        self.y = y_data.values
        self.mlr.fit(self.x, self.y)
        self.original_coef = self.mlr.coef_
        self.original_intercept = self.mlr.intercept_
        self.original_r2 = self.mlr.score(self.x, self.y)
        self.original_q2 = q2_score(self.y, self.mlr.predict(self.x))



    def kfoldcv(self, k=5, verbose=False, show_plots=False):
        """
        Performs k-fold cross validation

        Parameters
        ----------
        k : int, number of folds
        verbose : boolean, if true, performs all scoring functions
        show_plots : boolean, if true, shows plots of validation results

        Returns
        -------
        None
        """
        # create a k-fold object iterator for data
        kf = KFold(n_splits=k, shuffle=True)
        kf.get_n_splits(self.x)
        # initialize lists to store results
        best_model = []
        predY = np.zeros_like(self.y)
        r2kf = []
        q2kf = []
        interceptkf = []
        coefkf = []
        trainset = []
        testset = []
        if verbose:
            q2f1kf = []
            q2f2kf = []
            q2f3kf = []
            ccckf = []
        # iterate over folds
        for train, test in kf.split(self.x):
            scaler = MinMaxScaler()
            scaler.fit(self.x[train])
            x_train = scaler.transform(self.x[train])
            x_test = scaler.transform(self.x[test])
            # fit model to training data
            clf = LinearRegression()
            clf.fit(x_train, self.y[train])
            predY[train] = clf.predict(x_train)
            predY[test] = clf.predict(x_test)
            rs = clf.score(x_train, self.y[train])
            qs = clf.score(x_test, self.y[test])
            intercept = clf.intercept_
            coef = clf.coef_
            trainset_index = train
            testset_index = test
            if verbose:
                q2f1 = q2f_score(self.y[test], predY[test],  np.mean(self.y[train]))
                q2f2 = q2f_score(self.y[test], predY[test], np.mean(self.y[test]))
                q2f3 = q2f3_score(self.y[test], predY[test], len(train), len(test))
                ccc = ccc_score(self.y, predY)
            # store results
            r2kf.append(rs)
            q2kf.append(qs)
            interceptkf.append(intercept)
            coefkf.append(coef)
            trainset.append(trainset_index)
            testset.append(testset_index)
            if verbose:
                q2f1kf.append(q2f1)
                q2f2kf.append(q2f2)
                q2f3kf.append(q2f3)
                ccckf.append(ccc)
            # calculate average results
            rmse = np.sqrt(mean_squared_error(predY, self.y))
            maxq2 = np.max(np.array(q2kf))
            index = q2kf.index(maxq2)
            # store results
            mid = []
            mid.append(np.mean(np.array(q2kf)))
            mid.append(np.mean(np.array(r2kf)))
            mid.append(rmse)
            mid.append(coefkf[index])
            mid.append(interceptkf[index])
            mid.append(trainset_index[index])
            mid.append(testset_index[index])
            if verbose:
                mid.append(np.mean(np.array(q2f1kf)))
                mid.append(np.mean(np.array(q2f2kf)))
                mid.append(np.mean(np.array(q2f3kf)))
                mid.append(np.mean(np.array(ccckf)))
            best_model.append(mid)

        best_model.sort()
        best = best_model[-1]


        print('R^2CV mean: {:.6}'.format(best[1]))
        print('Q^2CV mean: {:.6}'.format(best[0]))
        print('RMSE CV : {:.6}'.format(best[2]))
        print('Features set =', self.feature_set)
        print('Model coeff = ',best[3])
        print('Model intercept = ',best[4])
        if verbose:
            print('Q^2F1CV mean: {:.6}'.format(best[7]))
            print('Q^2F2CV mean: {:.6}'.format(best[8]))
            print('Q^2F3CV mean: {:.6}'.format(best[9]))
            print('CCC CV : {:.6}'.format(best[10]))


        if show_plots:
            pred_plotY = np.zeros_like(self.y)
            g_mlr = LinearRegression()
            g_mlr.fit(self.x[trainset_index], self.y[trainset_index])
            pred_plotY[trainset_index] = g_mlr.predict(self.x[trainset_index])
            pred_plotY[testset_index] = g_mlr.predict(self.x[testset_index])
            plt.ylabel('Predicted Y')
            plt.xlabel('Actual Y')
            plt.scatter(self.y[trainset_index], pred_plotY[trainset_index], label='Training set', color='grey')
            plt.scatter(self.y[testset_index], pred_plotY[testset_index], label='Test set', color='red')
            plt.plot([self.y.min() , self.y.max()] , [[self.y.min()],[self.y.max()]],"black" )
            plt.show()
