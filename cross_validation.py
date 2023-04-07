#-*- coding: utf-8 -*-
# Author: Stephen Szwiec
# Date: 2023-02-19
# Description: Cross Validation Module
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
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
import numpy as np
from qsar_scoring import q2_score, q2f_score, q2f3_score, ccc_score

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

        self.x = X_data.loc[:, feature_set].values
        self.y = y_data.values
        self.mlr.fit(self.x, self.y)
        self.original_coef = self.mlr.coef_
        self.original_intercept = self.mlr.intercept_
        self.original_r2 = self.mlr.score(self.x, self.y)
        self.original_q2 = q2_score(self.y, self.mlr.predict(self.x))

    def loocv(self, verbose=False, show_plots=False):
        """
        Performs leave-one-out cross validation
        
        Parameters
        ----------
        verbose : boolean, if true, performs all scoring functions
        show_plots : boolean, if true, shows plots of validation results

        Returns
        -------
        None
        """

        # create a leave-one-out cross validation iterator of the data
        loo = LeaveOneOut()
        loo.get_n_splits(self.x)
        # placeholder for the predicted values
        y_pred = []
        r2loo = []
        q2loo = []
        q2f1loo = []
        q2f2loo = []
        q2f3loo = []
        cccloo = []
        for train_index, test_index in loo.split(self.x):
            # split the data into training and test sets
            x_train, x_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            mlr_loo = LinearRegression()
            mlr_loo.fit(x_train, y_train)
            # predict the value
            y_pred_test = mlr_loo.predict(x_test)
            y_pred.append(y_pred_test)
            # calculate the r2 score
            r2loo.append(mlr_loo.score(x_test, y_test))
            # calculate the q2 score
            q2loo.append(q2_score(y_test, y_pred_test))
            if verbose:
                # calculate the q2f1 score
                q2f1loo.append(q2f_score(y_test, y_pred_test, np.mean(y_train)))
                # calculate the q2f2 score
                q2f2loo.append(q2f_score(y_test, y_pred_test, np.mean(y_test)))
                # calculate the q2f3 score
                q2f3loo.append(q2f3_score(y_test, y_pred_test, len(y_train), len(y_test)))
                # calculate the ccc score
                cccloo.append(ccc_score(y_test, y_pred_test))
        # calculate the mean of the r2 scores
        r2loo_mean = np.mean(r2loo)
        # calculate the mean of the q2 scores
        q2loo_mean = np.mean(q2loo)
        if verbose:
            # calculate the mean of the q2f1 scores
            q2f1loo_mean = np.mean(q2f1loo)
            # calculate the mean of the q2f2 scores
            q2f2loo_mean = np.mean(q2f2loo)
            # calculate the mean of the q2f3 scores
            q2f3loo_mean = np.mean(q2f3loo)
            # calculate the mean of the ccc scores
            cccloo_mean = np.mean(cccloo)
        # print the results
        print("LOOCV")
        print("Original model")
        print("r2: ", self.original_r2)
        print("q2: ", self.original_q2)
        print("LOO CV model")
        print("r2loo: ", r2loo_mean)
        print("q2loo: ", q2loo_mean)
        if verbose:
            print("q2f1loo: ", q2f1loo_mean)
            print("q2f2loo: ", q2f2loo_mean)
            print("q2f3loo: ", q2f3loo_mean)
            print("cccloo: ", cccloo_mean)
        if show_plots:
            # plot the results
            plt.figure()
            # plot the predictions
            plt.scatter(self.y, y_pred, color='orange')
            # plot the original model
            plt.plot(self.y, self.mlr.predict(self.x), color='blue')
            # plot the line y=x
            plt.plot(self.y, self.y, color='black')
            plt.xlabel('Predicted')
            plt.ylabel('Observed')
            plt.title('LOOCV')
            # legend
            plt.legend(['Original Model', 'LOOCV', 'y vs y line'])
            plt.show()


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
        # create a k-fold cross validation iterator of the data 
        kf = KFold(n_splits=k)
        kf.get_n_splits(self.x)
        # placeholder for the predicted values
        y_pred = []
        r2kf = []
        q2kf = []
        if verbose:
            q2f1kf = []
            q2f2kf = []
            q2f3kf = []
            ccckf = []
        for train_index, test_index in kf.split(self.x):
            # split the data into training and test sets
            x_train, x_test = self.x[train_index], self.x[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            # predict the value
            y_pred_test = self.fit_predict(x_train, y_train)
            y_pred.append(y_pred_test)
            # calculate the r2 score
            r2kf.append(self.mlr.score(x_test, y_test))
            # calculate the q2 score
            q2kf.append(q2_score(y_test, y_pred_test))
            if verbose:
                # calculate the q2f1 score
                q2f1kf.append(q2f_score(y_test, y_pred_test, np.mean(y_train)))
                # calculate the q2f2 score
                q2f2kf.append(q2f_score(y_test, y_pred_test, np.mean(y_test)))
                # calculate the q2f3 score
                q2f3kf.append(q2f3_score(y_test, y_pred_test, len(y_train), len(y_test)))
                # calculate the ccc score
                ccckf.append(ccc_score(y_test, y_pred_test))
        # calculate the mean of the r2 scores
        r2kf_mean = np.mean(r2kf)
        # calculate the mean of the q2 scores
        q2kf_mean = np.mean(q2kf)
        if verbose:
            # calculate the mean of the q2f1 scores
            q2f1kf_mean = np.mean(q2f1kf)
            # calculate the mean of the q2f2 scores
            q2f2kf_mean = np.mean(q2f2kf)
            # calculate the mean of the q2f3 scores
            q2f3kf_mean = np.mean(q2f3kf)
            # calculate the mean of the ccc scores
            ccckf_mean = np.mean(ccckf)
        # print the results
        print("K-Fold CV")
        print("Original model")
        print("r2: ", self.original_r2)
        print("q2: ", self.original_q2)
        print("K-Fold CV model")   
        print("r2: ", r2kf_mean)
        print("q2: ", q2kf_mean)
        if verbose:
            print("q2f1: ", q2f1kf_mean)
            print("q2f2: ", q2f2kf_mean)
            print("q2f3: ", q2f3kf_mean)
            print("ccc: ", ccckf_mean)
        if show_plots:
            # plot the results
            plt.figure()
            # plot the predictions
            plt.scatter(self.y, y_pred, color='orange')
            # plot the original model
            plt.plot(self.y, self.mlr.predict(self.x), color='blue')
            # plot the line y=x
            plt.plot(self.y, self.y, color='black')
            plt.xlabel('Predicted')
            plt.ylabel('Observed')
            plt.title('K-Fold CV')
            # legend
            plt.legend(['Original Model', 'K-Fold CV', 'y vs y line'])
            plt.show()

