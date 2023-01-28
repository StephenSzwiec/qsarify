#-*- coding: utf-8 -*-
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from qsar_utility_functions import q2_score, q2f3_score, ccc_score
from sklearn.model_selection import train_test_split, LeaveOneOut

class cross_validation:


    def __init__ (self, X_data, y_data, feature_set, exdataX=None, exdataY=None):
        """
        Does the priliminary work of regenerating the original model for later use and comparison
        """

        self.mlr = LinearRegression()

        # are there external data sets? if so, boolean is true
        if ((type(exdataX) is not None) ^ (type(exdataY) is not None)):
            # if only one is present, raise an error
            raise ValueError('Both exdataX and exdataY must be provided.')
        self.external = bool(exdataX)

        self.x = X_data.loc[:, feature_set].values
        self.y = y_data.values

        if self.external:
            self.x_ext = exdataX.loc[:, feature_set].values
            self.y_ext = exdataY.values

        self.mlr.fit(self.x, self.y)
        self.original_coef = self.mlr.coef_
        self.original_intercept = self.mlr.intercept_
        self.original_r2 = self.mlr.score(self.x, self.y)
        self.original_q2 = q2_score(self.y, self.mlr.predict(self.x))
        if self.external:
            self.original_r2_ext = self.mlr.score(self.x_ext, self.y_ext)
            self.original_q2_ext = q2_score(self.y_ext, self.mlr.predict(self.x_ext))
            self.original_ccc_ext = ccc_score(self.y_ext, self.mlr.predict(self.x_ext))

    # a utility function to perform the regression and returns predictions
    # used for functional mapping
    def fit_predict(self, split):
        self.mlr.fit(self.x[split.train], self.y[split.train])
        return self.mlr.predict(self.x[split.test])

    def loo_cv(self):
        """
        Perform Leave-One-Out cross validation on the data set.

        Parameters
        ----------
        X_data : pandas dataframe, shape = [n_samples, n_features]
        y_data : pandas dataframe, shape = [n_samples, ]
        exdataX : optional external dataset, pandas dataframe, shape = [n_samples, n_features]
        exdataY : optional external dataset, pandas dataframe, shape = [n_samples, ]
        feature_set : list, set of features to be used for the model

        Returns
        -------
        None
        """

        # perform leave-one-out cross validation
        loo = LeaveOneOut()
        split = loo.split(self.x)
        y_pred = map(self.mlr.predict, split)
        if self.external:
            y_pred_ext = map(self.fit_predict, split)

        # calculate the performance metrics
        r2 = np.array(map(r2_score, self.y, y_pred))
        q2 = np.array(map(q2_score, self.y, y_pred))
        rmse = np.array(map(np.sqrt, map(mean_squared_error, self.y, y_pred)))
        if self.external:
            r2_ext = np.array(map(r2_score, self.y_ext, y_pred_ext))
            q2_ext = np.array(map(q2_score, self.y_ext, y_pred_ext))
            rmse_ext = np.array(map(np.sqrt, map(mean_squared_error, self.y_ext, y_pred_ext)))

        # print the results
        print('LOO CV Results')
        print('R2 mean: {:.3f} +/- {:.3f}'.format(np.mean(r2), np.std(r2)))
        print('Q2 mean: {:.3f} +/- {:.3f}'.format(np.mean(q2), np.std(q2)))
        print('RMSE mean: {:.3f} +/- {:.3f}'.format(np.mean(rmse), np.std(rmse)))
        if self.external:
            print('External R2 mean: {:.3f} +/- {:.3f}'.format(np.mean(r2_ext), np.std(r2_ext)))
            print('External Q2 mean: {:.3f} +/- {:.3f}'.format(np.mean(q2_ext), np.std(q2_ext)))
            print('External RMSE mean: {:.3f} +/- {:.3f}'.format(np.mean(rmse_ext), np.std(rmse_ext)))
        print('Model coefficients: {}'.format(self.original_coef))
        print('Model intercept: {}'.format(self.original_intercept))

        # plot the results
        plt.ylabel("Predicted Y")
        plt.xlabel("Actual Y")
        plt.title("LOO CV")
        plt.scatter(self.y, self.y_pred, color='greself.y')
        plt.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], 'k--', lw=4)
        if self.external:
            plt.scatter(self.y_ext, self.y_pred_ext, color='red')
        plt.show()

    """
    Perform repeated k-fold cross validation on the data set.

    Parameters
    ----------
    X_data : pandas dataframe, shape = [n_samples, n_features]
    y_data : pandas dataframe, shape = [n_samples, ]
    exdataX : optional external dataset, pandas dataframe, shape = [n_samples, n_features]
    exdataY : optional external dataset, pandas dataframe, shape = [n_samples, ]
    feature_set : list, set of features to be used for the model
    run : int, number of times to repeat the k-fold cross validation
    k : int, number of folds

    Returns
    -------
    None
    """
    def k_fold(self, run=100, k=5):
        # perform k-fold cross validation
        kf = KFold(n_splits=k, shuffle=True)
        split = kf.split(self.x)
        self.y_pred = np.array(map(self.mlr.predict, split))
        r2 = np.array(map(r2_score, self.y, self.y_pred))
        q2 = np.array(map(q2_score, self.y, self.y_pred))
        rmse = np.array(map(np.sqrt, map(mean_squared_error, self.y, self.y_pred)))

        # print the results
        print('K-Fold CV Results')
        print('R2 mean: {:.3f} +/- {:.3f}'.format(np.mean(r2), np.std(r2)))
        print('Q2 mean: {:.3f} +/- {:.3f}'.format(np.mean(q2), np.std(q2)))
        print('RMSE mean: {:.3f} +/- {:.3f}'.format(np.mean(rmse), np.std(rmse)))
        print('Features set: {}'.format(self.x.columns))
        print('Model coefficients: {}'.format(self.original_coef))
        print('Model intercept: {}'.format(self.original_intercept))

        # plot the results
        plt.ylabel("Predicted Y")
        plt.xlabel("Actual Y")
        plt.title("K-Fold CV")
        plt.scatter(self.y, self.y_pred, color='greself.y')
        plt.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], 'k--', lw=4)
        if self.external:
            plt.scatter(self.y_ext, self.y_pred_ext, color='red')
        plt.show()
