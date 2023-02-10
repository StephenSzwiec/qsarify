#-*- coding: utf-8 -*-
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
import numpy as np

def q2_score(y_true, y_pred):
    """
    Calculates the Q2 score

    Parameters
    ----------
    y_true : numpy array , shape (n_samples,)
    y_pred : numpy array, shape (n_samples,)

    Returns
    -------
    float
    """
    press = np.sum(np.square(y_true - y_pred))
    tss =  np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - press/tss

def q2f_score(y_true, y_pred, y_mean):
    """
    Calculates the Q2_f1 or Q2_f2 score 
    depending on whether the mean is calculated from the training set or the external set

    Parameters
    ----------
    y_true : numpy array, shape (n_samples,)
    y_pred : numpy array, shape (n_samples,)
    y_mean : float
        mean of the training set

    Returns
    -------
    float
    """
    press = np.sum(np.square(y_true - y_pred))
    tss =  np.sum(np.square(y_true - y_mean))
    return 1 - press/tss

def q2f3_score(y_true, y_pred, n_train, n_external):
    """
    Calculates the Q2_f3 score

    Parameters
    ----------
    y_true : numpy array, shape (n_samples,)
    y_pred : numpy array, shape (n_samples,)
    n_external : int
        number of external samples
    n_train : int
        number of training samples

    Returns
    -------
    float
    """
    press = np.sum(np.square(y_true - y_pred))
    # create a sum of the squared differences between the true values and mean value, using a map function
    tss = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - (press / n_external) / (tss * n_train)

def ccc_score(y_true, y_pred):
    """
    Calculates the CCC score

    Parameters
    ----------
    y_true : numpy array, shape (n_samples,)
    y_pred : numpy array, shape (n_samples,)

    Returns
    -------
    float
    """
    mean_true = y_true.mean()
    mean_pred = y_pred.mean()
    var_true = y_true.var()
    var_pred = y_pred.var()
    covar_true_pred = np.cov(y_true, y_pred)[0,1]
    return 2 * covar_true_pred / (var_true + var_pred + (mean_true - mean_pred)**2)

"""
Class for performing cross validation on a data set using a linear regression model
"""
class cross_validation:

    """
    initializes a cross_validation object, performs the regression and stores the results

    Parameters
    ----------
    X_data : pandas dataframe, shape = [n_samples, n_features]
    y_data : pandas dataframe, shape = [n_samples, ]
    exdataX : pandas dataframe, shape = [n_samples, n_features]
    exdataY : pandas dataframe, shape = [n_samples, ]
    feature_set : list, set of features to be used for the model
    verbose : boolean, if true, performs all scoring functions
    """
    def __init__ (self, X_data, y_data, exdataX, exdataY, feature_set, verbose=False):
        """
        Does the preliminary work of regenerating the original model locally for later use and comparison
        """
        self.mlr = LinearRegression()

        self.x = X_data.loc[:, feature_set].values
        self.y = y_data.values
        self.x_ext = exdataX.loc[:, feature_set].values
        self.y_ext = exdataY.values
        self.mlr.fit(self.x, self.y)
        self.original_coef = self.mlr.coef_
        self.original_intercept = self.mlr.intercept_
        self.original_r2 = self.mlr.score(self.x, self.y)
        self.original_q2 = q2_score(self.y, self.mlr.predict(self.x))
        self.original_r2_ext = self.mlr.score(self.x_ext, self.y_ext)
        self.original_q2_ext = q2_score(self.y_ext, self.mlr.predict(self.x_ext))  
        if verbose:
            self.original_q2f1 = q2f_score(self.y, self.mlr.predict(self.x), np.mean(self.y))
            self.original_q2f2 = q2f_score(self.y, self.mlr.predict(self.x), np.mean(self.y_ext))
            self.original_q2f3 = q2f3_score(self.y, self.mlr.predict(self.x), len(self.y), len(self.y_ext))
            self.original_ccc = ccc_score(self.y_ext, self.mlr.predict(self.x_ext))
        self.xx = np.concatenate((self.x, self.x_ext))
        self.yy = np.concatenate((self.y, self.y_ext))   



    def fit_predict(self, split):
        """
        A utility function to perform the regression and returns predictions
        used for functional mapping
        """
        train = list(split[0]) 
        test = list(split[1])
        model = self.mlr.fit(self.xx[train], self.yy[train])
        pred = model.predict(self.xx[train])
        pred_ext = self.mlr.predict(self.xx[test])
        return pred, pred_ext



    def loo_cv(self, verbose=False):
        """
        Perform Leave-One-Out cross validation on the data set.

        Parameters
        ----------
        verbose : boolean, if true, performs all scoring functions, default is False

        Returns
        -------
        None
        """
        # perform leave-one-out cross validation
        loo = LeaveOneOut()
        loo.get_n_splits(self.xx)
        split = loo.split(self.xx)
        pred_y, pred_y_ext = np.array(zip(*[self.fit_predict(s) for s in split]))
        # calculate the scores
        r2 = r2_score(self.yy, pred_y)
        q2 = q2_score(self.yy, pred_y_ext)
        if verbose:
            q2f1 = q2f_score(self.yy, pred_y, np.mean(self.yy))
            q2f2 = q2f_score(self.yy, pred_y, np.mean(self.y_ext))
            q2f3 = q2f3_score(self.yy, pred_y, len(self.yy), len(self.y_ext))
            ccc = ccc_score(self.y_ext, pred_y_ext)
        # print the results
        print('LOO CV Results')
        print('R2 LOOCV  mean: {:.3f} +/- {:.3f}'.format(np.mean(r2), np.std(r2)))
        print('Q2 LOOCV mean: {:.3f} +/- {:.3f}'.format(np.mean(q2), np.std(q2)))
        if verbose:
            print('Q2F1 LOOCV mean: {:.3f} +/- {:.3f}'.format(np.mean(q2f1), np.std(q2f1)))
            print('Q2F2 LOOCV mean: {:.3f} +/- {:.3f}'.format(np.mean(q2f2), np.std(q2f2)))
            print('Q2F3 LOOCV mean: {:.3f} +/- {:.3f}'.format(np.mean(q2f3), np.std(q2f3)))
            print('CCC LOOCV mean: {:.3f} +/- {:.3f}'.format(np.mean(ccc), np.std(ccc)))
        print('Model coefficients: {}'.format(self.original_coef))
        print('Model intercept: {}'.format(self.original_intercept))

        # plot the results
        plt.ylabel("Predicted Y")
        plt.xlabel("Actual Y")
        plt.title("LOO CV")
        plt.scatter(self.y, y_pred, color='grey')
        plt.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], 'k--', lw=4)
        plt.scatter(self.y_ext, y_ext_pred, color='red')
        plt.show()

    def k_fold(self, run=100, k=5, verbose=False):    
        """
        Perform repeated k-fold cross validation on the data set.

        Parameters
        ----------
        run : int, number of times to repeat the k-fold cross validation, default is 100
        k : int, number of folds, default is 5
        verbose : boolean, if true, performs all scoring functions, default is False

        Returns
        -------
        None
        """
        # perform k-fold cross validation
        kf = KFold(n_splits=k, shuffle=True)
        split = kf.split(self.x)
        self.y_pred = np.array(map(self.mlr.predict, split))
        r2 = np.array(map(r2_score, self.y, self.y_pred))
        q2 = np.array(map(q2_score, self.y, self.y_pred))
        rmse = np.array(map(np.sqrt, map(mean_squared_error, self.y, self.y_pred)))
        if verbose:
            q2f1 = np.array(map(q2f_score, self.y, self.y_pred, np.mean(self.y)))
            q2f2 = np.array(map(q2f_score, self.y, self.y_pred, np.mean(self.y_ext)))
            q2f3 = np.array(map(q2f3_score, self.y, self.y_pred, len(self.y), len(self.y_ext)))
            ccc = np.array(map(ccc_score, self.y_ext, self.y_pred_ext))

        # print the results
        print('K-Fold CV Results')
        print('R2 mean: {:.3f} +/- {:.3f}'.format(np.mean(r2), np.std(r2)))
        print('Q2 mean: {:.3f} +/- {:.3f}'.format(np.mean(q2), np.std(q2)))
        print('RMSE mean: {:.3f} +/- {:.3f}'.format(np.mean(rmse), np.std(rmse)))
        if verbose:
            print('Q2F1 mean: {:.3f} +/- {:.3f}'.format(np.mean(q2f1), np.std(q2f1)))
            print('Q2F2 mean: {:.3f} +/- {:.3f}'.format(np.mean(q2f2), np.std(q2f2)))
            print('Q2F3 mean: {:.3f} +/- {:.3f}'.format(np.mean(q2f3), np.std(q2f3)))
            print('CCC mean: {:.3f} +/- {:.3f}'.format(np.mean(ccc), np.std(ccc)))
        print('Features set: {}'.format(self.x.columns))
        print('Model coefficients: {}'.format(self.original_coef))
        print('Model intercept: {}'.format(self.original_intercept))

        # plot the results
        plt.ylabel("Predicted Y")
        plt.xlabel("Actual Y")
        plt.title("K-Fold CV")
        plt.scatter(self.y, self.y_pred, color='grey')
        plt.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], 'k--', lw=4)
        if self.external:
            plt.scatter(self.y_ext, self.y_pred_ext, color='red')
        plt.show()