#-*- coding: utf-8 -*-

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
# utility functions for QSAR specific scoring functions
from qsar_utility_functions import q2_score, q2f3_score, ccc_score

class ModelExport:
    """
    Summary model information, plotting, and exporting

    Parameters
    ----------
    X_data : pandas DataFrame, shape = (n_samples, n_features)
    y_data : pandas DataFrame , shape = (n_samples,)
    feature_set : list, set of features that make up model

    Sub functions
    -------------
    train_plot(self)
    mlr(self)
    features_table(self)
    model_corr(self)
    """

    def __init__(self, X_data, y_data, exXdata, exYdata, feature_set):
        self.x = X_data.loc[:,feature_set].values
        self.y = y_data.values
        self.ex = exXdata.loc[:, feature_set].values
        self.ey = exYdata.values
        self.feature_set = feature_set
        self.lr = LinearRegression()
        self.fit = self.lr.fit(self.x, self.y)
        self.y_pred = self.lr.predict(self.x)
        self.ey_pred = self.lr.predict(self.ex)

    """
    Show training data prediction plot

    Returns
    -------
    None
    """
    def train_plot(self):
        plt.ylabel("Predicted Y")
        plt.xlabel("Actual Y")
        plt.scatter(self.y,self.y_pred,color=['gray'])
        plt.plot([self.y.min() , self.y.max()] , [[self.y.min()],[self.y.max()]],"black" )
        plt.show()

    """
    Show residuals of training plot
    Optionally, show residuals of test plot

    Returns
    -------
    None
    """
    def williams_plot(self):
        # standard deviation of y
        y_std = np.std(self.y)
        residuals = res = (self.y - self.y_pred)/y_std
        residuals_ex = (self.ey - self.ey_pred)/y_std
        print(len(residuals))
        # Calculate the Hat matrix using H = X(XT X)^-1 XT
        H = Hin = self.x @ np.linalg.inv( self.x.T @ self.x ) @ self.x.T
        leverage = leverage_in = np.diag(H)
        Hex = np.dot(self.ex, np.dot(np.linalg.inv(np.dot(self.ex.T, self.ex)), self.ex.T))
        leverage_ex = np.diag(Hex)
        leverage = np.append(leverage, leverage_ex)
        residuals = np.append(residuals, residuals_ex)  # append residuals of test data
        # The applicability domain is defined as the set of points where the residuals are less than 3 standard deviations
        # the max leverage is 3(k+1)/n where k is the number of features and n is the number of training points
        hii = (3 *(len(self.feature_set)+1))/len(self.y)
        # plot the residuals vs the leverage
        l_min = min(leverage)-0.1
        plt.axline(xy1=(l_min,0),slope=0)
        plt.axline(xy1=(l_min,3),slope=0,linestyle="--")
        plt.axline(xy1=(l_min,-3),slope=0,linestyle="--")
        plt.axline(xy1=(hii, -3.5), xy2=(hii, 3.5))
        plt.ylabel("Std. Residuals")
        plt.xlabel(F"Hat Values (h*={hii:.2f})")
        plt.ylim([-3.5,3.5])
        plt.scatter(leverage_in,res,color=['gray'])
        plt.scatter(leverage_ex,residuals_ex,color=['red'])
        plt.plot()
        plt.show()

    """
    Model information with result of multiple linear regression

    Returns
    -------
    None
    """
    def mlr(self) :
        print('Model features: ',self.feature_set)
        print('Coefficients: ', self.fit.coef_)
        print('Intercept: ',self.fit.intercept_)
        #MSE
        #print "MSE: %.3f" % np.mean((mlr.predict(x) - y) ** 2)
        #print mean_squared_error(mlr.predict(x),y)
        print("RMSE: %.6f" % np.sqrt(mean_squared_error(self.y_pred,self.y)))
        # Explained variance score
        print('R^2: %.6f' % r2_score(self.y, self.y_pred))

    """
    Show correlation of features

    Returns
    -------
    table
    """
    def features_table(self) :
        desc = DataFrame(self.x, columns=self.feature_set)
        result = pd.concat([desc, self.y], axis=1, join='inner')
        return result

    """
    Correlation coefficient of features table

    Returns
    -------
    table
    """
    def model_corr(self) :
        X = DataFrame(self.x, columns=self.feature_set)
        result = pd.concat([X, self.x], axis=1, join='inner')
        pd.plotting.scatter_matrix (result, alpha=0.5, diagonal='kde')
        return result.corr()

    """
    Prediction external data set

    Parameters
    ----------
    X_data : pandas DataFrame , shape = (n_samples, n_features)
    y_data : pandas DataFrame , shape = (n_samples,)
    exdataX :pandas DataFrame , shape = (n_samples, n_features)
    => External data set x
    exdataY :pandas DataFrame , shape = (n_samples,)
    => External data set y
    feature_set : list, set of features that make up model

    Returns
    -------
    None
    """
    def external_set(self):
        #print('Predicted external Y \n',expred)
        print('R2',r2_score(self.y, self.y_pred))
        print('R2_ext', r2_score(self.ey, self.ey_pred))
        print('RMSE', np.sqrt(mean_squared_error(self.y_pred,self.y)))
        print('coef', self.fit.coef_)
        print('intercept', self.fit.intercept_)
        plt.ylabel("Predicted Y")
        plt.xlabel("Actual Y")
        plt.scatter(self.y,self.y_pred,color=['gray'])
        plt.scatter(self.ey,self.ey_pred,color=['red'])
        plt.plot([self.y.min() , self.y.max()] , [[self.y.min()],[self.y.max()]],"black" )
        plt.show()


    """
    Y-scrambling plot for model validation

    Parameters
    ----------
    X_data : pandas DataFrame , shape = (n_samples, n_features)
    y_data : pandas DataFrame , shape = (n_samples,)
    feature_set : list, set of features that make up model
    n : int, number of iterations for y-scrambling, defaults to 1000

    Returns
    -------
    None
    """
    def y_scrambling(self, n=1000):

        """
        Returns Kxy value for a given covariance matrix

        Parameters
        ----------
        X : pandas dataframe, shape (n_samples, n_features)
        Y : pandas dataframe, shape (n_samples)

        Returns
        -------
        kxy : the Kxy value for the given covariance matrix or zero if the covariance matrix contains NaN values
        """
        def kxy_value(x, y):
            # calculate the covariance matrix
            cov = np.corrcoef(x, y, rowvar=False)
            if (np.isnan(np.sum(cov))):
                return 0
            else:
                # calculate the kxy value
                eig = np.linalg.eig(cov)[0]
                return sum(abs(eig / sum(eig) - 1 / len(eig))) / (2 * (len(eig) - 1) / len(eig))

        original_kxy = kxy_value(self.x, self.y)
        original_kxy_ex = kxy_value(self.ex, self.ey)
        original_r2 = r2_score(self.y, self.y_pred)
        original_q2 = r2_score(self.ey, self.ey_pred)
        rng = np.random.default_rng()
        scrambled = [ rng.permutation(self.y) for _ in range(n) ]
        scrambled_ex = [ rng.permutation(self.ey) for _ in range(n) ]
        # generate new models with scrambled y rows and calculate r2
        scr = [ LinearRegression().fit(self.x, sy) for sy in scrambled ]
        r2 = [ r2_score(self.y, s.predict(self.x)) for s in scr ]
        q2 = [ r2_score(self.ey, s.predict(self.ex)) for s in scr ]
        kxy = [ kxy_value(self.x, sy) for sy in scrambled ]
        kxy_ex = [ kxy_value(self.ex, sy) for sy in scrambled_ex ]
        # Set figure size and title
        plt.title("Y-Scrambling Plot")
        # Plot scrambled model scores
        plt.scatter(kxy, r2, c="yellow", label="Scr. R2")
        plt.scatter(kxy_ex, q2, c="red", label="Scr. Q2")
        # Plot the original scores
        plt.scatter([original_kxy], [original_r2], c="cyan", alpha=0.5, label="Mod. R2")
        plt.scatter([original_kxy_ex], [original_q2], c="blue", alpha=0.5, label="Mod. Q2")
        # Set axis labels
        plt.xlabel("Kxy")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        # Add legend and show plot
        plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')
        plt.show()
