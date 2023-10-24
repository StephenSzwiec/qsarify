#-*- coding: utf-8 -*-
# Author: Stephen Szwiec
# Date: 2023-02-19
# Description: Model Export Module
#
# Copyright (C) 2023 Stephen Szwiec
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
Model Export Module

This module contains the ModelExport class which is used to
summarize and export the results of a model. It contains
functions to plot the training data, print scoring metrics,
and show useful plots such as the Williams Plot.

"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.path import Path
from matplotlib.patches import Circle, RegularPolygon
from qsarify.qsar_scoring import q2_score, q2f_score, q2f3_score, ccc_score

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
    scores(self)
    model_corr(self)
    """

    def __init__(self, X_data, y_data, exXdata, exYdata, feature_set):
        """
        Initializes a ModelExport object

        Parameters
        ----------
        X_data : pandas DataFrame, shape = (n_samples, n_features)
        y_data : pandas DataFrame , shape = (n_samples,)
        exXdata : pandas DataFrame, shape = (n_samples, n_features)
        exYdata : pandas DataFrame , shape = (n_samples,)
        feature_set : list, set of features that make up model
        """
        self.x = X_data.loc[:,feature_set].values
        self.y = y_data.values
        self.ex = exXdata.loc[:, feature_set].values
        self.ey = exYdata.values
        self.feature_set = feature_set
        self.lr = LinearRegression()
        self.fit = self.lr.fit(self.x, self.y)
        self.y_pred = self.lr.predict(self.x)
        self.ey_pred = self.lr.predict(self.ex)

    def train_plot(self, interval=False):
        """
        Show training data prediction plot

        Parameters
        ----------
        interval : bool, optional, if True, shows 95% confidence interval

        Returns
        -------
        None
        """
        plt.ylabel("Predicted Y")
        plt.xlabel("Actual Y")
        plt.scatter(self.y,self.y_pred,color=['lightblue'])
        if interval:
            plt.errorbar(self.y, self.y_pred, yerr=1.96*np.sqrt(self.y_pred), fmt='o', color='blue', ecolor='blue', elinewidth=1, capsize=0)
        plt.plot([self.y.min() , self.y.max()] , [[self.y.min()],[self.y.max()]],"black" )
        plt.legend(["Actual Y", "Predicted Y"], loc="upper left")
        plt.show()

    def scores(self, verbose=False):
        """
        Print scoring metrics for the model

        Parameters
        ----------
        verbose : bool, optional, if True, prints all scoring metrics

        Returns
        -------
        None
        """
        scores = {}
        scores["RMSE"] = np.sqrt(mean_squared_error(self.y, self.y_pred))
        scores["RMSE External"] = np.sqrt(mean_squared_error(self.ey, self.ey_pred))
        scores["R2"] = r2_score(self.y, self.y_pred)
        scores["Q2"] = q2_score(self.ey, self.ey_pred)
        if verbose:
            scores["Q2F1"] = q2f_score(self.ey, self.ey_pred, np.mean(self.y))
            scores["Q2F2"] = q2f_score(self.ey, self.ey_pred, np.mean(self.ey))
            scores["Q2F3"] = q2f3_score(self.ey, self.ey_pred, len(self.y), len(self.ey))
            scores["CCC"] = ccc_score(np.append(self.y, self.ey), np.append(self.y_pred, self.ey_pred))
        for key, value in scores.items():
                print(key, ":", value)


    def williams_plot(self):
        """
        Show the Williams Plot for the data
        defines the applicability domain of the model
        and the points that are outliers in the data

        Returns
        -------
        None
        """
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
        plt.scatter(leverage_in,res,color=['lightblue'], label="Training Data")
        plt.scatter(leverage_ex,residuals_ex,color=['orange'], label="Test Data")
        plt.plot()
        plt.legend(loc="upper right")
        plt.show()

    def mlr(self):
        """
        Model information with result of multiple linear regression

        Returns
        -------
        None
        """
        print('Model features: ',self.feature_set)
        print('Coefficients: ', self.fit.coef_)
        print('Intercept: ',self.fit.intercept_)
        #MSE
        #print "MSE: %.3f" % np.mean((mlr.predict(x) - y) ** 2)
        #print mean_squared_error(mlr.predict(x),y)
        print("RMSE: %.6f" % np.sqrt(mean_squared_error(self.y_pred,self.y)))
        # Explained variance score
        print('R^2: %.6f' % r2_score(self.y, self.y_pred))

    def features_table(self) :
        """
        Show correlation of features to target

        Returns
        -------
        table
        """
        desc = DataFrame(self.x, columns=self.feature_set)
        target = DataFrame(self.y, columns=['response'])
        result = pd.concat([desc, target], axis=1, join='inner')
        return result


    def model_corr(self) :
        """
        Correlation coefficient of features table

        Returns
        -------
        table
        """
        X = DataFrame(self.x, columns=self.feature_set)
        fig = pd.plotting.scatter_matrix(X, alpha=0.7, diagonal='kde')
        for subaxis in fig:
            for ax in subaxis:
                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])
                ax.set_ylabel('')

    def external_set(self, verbose=False, interval=False):
        """
        Prediction external data set

        Parameters
        ----------
        verbose : bool, optional, if True show more scoring metrics. The default is False.
        interval : bool, optional, if True show 95-percent confidence interval. The default is False.


        Returns
        -------
        None
        """
        print('R2',r2_score(self.y, self.y_pred))
        print('Q2', r2_score(self.ey, self.ey_pred))
        print('RMSE', np.sqrt(mean_squared_error(self.y_pred,self.y)))
        print('coef', self.fit.coef_)
        print('intercept', self.fit.intercept_)
        if verbose:
            print('Q2F1', q2f_score(self.ey, self.ey_pred, np.mean(self.y)))
            print('Q2F2', q2f_score(self.ey, self.ey_pred, np.mean(self.ey)))
            print('Q2F3', q2f3_score(self.ey, self.ey_pred, len(self.y), len(self.ey)))
            print('CCC', ccc_score(self.ey, self.ey_pred))
        plt.ylabel("Predicted Y")
        plt.xlabel("Actual Y")
        plt.scatter(self.y,self.y_pred,color=['lightblue'])
        plt.scatter(self.ey,self.ey_pred,color=['orange'])
        if interval:
            plt.errorbar(self.y, self.y_pred, yerr=self.y_pred_std, fmt='o', color='lightblue', ecolor='lightgray', elinewidth=3, capsize=0)
            plt.errorbar(self.ey, self.ey_pred, yerr=self.ey_pred_std, fmt='o', color='orange', ecolor='lightgray', elinewidth=3, capsize=0)
        plt.plot([self.y.min() , self.y.max()] , [[self.y.min()],[self.y.max()]],"black" )
        plt.legend(["Training Data", "Test Data", "Predicted"], loc="upper left")
        plt.show()


    def y_scrambling(self, n=1000):
        """
        Y-scrambling plot for model validation

        Parameters
        ----------
        n : int, number of iterations for y-scrambling, defaults to 1000

        Returns
        -------
        None
        """
        def kxy_value(x, y):
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
        plt.legend()
        plt.show()

    def p_value(self):
        """
        P-value of features

        Returns
        -------
        table
        """
        X = DataFrame(self.x, columns=self.feature_set)
        p_values = f_regression(X, self.y)[1]
        p_values = np.round(p_values, 3)
        p_values = p_values.tolist()
        return p_values

    def p_value_plot(self):
        """
        P-value plot of features

        Returns
        -------
        None
        """
        p_values = self.p_value()
        plt.bar(self.feature_set, p_values)
        plt.ylabel("P-value")
        plt.xlabel("Features")
        plt.show()

    def p_value_table(self):
        """
        P-value table of features

        Returns
        -------
        table
        """
        p_values = self.p_value()
        return DataFrame(p_values, index=self.feature_set, columns=['p-value'])

    def feature_importance(self):
        """
        Feature importance of features

        Returns
        -------
        table
        """
        X = DataFrame(self.x, columns=self.feature_set)
        feature_importance = self.fit.coef_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        feature_importance = np.round(feature_importance, 2)
        feature_importance = feature_importance.tolist()
        return feature_importance

    def feature_importance_plot(self):
        """
        Feature importance plot of features

        Returns
        -------
        None
        """
        feature_importance = self.feature_importance()
        plt.bar(self.feature_set, feature_importance)
        plt.ylabel("Feature importance")
        plt.xlabel("Features")
        plt.show()

    def feature_importance_table(self):
        """
        Feature importance table of features

        Returns
        -------
        table
        """
        feature_importance = self.feature_importance()
        return DataFrame(feature_importance, index=self.feature_set, columns=['feature importance'])
