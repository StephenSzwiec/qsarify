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

    def __init__(self, X_data, y_data, feature_set):
        self.X_data = X_data
        self.y_data = y_data
        self.feature_set = feature_set

    """
    Show training data prediction plot

    Returns
    -------
    None
    """
    def train_plot(self):
        x = self.X_data.loc[:,self.feature_set].values
        y = self.y_data.values
        pred_plotY = np.zeros_like(y)
        g_mlrr = LinearRegression()
        g_mlrr.fit(x,y)
        pred_plotY = g_mlrr.predict(x)
        plt.ylabel("Predicted Y")
        plt.xlabel("Actual Y")
        plt.scatter(y,pred_plotY,color=['gray'])
        plt.plot([y.min() , y.max()] , [[y.min()],[y.max()]],"black" )
        plt.show()

    """
    Show residuals of training plot
    Optionally, show residuals of test plot

    Returns
    -------
    None
    """
    def williams_plot(self, exdataX=None, exdataY=None) :
        if ((type(exdataX) is not None) ^ (type(exdataY) is not None)) :
            print("Please input both X and Y data")
            return
        test_set = bool(exdataX and exdataY)
        x = self.X_data.loc[:,self.feature_set].values
        y = self.y_data.values
        g_mlrr = LinearRegression()
        g_mlrr.fit(x, y)
        std = np.std(y)
        Y_pred = g_mlrr.predict(x)
        # Calculate the matrix of residuals using H = I - X(X'X)^-1X' (H is Hat matrix)
        H = Hin = x.T @ np.linalg.inv(x.T @ x) @ x.T
        residuals = res = (Y_pred - y) / std
        if (exdataX != None and exdataY != None):
            test_set = True
            xex = exdataX.loc[:self.feature_set].values
            yex = exdataY.values
            Y_pred_ex = g_mlrr.predict(xex)
            residuals_ex = (Y_pred_ex - yex) / std
            Hex =  (xex.T * np.linalg.inv(xex.T.dot(xex)).dot(xex.T)).sum(0)
            H.append(Hex)   # append Hex to H
            residuals.append(residuals_ex)  # append residuals of test data
        # The applicability domain is defined as the set of points where the residuals are less than 3 standard deviations
        hii = 3 * ( (len(self.feature_set) + 1) / len(Y_pred) )
        H_min = min(H-0.1)
        plt.axline(xy1=(H_min,0),slope=0)
        plt.axline(xy1=(H_min,3),slope=0,linestyle="--")
        plt.axline(xy1=(H_min,-3),slope=0,linestyle="--")
        plt.axline(xy1=(hii, -3.5), xy2=(hii, 3.5))
        plt.ylabel("Std. Residuals")
        plt.xlabel(F"Hat Values (h*={hii:.2f})")
        plt.ylim([-3.5,3.5])
        plt.scatter(Hin,res,color=['gray'])
        if(test_set) :
            plt.scatter(Hex,residuals_ex,color=['red'])
        plt.plot()
        plt.show()

    """
    Model information with result of multiple linear regression

    Returns
    -------
    None
    """
    def mlr(self) :
        x = self.X_data.loc[:,self.feature_set].values
        y = self.y_data.values
        mlr = LinearRegression()
        mlr.fit(x,y)
        print('Model features: ',self.feature_set)
        print('Coefficients: ', mlr.coef_)
        print('Intercept: ',mlr.intercept_)
        #MSE
        #print "MSE: %.3f" % np.mean((mlr.predict(x) - y) ** 2)
        #print mean_squared_error(mlr.predict(x),y)
        print("RMSE: %.6f" % np.sqrt(mean_squared_error(mlr.predict(x),y)))
        # Explained variance score
        print('R^2: %.6f' % mlr.score(x, y))


    """
    Show correlation of features

    Returns
    -------
    table
    """
    def features_table(self) :
        desc = DataFrame(self.X_data, columns=self.feature_set)
        result = pd.concat([desc, self.y_data], axis=1, join='inner')
        return result

    """
    Correlation coefficient of features table

    Returns
    -------
    table
    """
    def model_corr(self) :
        X = DataFrame(self.X_data, columns=self.feature_set)
        result = pd.concat([X, self.y_data], axis=1, join='inner')
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
    def external_set(self, exdataX,exdataY) :
        x = self.X_data.loc[:,self.feature_set].values
        y = self.y_data.values
        feature_set = self.feature_set
        exd = exdataX.loc[:,feature_set].values
        exdY = exdataY.values

        scalerr = MinMaxScaler()
        scalerr.fit(x)
        x_s = scalerr.transform(x)
        exd_s = scalerr.transform(exd)
        mlrm = LinearRegression()
        mlrm.fit(x_s,y)

        trainy= mlrm.predict(x_s)
        expred = mlrm.predict(exd_s)

        #print('Predicted external Y \n',expred)
        print('R2',mlrm.score(x_s,y))
        print('Q2',q2_score(y,trainy))
        print('RMSE',np.sqrt(mean_squared_error(trainy,y)))
        print('external R2',mlrm.score(exd_s,exdY))
        print('external Q2',q2_score(y, expred))
        print('coef',mlrm.coef_)
        print('intercept',mlrm.intercept_)
        plt.ylabel("Predicted Y")
        plt.xlabel("Actual Y")
        plt.scatter(y,trainy,color=['gray'])
        plt.scatter(exdY,expred,color=['red'])
        plt.plot([y.min() , y.max()] , [[y.min()],[y.max()]],"black" )
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
    def y_scrambling(self, exdataX, exdataY, n=1000):
        x = self.X_data.loc[:,self.feature_set].values
        y = self.y_data.values
        feature_set = self.feature_set
        mlr = LinearRegression()
        original_model = mlr.fit(x[feature_set], y)
        original_r2 = original_model.score(x[feature_set], y)
        original_q2 = q2_score(y, original_model.predict(x[feature_set]))
        scrambled = map(lambda i: y.sample(frac=1).reset_index(drop=True), range(n))
        r2 = [mlr.fit(x[feature_set], sy).score(x[feature_set], sy) for sy in scrambled]
        q2 = [q2_score(y, mlr.fit(x[feature_set], sy).predict(x[feature_set])) for sy in scrambled]
        # Set figure size and title
        plt.title("Y-Scrambling Plot")
        # Plot scrambled model scores
        plt.scatter(range(1, len(r2)+1), r2, c="yellow", label="Scrambled R2 Scores")
        plt.scatter(range(1, len(q2)+1), q2, c="red", label="Scrambled Q2 Scores")
        # Plot the original scores
        plt.scatter([0], [original_r2], c="cyan", label="Mod. R2")
        plt.scatter([0], [original_q2], c="blue", label="Mod. Q2")
        # Set axis labels
        plt.xlabel("Kxy")
        plt.ylabel("Score")
        # Add legend and show plot
        plt.legend()
        plt.show()
