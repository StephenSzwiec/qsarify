#-*- coding: utf-8 -*-
# Author: Stephen Szwiec
# Date: 2023-02-19
# Description: Classification Scoring Module
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

"""
Classification Scoring Module

This module provides summary information about Classification
"""

import numpy as np
from sklearn.metrics import accuracy_score

class ClassifierScore :
    """
    Provides summary information about Classification

    Parameters
    ----------
    y_data : pandas DataFrame , shape = (n_samples,)
    pred_y : pandas DataFrame , shape = (n_samples,)
    => predicted Y values as result of classification

    Sub functions
    -------
    score (self)
    tf_table(self)
    """

    def __init__ (self,y_data,pred_y) :
        """
        Initializes the classifer
        """
        # Initialize the variables
        self.y_data = y_data
        self.pred_y = pred_y
        self.real_y = [] #hash y_data
        # Hash the y_data
        for i in np.array(self.y_data) :
            self.real_y.append(i[0])

    def score (self) :
        """
        Calculate accuracy score
        Returns
        -------
        None
        """
        # Initialize the variables
        n = 0
        cnt = 0
        # Count the number of wrong predictions
        for i in np.array(self.real_y) :
            if i != self.pred_y[n] :
                cnt += 1
            n += 1
        print('Number of all :',n) #all data
        print('Number of worng :', cnt)
        print('AccuracyScore :',accuracy_score(self.real_y, self.pred_y))

    def tf_table(self) :
        """
        Calculate Precision & Recall
        Generates a confusion matrix

        Returns
        -------
        None
        """
        # Initialize the variables
        one = 0
        zero = 0
        n = 0
        cnt = 0
        realzero = 0
        realone = 0
        # Initialize the confusion matrix
        for i in np.array(self.y_data) :
            if i[0] == 0 :
                zero += 1
            if i[0] == 1 :
                one += 1
        # Count the number of wrong predictions
        for i in np.array(self.y_data):
            if i[0] != self.pred_y[n]:
                #print ('real',i[0],'///','pred',y_pred[n])
                if i[0] == 0 :
                    realzero += 1
                if i[0] == 1 :
                    realone += 1
                cnt +=1
            n += 1
        # Print the results
        print(('Number of 1 :',one))
        print('Number of 0 :',zero)
        print('True Positive(real 1 but pred 1) :',one-realone) #TP
        print('True Negative(real 0 but pred 0) :',zero-realzero) #TN
        print('False Positive(real 0 but pred 1) :',realzero) #FP
        print('False Negative(real 1 but pred 0) :',realone)  #FN
        print('Precision', (one-realone)/((one-realone)+realzero)) # TP / TP+FP
        print('Recall',(one-realone)/((one-realone)+realone)) #  TP / TP+FN
