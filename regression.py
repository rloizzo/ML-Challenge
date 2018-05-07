#!/usr/bin/env python2.7

# Ryan Loizzo and Nick Rocco
# Challenge Assignment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Perceptron
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    data_train = pd.read_csv("regression_train.data", header=None)
    data_test = pd.read_csv("regression_test.test", header=None)

    class_row = len(data_train.columns)-1
    X = data_train.loc[:,0:class_row-1]
    y = data_train.loc[:,class_row]
   

    lr = LinearRegression()
    sgrd = SGDRegressor()
    p = Perceptron()

    skf = StratifiedKFold(n_splits=10)

    lr_mse = []
    sgrd_mse = []
    p_mse = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.as_matrix()[train_index], X.as_matrix()[test_index]
        y_train, y_test = y.as_matrix()[train_index], y.as_matrix()[test_index]
        lr.fit(X_train, y_train)
        sgrd.fit(X_train, y_train)
        p.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        sgrd_pred = sgrd.predict(X_test)
        p_pred = p.predict(X_test)
        lr_mse.append(mean_squared_error(y_test, lr_pred))
        sgrd_mse.append(mean_squared_error(y_test, sgrd_pred))
        p_mse.append(mean_squared_error(y_test, p_pred))

    print "LinearRegression sum squared error: {}".format(np.sum(lr_mse))
    print "Stochastic Gradient Descent sum squared error: {}".format(np.sum(sgrd_mse))
    print "Perceptron sum squared error: {}".format(np.sum(p_mse))
       

    test_X = data_test.loc[:,0:class_row-1]
    final_predictions = lr.predict(test_X)

    with open("regression_predictions.data", "w") as f:
        for i in final_predictions:
            f.write(str(i))
            f.write('\n') 
