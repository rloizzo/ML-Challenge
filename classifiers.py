#!/usr/bin/env python2.7

# Ryan Loizzo
# Nicholas Rocco
# Challenge Assignment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import RFECV, GenericUnivariateSelect

# read in data
c_train = pd.read_csv("classification_train.data", header=None)
c_test = pd.read_csv("classification_test.test", header=None)

class_row = len(c_train.columns) - 1
X = c_train.loc[:,0:class_row-1]
y = c_train.loc[:,class_row]

pos_count = neg_count = 0
for c in c_train[class_row]:
    if c == -1:
        neg_count += 1
    else:
        pos_count += 1

print "{} negative classes".format(neg_count)
print "{} positive classes".format(pos_count)
print ""

# determine which number of neighbors is optimal
#K = []
#accuracy = []
#for i in range(1,30,2):
#    K.append(i)
#    knn = KNeighborsClassifier(n_neighbors = i)
#    accuracy.append(np.mean(cross_val_score(knn,X,y=y,cv=10)))

#plt.plot(K,accuracy)
#plt.show()
# plot showed a clear elbow at K = 15

dt = DecisionTreeClassifier(criterion="entropy")
knn = KNeighborsClassifier(n_neighbors=15)
mlp = MLPClassifier(solver="sgd")

skf = StratifiedKFold(n_splits=10)

dt_accs = []
knn_accs = []
mlp_accs = []
dt_F1s = []
knn_F1s = []
mlp_F1s = []
for train_index, test_index in skf.split(X,y):
    X_train, X_test = X.as_matrix()[train_index], X.as_matrix()[test_index]
    y_train, y_test = y.as_matrix()[train_index], y.as_matrix()[test_index]
    dt.fit(X_train,y_train)
    knn.fit(X_train,y_train)
    mlp.fit(X_train,y_train)
    dt_pred = dt.predict(X_test)
    knn_pred = knn.predict(X_test)
    mlp_pred = mlp.predict(X_test)
    dt_accs.append(accuracy_score(y_test,dt_pred))
    dt_F1s.append(f1_score(y_test,dt_pred))
    knn_accs.append(accuracy_score(y_test,knn_pred))
    knn_F1s.append(f1_score(y_test,knn_pred))
    mlp_accs.append(accuracy_score(y_test,mlp_pred))
    mlp_F1s.append(f1_score(y_test,mlp_pred))

#dt_accs = cross_val_score(dt,X,y=y,cv=10)
#knn_accs = cross_val_score(knn,X,y=y,cv=10)
#mlp_accs = cross_val_score(mlp,X,y=y,cv=10)

print "Decision Tree accuracy mean: {} std: {}".format(np.mean(dt_accs), np.std(dt_accs))
print "15-Nearest Neighbors accuracy mean: {} std: {}".format(np.mean(knn_accs), np.std(knn_accs))
print "Neural Net accuracy mean: {} std: {}".format(np.mean(mlp_accs), np.std(mlp_accs))
print ""
print "Decision Tree F1 mean: {} std: ".format(np.mean(dt_F1s), np.std(dt_F1s))
print "15-Nearest Neighbors F1 mean: {} std: {}".format(np.mean(knn_F1s), np.std(knn_F1s))
print "Neural Net F1 mean: {} std: {}".format(np.mean(mlp_F1s), np.std(mlp_F1s))
print ""

for i in range(3):
    AD = [] #accuracy differences
    FD = [] # F1 differences
    if i == 0:
        print "Decision Tree - 15-Nearest Neighbors"
        for j in range(10):
            AD.append(dt_accs[j] - knn_accs[j])
            FD.append(dt_F1s[j] - knn_F1s[j])
    elif i == 1:
        print "Decision Tree - Neural Net"
        for j in range(10):
            AD.append(dt_accs[j] - mlp_accs[j])
            FD.append(dt_F1s[j] - mlp_F1s[j])
    else:
        print "15-Nearest Neighbors - Neural Net"
        for j in range(10):
            AD.append(knn_accs[j] - mlp_accs[j])
            FD.append(knn_F1s[j] - mlp_F1s[j])
    AD_mean = float(np.mean(AD))
    AD = map(lambda x: x - AD_mean, AD)
    AD = map(lambda x: pow(x,2), AD)
    S = float(pow(sum(AD) / 90., .5))
    T_prime = AD_mean / S
    print "accuracy t': {}".format(T_prime)
    FD_mean = float(np.mean(FD))
    FD = map(lambda x: x - FD_mean, FD)
    FD = map(lambda x: pow(x,2), FD)
    S = float(pow(sum(FD) / 90., .5))
    T_prime = FD_mean / S
    print "F1 t': {}".format(T_prime)
    print ""

#final predictions
test_X = c_test.loc[:,0:class_row-1]
dt_rfecv = RFECV(dt,step=1,cv=10)
dt_rfecv.fit(X,y)
final_predictions = dt_rfecv.predict(test_X)

neg_predictions = pos_predictions = 0
with open("classifier_predictions.data", "w") as f:
    for c in final_predictions:
        if c == -1:
            neg_predictions += 1
        else:
            pos_predictions += 1
        f.write(str(c))
        f.write('\n')
print "negative predictions: {}".format(neg_predictions)
print "positive predictions: {}".format(pos_predictions)      
