#!/usr/bin/python
# -*- coding: utf-8 -*

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

raw_data = pd.read_csv(r'./data/FEE.csv')
heads = raw_data.columns
# (0,1) transformation
scaler = MinMaxScaler(feature_range=(0, 1))
raw_data = pd.DataFrame(scaler.fit_transform(raw_data))
raw_data.columns = heads
# X,y
X = raw_data.drop(['OP_TIME', 'FEE'], axis=1)
poly = PolynomialFeatures(degree=4)
poly_X = pd.DataFrame(poly.fit_transform(raw_data))

y = raw_data['FEE']
kf = KFold(n_splits=10)
shuffle = ShuffleSplit(n_splits=1,test_size=.3)
for shuffle_train_index,shuffle_test_index in shuffle.split(poly_X):
    shuffle_X_train, shuffle_X_test = poly_X.ix[shuffle_train_index], poly_X.ix[shuffle_test_index]
    shuffle_y_train, shuffle_y_test = y.ix[shuffle_train_index], y.ix[shuffle_test_index]
    kf.get_n_splits(shuffle_X_train)
    regressions = ['Lasso', 'Ridge', 'GradientBoostingRegression']

    print("start trainning %s model..." % (regressions[0]))
    # nested 10-fold cross-validation
    alphas = []

    for train_index, test_index in kf.split(shuffle_X_train):
        # print("TRAIN:", train_index, "TEST:", test_index)
        print("start training...")
        X_train, X_validation = shuffle_X_train.ix[train_index], shuffle_X_train.ix[test_index]
        y_train, y_validation = shuffle_y_train.ix[train_index], shuffle_y_train.ix[test_index]
        lasso = linear_model.LassoCV(cv=10, max_iter=1500)
        lasso.fit(X_train, y_train)
        print ("best alpha is %s" %(lasso.alpha_))
        # lasso.score()
        print(lasso.mse_path_)
        alphas.append(lasso.alpha_)
    best_alpha = np.max(alphas)
    lasso = linear_model.Lasso(alpha=best_alpha, max_iter=1500)
    # lasso.fit
    # print scores
    # mean_score = np.mean(np.asarray(scores))
    # print ('mean_score is %f'% (mean_score))
    # print(lasso.predict(poly_X.ix[test_index]))

